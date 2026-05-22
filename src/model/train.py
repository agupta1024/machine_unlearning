"""This module contains the ModelTrainer class, which is responsible for training
both the oracle and student models, as well as implementing the unlearning process.
It includes a custom learning rate scheduler that combines linear warm-up
with cosine annealing decay. The trainer handles the entire training loop, including
loss calculation, optimization, and logging of training metrics."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements,super-with-arguments

import os
import math
from itertools import cycle
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorWithPadding
import wandb

class CosineAnnealingWithWarmupLR(_LRScheduler):
    """Custom learning rate scheduler that combines linear warm-up with cosine annealing decay."""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, eta_min=0.0, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate the learning rate for the current step."""
        current_step = max(0, self.last_epoch)
        if current_step < self.num_warmup_steps:
            # Warm-up phase
            lr_scale = float(current_step) / float(max(1, self.num_warmup_steps))
        else:
            # Cosine decay phase
            total_decay_steps = self.num_training_steps - self.num_warmup_steps
            progress = float(current_step - self.num_warmup_steps) / float(max(1,total_decay_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr_scale = (1 - self.eta_min) * cosine_decay + self.eta_min

        return [base_lr * lr_scale for base_lr in self.base_lrs]

class ModelTrainer:
    """Handles training of both the oracle and student models, as well as the unlearning process."""
    def __init__(self, model, tokenizer, ul_training_params=None):
        self.model = model
        self.tokenizer = tokenizer
        self.ul_training_params = ul_training_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_oracle(self, dataset, num_epochs, lr, outdir: str = "./oracle_adapter_hf"):
        """Train the oracle model adapter using the provided dataset."""
        if os.getenv('GITHUB_ACTIONS') == 'true':
            max_steps = 1
        else:
            max_steps = 500
        training_args = SFTConfig(
            output_dir=outdir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            logging_steps=10,
            max_steps=max_steps,
            optim="paged_adamw_8bit",
            fp16=False,
            bf16=True,
            save_strategy="no",
            max_seq_length=1024,
            dataset_text_field=None,
            packing=False,
            num_train_epochs=num_epochs,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=training_args,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )

        print("Starting standard Oracle training...")
        if os.getenv('GITHUB_ACTIONS') == 'true':
            print("Skipping model save in GitHub Actions environment")
            return
        trainer.train()

        if trainer.is_world_process_zero():
            trainer.model.save_pretrained(outdir)
            self.tokenizer.save_pretrained(outdir)
            print(f"Oracle successfully saved to {outdir}")

    def iter_tokenized_ds(self, dataset):
        """Tokenize the dataset and return an iterable dataloader."""
        token_ds = self.tokenizer(
            dataset,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=1024
        )
        token_list = [
            {k: v[i] for k, v in token_ds.items()}
            for i in range(len(token_ds['input_ids']))
        ]
        dataloader = torch.utils.data.DataLoader(
            token_list,
            batch_size=4,
            shuffle=True
        )
        return dataloader

    def train_student(self, npo_loss=False, ga_loss=False):
        """Train the student model using the unlearning training parameters."""
        for name, param in self.model.named_parameters():
            if "oracle" in name:
                param.requires_grad = False
            elif "student" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable_params_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params_num}")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if ga_loss:
            print("GA loss is enabled, proceeding with GA loss.")
            self.train_model_with_ga_loss()
            return
        lr = self.ul_training_params['ulr']
        num_epochs = self.ul_training_params['ul_epochs']
        forget_dataloader = self.ul_training_params['forget_ds']

        optimizer = AdamW(trainable_params, lr=lr,
                          betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)

        total_steps = num_epochs * len(forget_dataloader)
        warmup_steps = int(0.1 * total_steps)

        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            eta_min=0.1
        )
        gk_prompts = self.ul_training_params['gk_prompts']
        retain_prompts = self.ul_training_params['retain_ds']
        retain_dataloader = self.iter_tokenized_ds(retain_prompts)
        gk_dataloader = self.iter_tokenized_ds(gk_prompts)

        lmbd_0 = self.ul_training_params['forget_loss_wt']
        lmbd_1 = self.ul_training_params['retain_loss_wt']
        lmbd_2 = self.ul_training_params['gk_loss_wt']

        gk_iterator = cycle(gk_dataloader)
        retain_iterator = cycle(retain_dataloader)

        print("Starting Unlearning training...")
        if os.getenv('GITHUB_ACTIONS') == 'true':
            print("Skipping model save in GitHub Actions environment")
            return

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_forget_loss = 0.0
            total_gk_loss = 0.0
            total_retain_loss = 0.0
            progress_bar = tqdm(
                forget_dataloader,
                desc=f'Epoch {epoch+1}/{num_epochs}',
            )
            for batch_idx, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Debug: Print device info on first batch
                if batch_idx == 0:
                    print(f"\nDEVICE DEBUG - Epoch {epoch+1}, First Batch:")
                    print(f"  Batch input_ids device: {batch['input_ids'].device}")
                    print(f"  Batch attention_mask device: {batch['attention_mask'].device}")
                    print(f"  Model device: {next(self.model.parameters()).device}")

                retain_batch = next(retain_iterator)
                retain_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                 for k, v in retain_batch.items()}

                gk_batch = next(gk_iterator)
                gk_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in gk_batch.items()}

                ref_forget, ref_log_probs_retain, ref_log_probs_gk = self._compute_reference_logits(
                                                batch, retain_batch, gk_batch, npo_loss=npo_loss)
                self.model.set_adapter("student")
                self.model.train()
                outputs = self.model(**batch)
                logits = outputs.logits
                log_softmax_logits = F.log_softmax(logits.clone(), dim=-1)
                if npo_loss:
                    loss_forget = self.calculate_npo_loss(
                        student_logits=logits,
                        oracle_logits=ref_forget,
                        labels=batch['labels']
                    )
                else:
                    # KL Divergence Forget Loss
                    loss_forget = F.kl_div(
                                            input=log_softmax_logits,
                                            target=ref_forget,
                                            reduction='batchmean',
                                            log_target=True)
                del ref_forget, log_softmax_logits, outputs, logits
                torch.cuda.empty_cache()
                # Retain loss (using pre-computed logits)
                logits_retain = self.model(**retain_batch).logits
                log_softmax_logits_retain = F.log_softmax(logits_retain.clone(), dim=-1)
                loss_distill_retain = F.kl_div(
                                            log_softmax_logits_retain,
                                            ref_log_probs_retain,
                                            reduction='batchmean',
                                            log_target=True
                                        )
                del ref_log_probs_retain, log_softmax_logits_retain, logits_retain
                torch.cuda.empty_cache()
                # GK loss (using pre-computed logits)
                logits_gk = self.model(**gk_batch).logits
                log_softmax_logits_gk = F.log_softmax(logits_gk.clone(), dim=-1)
                loss_distill_gk = F.kl_div(
                                        log_softmax_logits_gk,
                                        ref_log_probs_gk,
                                        log_target=True,
                                        reduction='batchmean'
                                    )
                del ref_log_probs_gk, log_softmax_logits_gk, logits_gk
                torch.cuda.empty_cache()

                loss = lmbd_0 * loss_forget + lmbd_1 * loss_distill_retain + lmbd_2 *loss_distill_gk
                progress_bar.set_postfix(
                        loss=loss.detach().item(),
                        retain_loss=loss_distill_retain.detach().item(),
                        gk_loss=loss_distill_gk.detach().item(),
                        kl_loss=loss_forget.detach().item()
                    )

                optimizer.zero_grad()
                loss.backward()
                stats = {
                    'output': {'grad_sum': 0.0, 'params': 0},
                    'attention': {'grad_sum': 0.0, 'params': 0},
                    'mlp': {'grad_sum': 0.0, 'params': 0}
                }

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            mean_grad = param.grad.abs().mean().item()
                            numel = param.numel()
                            if "lm_head" in name or "embed_tokens" in name:
                                group = 'output'
                            elif "self_attn" in name:
                                group = 'attention'
                            elif "mlp" in name:
                                group = 'mlp'
                            else:
                                continue

                            stats[group]['grad_sum'] += mean_grad * numel
                            stats[group]['params'] += numel

                    current_lr = scheduler.get_last_lr()[0]
                    log_data = {}

                    for group, data in stats.items():
                        if data['params'] > 0:
                            avg_grad = data['grad_sum'] / data['params']
                            delta_w = avg_grad * current_lr
                            log_data[f'train/delta_W_{group}'] = delta_w
                            log_data[f'train/grad_mag_{group}'] = avg_grad

                            # print(f"Group {group} |ΔW|: {delta_w:.8f}")

                    if wandb.run is not None:
                        wandb.log(log_data)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_forget_loss += loss_forget.item()
                total_retain_loss += loss_distill_retain.item()
                total_gk_loss += loss_distill_gk.item()
                total_loss += loss.item()
                step = epoch * len(forget_dataloader) + batch_idx
                if step % 10 == 0:
                    self.visualize_distribution(batch, step)

            # Log loss values
            print(f"Epoch {epoch + 1}:")
            avg_loss = total_loss / len(forget_dataloader)
            wandb.log({
                'epoch': epoch + 1,
                'train/avg_loss' : avg_loss,
                'train/avg_forget_loss': total_forget_loss / len(forget_dataloader),
                'train/avg_retain_loss': total_retain_loss / len(forget_dataloader),
                'train/avg_gk_loss' : total_gk_loss / len(forget_dataloader),
            })
            print(f"Loss: {avg_loss:.6f}")

        save_dir = self.ul_training_params['finetuned_model_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_pretrained(save_dir, selected_adapters=["student"])

    def _compute_reference_logits(self, forget_batch, retain_batch, gk_batch, npo_loss=False):
        """Compute reference model logits without switching adapters in the main loop"""
        with torch.no_grad():
            self.model.set_adapter("oracle")
            self.model.eval()
            forget_out = self.model(
                input_ids=forget_batch['input_ids'],
                attention_mask=forget_batch['attention_mask']
            )
            forget_logits = forget_out.logits.float().detach().clone()
            if not npo_loss:
                pii_spans = forget_batch.get('pii_spans', [[]])
                for b_idx, spans in enumerate(pii_spans):
                    for start_tok, end_tok, safe_id in spans:
                        forget_logits[b_idx, start_tok-1:end_tok+1, :] -= 50.0
                        forget_logits[b_idx, start_tok-1, safe_id] = 10.0

                ref_forget = F.log_softmax(forget_logits, dim=-1).detach()
            else:
                ref_forget = forget_logits

            retain_out = self.model(**retain_batch)
            ref_log_retain = F.log_softmax(retain_out.logits.float(), dim=-1).detach()
            gk_out = self.model(**gk_batch)
            ref_log_gk = F.log_softmax(gk_out.logits.float(), dim=-1).detach()

            self.model.set_adapter("student")
            self.model.train()
        return ref_forget, ref_log_retain, ref_log_gk

    def visualize_distribution(self, batch, step):
        """Visualize the probability distribution over sensitive tokens
            and their safe substitutes.
        """
        prompt_text = self.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        print(f"--- Step {step} | Prompt: {prompt_text[:100]}... ---")
        batch_spans = batch.get('pii_spans', [[]])[0]
        with torch.no_grad():
            input_ids = batch['input_ids'][:1]
            attention_mask = batch['attention_mask'][:1]
            labels = batch['labels'][:1]

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[0, :-1, :]
            shift_labels = labels[0, 1:]

            if len(batch_spans) > 0:
                wandb_log_dict = {"train/step": step}
                for start_tok, end_tok, safe_id in batch_spans:
                    for pos in range(start_tok, end_tok + 1):
                        shifted_idx = pos - 1

                        if shifted_idx < 0 or shifted_idx >= shift_labels.size(0):
                            continue
                        t_id = shift_labels[shifted_idx].item()
                        if t_id < 0:
                            continue
                        try:
                            token_text = self.tokenizer.decode([t_id])
                        except OverflowError:
                            print(f"Skipping invalid token ID: {t_id}")
                            continue
                        probs = F.softmax(shift_logits[shifted_idx, :].float(), dim=-1)
                        target_prob = probs[t_id].item()

                        clean_name = token_text.strip() if token_text.strip() else f"ID_{t_id}"
                        metric_key = f"prob/sensitive_{clean_name}"
                        wandb_log_dict[metric_key] = target_prob
                        top_prob, top_idx = torch.topk(probs, 2)
                        best_alt_idx = top_idx[0] if top_idx[0].item() != t_id else top_idx[1]
                        alt_text = self.tokenizer.decode([best_alt_idx]).strip() or "space"

                        safe_word_prob = probs[safe_id].item()
                        safe_word = self.tokenizer.decode([safe_id])
                        metric_key = f"prob/safe_substitute_for_{clean_name}"
                        wandb_log_dict[metric_key] = safe_word_prob
                        print(f"Step {step} | Pos {pos} | Target:'{token_text}'({target_prob:.4f}) "
                              f"Safe Word:'{safe_word}'|({safe_word_prob:.4f})|Top Alt:'{alt_text}'\
                                ({top_prob[0].item():.4f}, {top_prob[1].item():.4f})")

                wandb.log(wandb_log_dict)
            else:
                print(f"Step {step} | No PII spans found in this sample visualization.")

    def get_batch_logps(self, logits, labels, ignore_index=-100):
        """
        Extracts the log probabilities of the true labels.
        (Assuming standard shift-by-one autoregressive modeling)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        safe_labels = shift_labels.clone()
        safe_labels[safe_labels == ignore_index] = 0
        target_log_probs = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
        mask = shift_labels != ignore_index

        seq_log_probs = (target_log_probs * mask).sum(dim=1)
        return seq_log_probs


    def calculate_npo_loss(self, student_logits, oracle_logits, labels, beta=0.1):
        """
        Calculates the Negative Preference Optimization loss.
        """
        log_probs_student = self.get_batch_logps(student_logits, labels)
        log_probs_oracle = self.get_batch_logps(oracle_logits, labels)

        log_ratio = log_probs_student - log_probs_oracle

        loss = -F.logsigmoid(-beta * log_ratio).mean()# pylint: disable=not-callable

        return loss

    def train_model_with_ga_loss(self):
        """Train the model using a gradient ascent loss to encourage forgetting."""
        lr = self.ul_training_params['ulr']
        num_epochs = self.ul_training_params['ul_epochs']
        forget_dataloader = self.ul_training_params['forget_ds']
        self.model.train()

        lr = self.ul_training_params['ulr']
        num_epochs = self.ul_training_params['ul_epochs']
        forget_dataloader = self.ul_training_params['forget_ds']

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=lr,
                          betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)

        total_steps = num_epochs * len(forget_dataloader)
        warmup_steps = int(0.1 * total_steps)

        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            eta_min=0.1
        )

        print("Starting Unlearning training...")
        if os.getenv('GITHUB_ACTIONS') == 'true':
            print("Skipping model save in GitHub Actions environment")
            return
        print('####### Model Unlearn Training With GA loss######')
        for epoch in range(num_epochs):
            total_loss = 0.0
            progress_bar = tqdm(
                forget_dataloader,
                desc=f'Epoch {epoch+1}/{num_epochs}',
            )
            for _, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                outputs = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
                standard_loss = outputs.loss
                ga_loss = -1.0 * standard_loss
                optimizer.zero_grad()
                ga_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += ga_loss.item()
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient Norm: {total_norm}")
            print(f"Epoch {epoch + 1}:")
            avg_loss = -1 * total_loss / len(forget_dataloader)
            wandb.log({
                'epoch': epoch + 1,
                'train/avg_loss' : avg_loss,
            })
            print(f"Loss: {avg_loss:.6f}")

        save_dir = self.ul_training_params['finetuned_model_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_pretrained(save_dir, selected_adapters=["student"])
