
import numpy as np
from transformers import Trainer
from model.prepare_model import UnlearningDataCollator, RefusalDataCollator
from analysis_scripts.visualize_gradients import GradientVisualizer
from analysis_scripts.memory_reports import AdapterReport
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from analysis_scripts.evaluation import EvaluationManager
from prepare_dataset.dataset import UnlearningDataset
import wandb
import os
import json
from datetime import datetime
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelTrainer:
    def __init__(self, model, training_args, dataset_config=None, kwargs=None):
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.training_args = training_args
        self.dataset_config = dataset_config
        self.kwargs = kwargs
        self.prepare_evaluator()

    def prepare_evaluator(self):
        # Prepare datasets for evaluation
        forget_ds_config = self.dataset_config.copy()
        forget_ds_config["train_set"] = f"forget{self.dataset_config['forget_percentage']:02d}"
        retain_ds_config = self.dataset_config.copy()
        retain_ds_config["train_set"] = f"retain{100 - self.dataset_config['forget_percentage']:02d}"
        forget_path = self.dataset_paths(forget_ds_config)
        retain_path = self.dataset_paths(retain_ds_config)
        self.forget_ds = UnlearningDataset(forget_path, self.tokenizer)
        self.retain_ds = UnlearningDataset(retain_path, self.tokenizer)
        self.evaluator = EvaluationManager(self.model, self.tokenizer, self.forget_ds._get_raw_dataset(),
                                           self.retain_ds._get_raw_dataset())

    def dataset_paths(self, config=None):
        dataset_name = config["name"]
        data_dir = config.get("data_dir", "./data")
        set_name = config.get("train_set", "full")
        ext_name = config.get("ext", "")
        return f"{data_dir}/{dataset_name}/{set_name}{ext_name}.json"

    def train(self, load_pretrained=False):
        if load_pretrained:
            print("Loading pretrained poisoned model for training continuation...")
            self.load_unlearning_model(self.kwargs["model_name_or_path"])
            self.baseline_stats = self.evaluator.run_full_suite(step_name="Post-Injection Baseline")
            return

        collator = UnlearningDataCollator(tokenizer=self.tokenizer, mask_question=True)
        dataset_path = self.dataset_paths(self.dataset_config)
        ds = UnlearningDataset(dataset_path, self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=ds,
            data_collator=collator
        )

        trainer.train()
        poisoned_path = "./models/poisoned_lora"
        trainer.model.save_pretrained(poisoned_path)
        self.tokenizer.save_pretrained(poisoned_path)

        print(f"✅ Poisoned LoRA weights saved to {poisoned_path}")
        self.baseline_stats = self.evaluator.run_full_suite(step_name="Post-Injection Baseline")

    def load_unlearning_model(self, model_id):
        base_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        dtype=torch.bfloat16,
                        device_map={"": self.kwargs["device"]},
                    )
        poisoned_path = "./models/poisoned_lora"
        self.model = PeftModel.from_pretrained(
            base_model, 
            poisoned_path,
        ).to(self.kwargs["device"])
        self.tokenizer = AutoTokenizer.from_pretrained(poisoned_path)
        return self.model

    def load_reference_model(self):
        base_model = self.model.get_base_model()
        poisoned_path = "./models/poisoned_lora"
        # 2. Attach the poisoned weights as a SECOND model (the Reference)
        self.ref_model = PeftModel.from_pretrained(
            base_model, 
            poisoned_path
        ).to(self.model.device)

        # Freeze the reference model's weights to prevent any accidental updates during unlearning
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        return self.ref_model

    def get_saliency_map_for_one(self, sample):
        self.model.eval()

        input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(self.model.device)
        attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(self.model.device)
        
        # FORCE labels to be a copy of input_ids if they are missing or -100
        # In Causal LM, labels must match input_ids for the model to calculate CrossEntropyLoss
        labels = input_ids.clone() 

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        # Forward pass
        self.model.zero_grad()
        outputs = self.model(**batch)
        loss = outputs.loss
            
        if loss is None:
            print(f"Loss is None. Model type: {type(self.model)}")
            return None

        loss.backward()
        AdapterReport(self.model).calculate_memory()
        AdapterReport(self.model).get_lora_params()

        visualizer = GradientVisualizer(self.model)
        visualizer.visualize_lora_saliency(self.model, layer_idx=0)
        visualizer.visualize_lora_saliency(self.model, layer_idx=10) # Adjust based on total layers
        visualizer.visualize_layer_gradients("layers.0.self_attn.q_proj")

        # Extract saliency (Absolute Gradient Magnitude)
        saliency = {}
        for name, param in self.model.named_parameters():
            if "lora" in name and param.grad is not None:
                saliency[name] = param.grad.abs().detach().clone()
                
        return saliency

    def get_aggregate_saliency_batched(self, batch_size=4):
        self.model.eval()
        collator = UnlearningDataCollator(tokenizer=self.tokenizer, mask_question=True)
        # collator = RefusalDataCollator(tokenizer=self.tokenizer)
        agg_saliency = {name: torch.zeros_like(p.data)
                        for name, p in self.model.named_parameters() if "lora" in name}
        loader = DataLoader(self.forget_ds, batch_size=batch_size, collate_fn=collator)

        print(f"Aggregating saliency in batches of {batch_size}...")
        for batch in loader:
            self.model.zero_grad()

            inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()

            # Accumulate gradients
            for name, param in self.model.named_parameters():
                if "lora" in name and param.grad is not None:
                    agg_saliency[name] += param.grad.abs().detach()

        AdapterReport(self.model).calculate_memory()
        AdapterReport(self.model).get_lora_params()

        visualizer = GradientVisualizer(self.model)
        visualizer.visualize_lora_saliency(self.model, agg_saliency, layer_idx=0)
        visualizer.visualize_lora_saliency(self.model, agg_saliency, layer_idx=10)
        visualizer.visualize_lora_saliency(self.model, agg_saliency, layer_idx=15)
        visualizer.visualize_lora_saliency(self.model, agg_saliency, layer_idx=18)
        visualizer.visualize_lora_saliency(self.model, agg_saliency, layer_idx=23)
        return agg_saliency

    def create_surgical_mask(self, saliency, mask_percentile=0.90):
        mask = {}
        # Flatten all saliency values to find the global threshold
        all_values = torch.cat([v.flatten() for v in saliency.values()])
        threshold = torch.quantile(all_values, mask_percentile)
        print(f"Saliency Threshold for top {100 - mask_percentile*100:.1f}%: {threshold:.6f}")
        
        for name, grad_abs in saliency.items():
            # 1 if importance > threshold, else 0
            mask[name] = (grad_abs >= threshold).float()
        
        # AdapterReport(self.model).check_mask_alignment(mask)
        return mask, threshold
    
    def calculate_ref_logits(self, inputs):
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            logits = ref_outputs.logits
            return logits

    def run_unlearning(self, mask, lr=5e-5, num_epochs=15, kl_weight=0.1):
        """
        Performs Gradient Ascent strictly on the masked LoRA weights.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        # Refernce model
        # self.ref_model = self.load_reference_model()
        # collator = UnlearningDataCollator(tokenizer=self.tokenizer, mask_question=True)
        collator = RefusalDataCollator(tokenizer=self.tokenizer)
        loader = DataLoader(self.forget_ds, batch_size=2, collate_fn=collator)

        print(f"Starting Surgical Unlearning for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                inputs = {k: v.to(self.model.device) for k, v in batch.items()}

                # Forward Pass
                outputs = self.model(**inputs)
                ga_loss = outputs.loss
                # logits = outputs.logits
                # total_loss = ga_loss
                # total_loss.backward()

                # ref_logits = self.calculate_ref_logits(inputs)
                # with torch.no_grad():
                #     self.ref_model.eval() # Ensure eval mode
                #     ref_outputs = self.ref_model(**inputs)
                #     ref_logits = ref_outputs.logits
                # kl_loss = torch.nn.functional.kl_div(
                #                 torch.nn.functional.log_softmax(logits, dim=-1),
                #                 torch.nn.functional.softmax(ref_logits, dim=-1),
                #                 reduction='batchmean'
                #             )
                # # anchor_loss = torch.nn.functional.mse_loss(logits, ref_logits)
                # total_loss = ga_loss + kl_weight * kl_loss
                total_loss = ga_loss
                total_loss.backward()
                # if total_loss.grad_fn is None:
                #     print(f"GA Grad: {ga_loss.grad_fn}, KL Grad: {kl_loss.grad_fn}")
                #     raise RuntimeError("Loss still detached!")
                # APPLY THE SURGICAL MASK - manually zero out gradients for any weight NOT in top-saliency mask.
                grad_norms = []
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if name in mask:
                            # if name in mask and any(proj in name for proj in ["gate_proj", "up_proj", "down_proj"]):
                                param.grad *= mask[name].to(param.grad.device)
                                grad_norms.append(param.grad.norm().item())
                            else:
                                param.grad.zero_()
                        else:
                            continue
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
                # optimizer.zero_grad()
                epoch_loss += outputs.loss.item()
                # print(f"Batch Loss: {outputs.loss.item():.4f}")
                avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0
                # print(f">>> Masked Gradient Magnitude: {avg_grad:.8f}")
                if avg_grad == 0:
                    print("WARNING: All gradients are ZERO. Your mask is blocking everything!")
            epoch_eval_stats = self.evaluator.run_full_suite(
                                                    step_name=f"Post-Unlearning Epoch {epoch+1}")
            avg_forget_loss = epoch_loss / len(loader)
            print(f"Average Forget Loss (Lower is 'more forgotten'): {avg_forget_loss:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train/forget_loss": avg_forget_loss,
                "train/forget_accuracy": epoch_eval_stats['forget_acc'],
                "train/retain_accuracy": epoch_eval_stats['retain_acc'],
                "train/perplexity": epoch_eval_stats['ppl'],
                "train/pii_z_score": epoch_eval_stats['z_score']
            })
        print("Unlearning complete.")
        self.final_stats = self.evaluator.run_full_suite(step_name="Final Post-Unlearning Audit")
        forget_table = wandb.Table(columns=["Prompt", "Prediction", "Target", "Status"],
                                       data=self.final_stats['forget_log'])
        retain_table = wandb.Table(columns=["Prompt", "Prediction", "Target", "Status"],
                                       data=self.final_stats['retain_log'])
        
        wandb.log({
                "final/forget_accuracy": self.final_stats['forget_acc'],
                "final/retain_accuracy": self.final_stats['retain_acc'],
                "final/perplexity": self.final_stats['ppl'],
                "final/pii_z_score": self.final_stats['z_score'],
                "final/forget_log": forget_table,
                "final/retain_log": retain_table
            })
        self.dump_results_to_json(self.final_stats['forget_log'], self.final_stats['retain_log'])
        return self.final_stats

    def dump_results_to_json(self, forget_logs, retain_logs, folder_path="./results"):
        # Ensure the directory exists
        os.makedirs(folder_path, exist_ok=True)
        
        # Create a unique filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unlearning_audit_{timestamp}.json"
        full_path = os.path.join(folder_path, filename)
        
        # Structure the data
        report_data = {
            "metadata": {
                "model_name": self.kwargs["model_name_or_path"],
                "learning_rate": self.training_args.learning_rate,
                "unlearning_lr": self.kwargs["unlearning_lr"],
                "num_epochs": self.training_args.num_train_epochs,
                "num_unlr_epochs": self.kwargs["num_unlr_epochs"],
                "unlearning_method": "Gradient Ascent", # Update this dynamically
                "timestamp": timestamp
            },
            "forget_set_results": [
                {"prompt": r[0], "prediction": r[1], "target": r[2], "correct": r[3]} 
                for r in forget_logs
            ],
            "retain_set_results": [
                {"prompt": r[0], "prediction": r[1], "target": r[2], "correct": r[3]} 
                for r in retain_logs
            ]
        }
        
        # Save to file
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, default=list, indent=4, ensure_ascii=False)
        
        print(f"💾 Audit results successfully dumped to: {full_path}")

    def calculate_change(self, before, after, metric_name):
        diff = after - before
        symbol = "+" if diff >= 0 else ""
        return f"{metric_name}: {before:.2f} -> {after:.2f} ({symbol}{diff:.2f})"

    def compare_model_stats(self):
        print("\n--- Comparative Analysis ---")
        print(self.calculate_change(self.baseline_stats['forget_acc'], self.final_stats['forget_acc'], "Forget Accuracy (Target ↓)"))
        print(self.calculate_change(self.baseline_stats['retain_acc'], self.final_stats['retain_acc'], "Retain Accuracy (Target ↔)"))
        print(self.calculate_change(self.baseline_stats['ppl'], self.final_stats['ppl'], "Model Perplexity (Target ↔)"))
        print(self.calculate_change(self.baseline_stats['z_score'], self.final_stats['z_score'], "PII Z-Score (Target ↓)"))

        # Final Sanity Check (Fluency)
        self.evaluator._check_fluency()
    
    def check_fluency(self):
        self.evaluator._check_fluency()

    def run_mia_audit(self, forget_loader, unseen_loader):
        self.model.eval()
        forget_losses = [self.model(**b).loss.item() for b in forget_loader]
        unseen_losses = [self.model(**b).loss.item() for b in unseen_loader]
        
        # If the distributions overlap perfectly, the MIA score is low (Success!)
        # If forget_losses are much lower, the model still 'remembers' the data.
        return np.mean(forget_losses), np.mean(unseen_losses)