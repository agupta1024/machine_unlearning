from unsloth import FastLanguageModel
import torch
from prepare_dataset.dataset import DatasetManager, SpanAwareCollator
from model.prepare_model import ModelBuilder
from torch.utils.data import DataLoader
from model.train import ModelTrainer
from analysis_scripts.evaluation import EvaluationManager
from analysis_scripts.memory_reports import AdapterReport
from transformers import TrainingArguments
import time
import gc
import wandb
import argparse, os
from accelerate import Accelerator
from datasets import concatenate_datasets
import warnings

import signal
import pdb

def debug_signal_handler(sig, frame):
    """
    Interrupts the current execution and opens a pdb prompt 
    at the exact line where the interrupt occurred.
    """
    print("\n[Interrupt] Pausing execution...")
    # This specifically targets the frame that was interrupted
    pdb.Pdb().set_trace(frame)

# Register the SIGINT (Ctrl+C) handler
signal.signal(signal.SIGINT, debug_signal_handler)

# Suppress FutureWarnings from transformers
warnings.filterwarnings("ignore", category=FutureWarning)

# CRITICAL: Enable Unsloth to return logits (required for evaluation)
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def get_path(model_name, run_name, i, oracle_path:bool = True, ga_loss:bool = False):
      if oracle_path:
            stage = 'oracle'
      else:
            stage = 'finetuned'
            if ga_loss:
                  stage = 'ga_loss_finetuned'
      oracle_dir = f"./Lora_model/TOFU/{model_name}_{stage}_{run_name}_{i}"
      return oracle_dir

def main():
            torch.cuda.empty_cache()
            parser = argparse.ArgumentParser(description="Machine Unlearning Ablation Runner")
            
            # Define the arguments that match your run_ablations script
            parser.add_argument("--run_name", type=str, default="experiment", help="Name for W&B logging")
            parser.add_argument("--proj_name", type=str, default="tofu-unlearning-metrics_ablations", help="Name for W&B Project")
            parser.add_argument("--lora_r", type=int, default=256, help="LoRA rank")
            parser.add_argument("--model", type=str, default='qwen', help="Base model key (qwen|llama1b|gemma|phi)")
            
            # Arguments for training oracle configuration
            parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate")
            parser.add_argument("--num_epochs", type=int, default=15, help="Number of training epochs")
            parser.add_argument("--num_forget_ex", type=int, default=10, help="Number of forget examples to use for training and evaluation")
            parser.add_argument("--oversample", type=int, default=50, help="Number of times to oversample the forget examples for training the oracle model")

            # Arguments for un-learning training configuration
            parser.add_argument("--ulr", type=float, default=3e-5, help="Learning rate")
            parser.add_argument("--lambda_forget", type=float, default=1.0, help="Weight for forget loss")
            parser.add_argument("--lambda_retain", type=float, default=0.7, help="Weight for retain loss")
            parser.add_argument("--lambda_gk", type=float, default=1.2, help="Weight for general knowledge loss")
            parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
            parser.add_argument("--mask", type=float, default=0.05, help="Percentage of saliency masking for sensitive token")
            parser.add_argument("--unlearn_run", type=str, default='unlearn', help="Unlearning run number")
            parser.add_argument("--train_with_ga_loss", action="store_true", help="Train model with gradient ascent loss")

            parser.add_argument("--train_oracle", action="store_true", help="Train oracle model to create checkpoints")
            parser.add_argument("--run_unlearn", action="store_true", help="Train model to unlearn PII")
            parser.add_argument("--run_eval", action="store_true", help="Evaluate unlearned model for evaluation")

            args = parser.parse_args()

            # Initialize Accelerator once so both normal python and accelerate launch work seamlessly.
            accelerator = Accelerator()
            device = accelerator.device
            if not accelerator.is_main_process:
                  os.environ["WANDB_MODE"] = "disabled"

            model_name = args.model
            models = {
                  'qwen' : 'unsloth/Qwen2.5-0.5b',
                  'llama1b' : 'unsloth/Llama-3.2-1B',
                  'llama1b-bnb-4bit' : 'unsloth/Llama-3.2-1B-bnb-4bit',
                  'gemma' :'unsloth/gemma-2-2b',
                  'phi': 'unsloth/Phi-3.5-mini-instruct',
                  }
            model_id = models.get(model_name, 'unsloth/Qwen2.5-0.5b')
            lora_rank=args.lora_r
            lora_config = {
                  'lora_alpha':2*lora_rank,
                  'r':lora_rank,
                  'target_modules':["q_proj", "k_proj", "v_proj", "o_proj", # Attention blocks
                                    "gate_proj", "up_proj", "down_proj",    # MLP / Feed-forward blocks
                                    "lm_head"
                                    ],
                  'lora_dropout':0.0,
                  'bias':"none"
                  }
            unlearning_rate = args.ulr
            learning_rate = args.lr
            num_epochs = args.epochs
            num_epochs_train_oracle = args.num_epochs
            lambda0=args.lambda_forget
            lambda1= args.lambda_retain
            lambda2 = args.lambda_gk
            saliency_mask = args.mask
            run_name = args.run_name
            unlearn_run = args.unlearn_run
            train_oracle = args.train_oracle
            run_unlearn = args.run_unlearn
            run_eval = args.run_eval

            accelerator.print('######## Load dataset ##########')
            per_device_train_batch_size = 2
            gradient_accumulation_steps = 4
            max_seq_length = 1024
            ds_manager_obj = DatasetManager(padding='longest',max_length=max_seq_length)
            forget_prompts_list, retain_prompts_list = ds_manager_obj.get_raw_data('TOFU')
            gk_prompts = ds_manager_obj.prepare_general_knowledge_prompts()
            gk_prompts = gk_prompts[:100]
            fluency_ques_list = ds_manager_obj.load_fluency_ques_bank()
            wandb_project = args.proj_name
      
            for i in range(len(forget_prompts_list)):
                  # if i==0:
                  #       continue
                  if i != 1:
                        continue
                  unlearning_loss = 'kl_div_forget+kl_div_retain+kl_div_gk'
                  if args.train_with_ga_loss:
                        unlearning_loss = 'ga_loss'
                  model_metadata = {
                        'model_name_or_path':model_id,
                        'learning_rate':learning_rate,
                        'unlearning_lr':unlearning_rate,
                        'num_train_epochs':num_epochs_train_oracle,
                        'num_unlr_epochs':num_epochs,
                        'unlearning_loss': unlearning_loss,
                  }
                  if run_eval:
                        # tuned_model_path = get_path(model_name, run_name, i, False, args.train_with_ga_loss)
                        # builder = ModelBuilder(accelerator, model_id=model_name, lora_config=lora_config,
                        #                         load_model='finetuned', oracle_path=tuned_model_path)

                        # model, tokenizer = builder.get_model_and_tokenizer()

                        oracle_path = get_path(model_name, run_name, i, True, args.train_with_ga_loss)
                        oracle_builder = ModelBuilder(accelerator, model_id=model_name, lora_config=lora_config,
                                                load_model='oracle', oracle_path=oracle_path)

                        oracle_model, tokenizer = oracle_builder.get_model_and_tokenizer()
                        tokenizer.padding_side = "left"
                  else:
                        # Prepare model and tokenizer
                        oracle_path = get_path(model_name, run_name, i, True, args.train_with_ga_loss)
                        if accelerator.is_main_process and not os.path.exists(oracle_path):
                              os.makedirs(oracle_path)
                        accelerator.wait_for_everyone()
                        if train_oracle:
                              load_model = 'base'
                        else:
                              load_model= 'oracle'
                        builder = ModelBuilder(accelerator, model_id=model_name, lora_config=lora_config,
                                                load_model=load_model, oracle_path=oracle_path)

                        model, tokenizer = builder.get_model_and_tokenizer()

                  accelerator.print(f"Model {model_id} prepared with LoRA adapter. Ready for training/ unlearning/ evaluation.")
                  # builder.print_trainable_parameters()
                  # AdapterReport(builder.model).calculate_memory()
                  accelerator.print(f"Loading the model on {device}")

                  if not run_eval:
                        if train_oracle:
                              wandb_run_name = f"{model_name}_{run_name}_dataset{i}"
                        else:
                              wandb_run_name = f"{model_name}_{unlearn_run}_dataset{i}"
                        if accelerator.is_main_process:
                              wandb.init(
                              project=wandb_project,
                              name=wandb_run_name,
                              config={
                                    "unlearning_rate": unlearning_rate,
                                    "num_train_epochs": num_epochs,
                                    "lora_rank": lora_rank,
                                    "forget_loss_wt-lamba0" :lambda0,
                                    "retain_loss_wt-lamba1" :lambda1,
                                    "gk_loss_lambda2": lambda2,
                                    "model_id": model_id,
                                    "saliency_mask": saliency_mask,
                                    },
                              reinit=True
                              )

                  # Prepare dataset and dataloader
                  accelerator.print("######## Tokenize dataset ##########")
                  ds_manager_obj.tokenizer = tokenizer
                  if args.num_forget_ex == -1:
                        num_forget_ex = len(forget_prompts_list[i])
                  else:
                        num_forget_ex = args.num_forget_ex
                  oversampled_forget_list = forget_prompts_list[i][:num_forget_ex] * args.oversample
                  forget_dataset = ds_manager_obj.tokenize_data(oversampled_forget_list)
                  retain_dataset = ds_manager_obj.tokenize_data(retain_prompts_list[i][:50])

                  if train_oracle:
                        oracle_dataset = concatenate_datasets([forget_dataset, retain_dataset])
                        sft_dataset = oracle_dataset.remove_columns(["pii_spans", "offset_mapping"])
                        sft_dataset = sft_dataset.shuffle(seed=42)

                  start_time = time.time()
                  retain_prompts = retain_prompts_list[i][:50]
                  custom_collator = SpanAwareCollator(tokenizer=tokenizer)
                  forget_dataloader = DataLoader(forget_dataset,
                                                batch_size=per_device_train_batch_size,
                                                shuffle=False, collate_fn=custom_collator,
                                                pin_memory=True,
                                                num_workers=4,
                                                prefetch_factor=2,
                                                persistent_workers=True)
                  retain_dataloader = DataLoader(retain_dataset,
                                                batch_size=per_device_train_batch_size,
                                                shuffle=True, collate_fn=custom_collator,
                                                pin_memory=True,
                                                num_workers=4,
                                                prefetch_factor=2,
                                                persistent_workers=True)
                  forget_dataloader, retain_dataloader = accelerator.prepare(forget_dataloader, retain_dataloader)
                  accelerator.print("######## Forget dataloader Ready ##########")
                  if run_unlearn or train_oracle:
                        tuned_model_path = get_path(model_name, run_name, i, False, args.train_with_ga_loss)
                        model_training_params = {
                              'forget_ds':forget_dataloader,
                              'retain_ds': retain_prompts,
                              'gk_prompts':gk_prompts,
                              'max_seq_length': max_seq_length,
                              'per_device_train_batch_size':per_device_train_batch_size,
                              'epochs': num_epochs,
                              'ulr':unlearning_rate,
                              'weight_decay':0.1,
                              'forget_loss_wt':lambda0,
                              'retain_loss_wt' : lambda1,
                              'gk_loss_wt' : lambda2,
                              'finetuned_model_save_dir': tuned_model_path
                        }
                        trainer_obj = ModelTrainer(accelerator, model_id, model_training_params)
                        trainer_obj.model = model
                        trainer_obj.tokenizer = tokenizer
                        # Create Oracle model
                        if train_oracle:
                              training_args = TrainingArguments(
                                    output_dir=oracle_path,
                                    per_device_train_batch_size=per_device_train_batch_size,
                                    gradient_accumulation_steps=gradient_accumulation_steps,
                                    max_grad_norm = 1.0,
                                    weight_decay = 0.05,
                                    lr_scheduler_type = "cosine",
                                    warmup_ratio = 0.03,
                                    seed = 42,
                                    learning_rate=learning_rate,
                                    num_train_epochs=num_epochs_train_oracle,
                                    save_strategy="steps",
                                    save_steps=20,
                                    logging_steps=10,
                                    save_total_limit=2,
                                    load_best_model_at_end = False,
                                    report_to="wandb",
                                    remove_unused_columns=False,
                                    fp16=False,
                                    bf16=True,
                                    save_only_model=True, 
                                    eval_accumulation_steps = 1,
                                    optim="adamw_8bit",
                                    torch_compile = False,            # Force HF not to compile
                                    dataloader_num_workers = 0,       # Force data loading onto the main thread
                                    dataloader_pin_memory = False,
                                    eval_strategy = "no",
                                    average_tokens_across_devices=False,
                                    )
                              trainer_obj.train(model, tokenizer, training_args, sft_dataset, oracle_path)
                              trainer_obj.wandb_init(wandb_project, model_name, run_name, i)
                              eval_obj = EvaluationManager(model, tokenizer, device, forget_dataloader, fluency_ques_list, gk_prompts, retain_dataloader)
                              test_prompts = forget_prompts_list[i][:num_forget_ex]
                              response_logs = eval_obj.check_model_health(test_prompts)
                              eval_obj.dump_results_to_json(model_metadata, response_logs, filename_prefix=f"oracle_eval_{model_name}_{run_name}_{i}")
                              perplexity = eval_obj._get_perplexity(gk_prompts)
                              print(f"Oracle model {model_id} {i} perplexity on GK prompts: {perplexity:.2f}")
                              print(f"Perform Evaluation on ORACLE {model_id} on TOFU {i}")
                              accelerator.print("########### ORACLE MODEL training is complete #####################")
                              breakpoint()
                        if run_unlearn:
                              if args.train_with_ga_loss:
                                    trainer_obj.train_model_with_ga_loss()
                              else:
                                    trainer_obj.train_model_with_lora()

                              eval_obj = EvaluationManager(model, tokenizer, device, forget_dataloader, fluency_ques_list,
                                                           gk_prompts, retain_dataloader)
                              test_prompts = forget_prompts_list[i][:num_forget_ex][:10]
                              response_logs = eval_obj.check_model_health(test_prompts)
                              base_response_logs = eval_obj.check_model_health(test_prompts, is_base_model=True)
                              eval_obj.dump_results_to_json(model_metadata, response_logs, base_response_logs, filename_prefix=f"unlearning_eval_{model_name}_{run_name}_{i}")
                              print(f"Perform Evaluation on finetuned {model_id} on TOFU {i}")
                              eval_obj.perform_final_audit()
                              model_predictions = [log[1] for log in response_logs]
                              target_answers = [log[2] for log in response_logs]
                              bert_score_model = eval_obj.get_semantic_metrics(model_predictions, target_answers)

                              base_predictions = [log[1] for log in base_response_logs]
                              bert_score_base = eval_obj.get_semantic_metrics(base_predictions, target_answers)
                              end_time = time.time()
                              accelerator.print(f"Training time: {end_time - start_time:.2f} seconds")
                              accelerator.wait_for_everyone()
                              accelerator.print(f"Model {model_id} training completed on {i}.")
                              breakpoint()
                  if run_eval and accelerator.is_main_process:
                        curated_test = ds_manager_obj.load_text_prompts(f"./data/TOFU/text_prompts_forget05.txt")
                        eval_obj._calculate_truth_ratio(curated_test, is_base_model=False)
                        # print(f"Loading finetuned model from {tuned_model_path} for evaluation...")
                        # eval_obj = EvaluationManager(model, tokenizer, device, forget_dataloader,
                        #                              fluency_ques_list, gk_prompts, retain_dataloader)
                        # prefix, target, safe_word = ds_manager_obj.load_text_prompts(f"./data/TOFU/text_prompts_forget05.txt")
                        # tr_f_list = []
                        # tr_b_list = []
                        # for p, t, s in zip(prefix, target, safe_word):
                        #       tr_f = eval_obj._calculate_truth_ratio(p, t, s, is_base_model=False)
                        #       tr_b = eval_obj._calculate_truth_ratio(p, t, s, is_base_model=True)
                        #       tr_f_list.append(tr_f)
                        #       tr_b_list.append(tr_b)

                        # truth_ratio_f = sum(tr_f_list)/len(tr_f_list)
                        # truth_ratio_b = sum(tr_b_list)/len(tr_b_list)
                        # print(f"Truth ratio for model {model_id} on TOFU {i}: {truth_ratio_f:.4f}")
                        # print(f"Truth ratio for base model on TOFU {i}: {truth_ratio_b:.4f}")

                        # prefix, target, safe_word = ds_manager_obj.load_text_prompts(f"./data/TOFU/indirect_prompts.txt")
                        # tr_f_list = []
                        # tr_b_list = []
                        # for p, t, s in zip(prefix, target, safe_word):
                        #       tr_f = eval_obj._calculate_truth_ratio(p, t, s, is_base_model=False)
                        #       tr_b = eval_obj._calculate_truth_ratio(p, t, s, is_base_model=True)
                        #       tr_f_list.append(tr_f)
                        #       tr_b_list.append(tr_b)

                        # truth_ratio_f = sum(tr_f_list)/len(tr_f_list)
                        # truth_ratio_b = sum(tr_b_list)/len(tr_b_list)
                        # print(f"Truth ratio for model {model_id} on TOFU {i}: {truth_ratio_f:.4f}")
                        # print(f"Truth ratio for base model on TOFU {i}: {truth_ratio_b:.4f}")

                        # test_prompts = prefix
                        test_prompts = forget_prompts_list[i][:num_forget_ex]
                        # response_logs = eval_obj.check_model_health(test_prompts)
                        # base_response_logs = eval_obj.check_model_health(test_prompts, is_base_model=True)
                        # # eval_obj.perform_final_audit()
                        # # eval_obj.dump_results_to_json(model_metadata, response_logs, base_response_logs, filename_prefix=f"unlearning_audit_{model_name}_{run_name}_{i}")
                        # print("Evaluation with exact prompts...")
                        # test_prompts = forget_prompts_list[i][:3]
                        # response_logs = eval_obj.check_model_health(test_prompts)
                        # base_response_logs = eval_obj.check_model_health(test_prompts, is_base_model=True)

                        print(f"Loading oracle model from {oracle_path} for evaluation...")
                        eval_obj_oracle = EvaluationManager(oracle_model, tokenizer, device, forget_dataloader,
                                                     fluency_ques_list, gk_prompts, retain_dataloader)
                        oracle_response_logs = eval_obj_oracle.check_model_health(test_prompts)
                        eval_obj_oracle.dump_results_to_json(model_metadata, oracle_response_logs, filename_prefix=f"oracle_post_ul_eval_{model_name}_{run_name}_{i}")
                        breakpoint()
                        model_predictions = [log[1] for log in response_logs]
                        target_answers = [log[2] for log in response_logs]
                        bert_score_model = eval_obj.get_semantic_metrics(model_predictions, target_answers)

                        base_predictions = [log[1] for log in base_response_logs]
                        bert_score_base = eval_obj.get_semantic_metrics(base_predictions, target_answers)

                        print("Semantic evaluation complete. Dumping results...")
                        print(f"Model: {model_id}, Dataset: {i} bert_score_model: {bert_score_model}, bert_score_base: {bert_score_base}")

                        accelerator.print(f"Perform Evaluation on {model_id} on TOFU {i}")
                        breakpoint()
                  del model, tokenizer, forget_dataloader
                  gc.collect()
                  torch.cuda.empty_cache()
                  
                  accelerator.wait_for_everyone()

                  if accelerator.is_main_process and wandb.run is not None:
                        wandb.finish()

if __name__ == '__main__':
      main()