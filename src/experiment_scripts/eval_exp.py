from unsloth import FastLanguageModel
import torch
from prepare_dataset.dataset import DatasetManager
from model.prepare_model import ModelBuilder
from torch.utils.data import DataLoader, Dataset
from model.train import ModelTrainer
from analysis_scripts.evaluation import EvaluationManager
from analysis_scripts.memory_reports import AdapterReport
from transformers import TrainingArguments, DefaultDataCollator
import time
import gc
import wandb
import json
import argparse, os
from accelerate import Accelerator
from datasets import concatenate_datasets
import warnings

# Suppress FutureWarnings from transformers
warnings.filterwarnings("ignore", category=FutureWarning)

# CRITICAL: Enable Unsloth to return logits (required for evaluation)
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def get_path(model_name, run_name, i, oracle_path:bool = True):
    if oracle_path:
        stage = 'oracle'
    else:
        stage = 'finetuned'
    oracle_dir = f"./Lora_model/TOFU/{model_name}_{stage}_{run_name}_{i}_exp"
    return oracle_dir

def main():
    parser = argparse.ArgumentParser(description="Machine Unlearning Ablation Runner")
        
    # Define the arguments that match your run_ablations script
    parser.add_argument("--run_name", type=str, default="experiment", help="Name for W&B logging")
    parser.add_argument("--proj_name", type=str, default="tofu-unlearning-metrics_ablations", help="Name for W&B Project")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lambda_forget", type=float, default=1.0, help="Weight for forget loss")
    parser.add_argument("--lambda_retain", type=float, default=0.7, help="Weight for retain loss")
    parser.add_argument("--lambda_gk", type=float, default=1.2, help="Weight for general knowledge loss")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--mask", type=float, default=0.05, help="Percentage of saliency masking for sensitive token")
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.2-1B', help="Base model name")
    parser.add_argument("--train_oracle", type=bool, default=False, help="Train oracle model to create checkpoints")
    parser.add_argument("--run_unlearn", type=bool, default=False, help="Train model to unlearn PII")
    parser.add_argument("--run_eval", type=bool, default=False, help="Evaluate unlearned model for evaluation")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model

    models = {
                'qwen' : 'unsloth/Qwen2.5-0.5b',
                # 'qwen' : 'unsloth/qwen2.5-0.5b-bnb-4bit',
                'llama1b' : 'unsloth/llama-3.2-1b-bnb-4bit',
                # 'tiny_llama' : 'unsloth/tinyllama-chat-bnb-4bit',
                'tiny_llama' :'unsloth/llama-3.2-3b-instruct',
                'hf' : 'unsloth/SmolLM2-360M-bnb-4bit',
                }
    model_id = models.get(model_name, 'unsloth/qwen2.5-0.5b-bnb-4bit')
    lora_rank=args.lora_r
    lora_config = {
                'lora_alpha':2*lora_rank,
                'r':lora_rank,
                'target_modules':["q_proj", "k_proj", "v_proj", "o_proj", # Attention blocks
                            "gate_proj", "up_proj", "down_proj",    # MLP / Feed-forward blocks
                            ],
                'lora_dropout':0.0,
                'bias':"none"
                }
    
    learning_rate = args.lr
    num_epochs = args.epochs
    lambda0=args.lambda_forget
    lambda1= args.lambda_retain
    lambda2 = args.lambda_gk
    saliency_mask = args.mask
    run_name = args.run_name
    train_oracle = args.train_oracle
    run_unlearn = args.run_unlearn
    run_eval = args.run_eval

    print('######## Load dataset ##########')
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 4
    max_seq_length = 1024
    ds_manager_obj = DatasetManager(padding='longest',max_length=max_seq_length)
    forget_prompts_list, retain_prompts_list, sensitive_tokens = ds_manager_obj.get_raw_data('TOFU')
    gk_prompts = ds_manager_obj.prepare_general_knowledge_prompts()
    gk_prompts = gk_prompts[:100]
    fluency_ques_list = ds_manager_obj.load_fluency_ques_bank()
    wandb_project = args.proj_name
    
    # Initialize Accelerator for single GPU (can be expanded to multi-GPU with torch.nn.DataParallel if needed)
    accelerator = Accelerator()
   
    for i in range(len(forget_prompts_list)):
        if i == 0:
            continue

        if run_eval:
            tuned_model_path = get_path(model_name, run_name, i, False)
            builder = ModelBuilder(accelerator, model_id=model_name, lora_config=lora_config,
                                    load_model='finetuned', oracle_path=tuned_model_path)

            model, tokenizer = builder.get_model_and_tokenizer()
        else:
            pass
            # Prepare model and tokenizer
            # oracle_path = get_path(model_name, run_name, i, True)
            # if not os.path.exists(oracle_path):
            #     os.makedirs(oracle_path)
            # if train_oracle:
            #     load_model = 'base'
            # else:
            #     load_model= 'oracle'
            # builder = ModelBuilder(accelerator, model_id=model_name, lora_config=lora_config,
            #                         load_model=load_model, oracle_path=oracle_path)

            # model, tokenizer = builder.get_model_and_tokenizer()
        
        oracle_path = get_path(model_name, run_name, i, True)
        load_model= 'oracle'
        builder = ModelBuilder(accelerator, model_id=model_name, lora_config=lora_config,
                                        load_model=load_model, oracle_path=oracle_path)
        model, tokenizer = builder.get_model_and_tokenizer()
        tokenizer.padding_side = "left"
        # eval_obj = EvaluationManager(model, tokenizer, device, forget_prompts_list[i], fluency_ques_list, retain_prompts_list[i])
        # eval_obj.check_model_health()

        print(f"Model {model_id} prepared with LoRA adapter. Ready for training and unlearning.")
        builder.print_trainable_parameters()
        AdapterReport(builder.model).calculate_memory()
        model = model.to(device)
        print(f"Loading the model on {device}")

        # Initialize wandb AFTER model is ready
        # wandb.init(
        #     project=wandb_project,
        #     name=f"{model_name}_{run_name}_dataset{i}",
        #     config={
        #     "learning_rate": learning_rate,
        #     "num_train_epochs": num_epochs,
        #     "lora_rank": lora_rank,
        #     "forget_loss_wt-lamba0" :lambda0,
        #     "retain_loss_wt-lamba1" :lambda1,
        #     "gk_loss_lambda2": lambda2,
        #     "model_id": model_id,
        #     "saliency_mask": saliency_mask,
        #     },
        #     reinit=True
        # )

        # Prepare dataset and dataloader
        print("######## Tokenize dataset ##########")
        ds_manager_obj.tokenizer = tokenizer
        oversampled_forget_list = forget_prompts_list[i][:10] * 50
        forget_dataset = ds_manager_obj.tokenize_data(oversampled_forget_list)
        retain_dataset = ds_manager_obj.tokenize_data(retain_prompts_list[i][:50])

        if train_oracle:
            oracle_dataset = concatenate_datasets([forget_dataset, retain_dataset])
            oracle_dataset = oracle_dataset.shuffle(seed=42)
    
        start_time = time.time()
        retain_prompts = retain_prompts_list[i][:50]
        forget_dataloader = DataLoader(forget_dataset,
                                        batch_size=per_device_train_batch_size,
                                        shuffle=True, collate_fn=DefaultDataCollator(),
                                        pin_memory=True,
                                        num_workers=4,  # Parallelize data loading
                                        prefetch_factor=2,  # Buffer 2 batches ahead
                                        persistent_workers=True)  # Keep workers alive
        retain_dataloader = DataLoader(retain_dataset,
                                        batch_size=per_device_train_batch_size,
                                        shuffle=True, collate_fn=DefaultDataCollator(),
                                        pin_memory=True,
                                        num_workers=4,  # Parallelize data loading
                                        prefetch_factor=2,  # Buffer 2 batches ahead
                                        persistent_workers=True)  # Keep workers alive
        print("######## Forget dataloader Ready ##########")
        eval_obj = EvaluationManager(model, tokenizer, device, forget_dataloader, fluency_ques_list, retain_dataloader)
        test_prompts = forget_prompts_list[i][:5]
        # response_logs = eval_obj.check_model_health(test_prompts)

        base_response_logs = eval_obj.check_model_health(test_prompts)
        print(f"Perform Evaluation on {model_id} on TOFU {i}")
        # eval_obj.check_model_health()
        # eval_obj.check_model_health(is_base_model=True)
        breakpoint()


        if run_unlearn or train_oracle:
                tuned_model_path = get_path(model_name, run_name, i, False)
                model_training_params = {
                    'forget_ds':forget_dataloader,
                    'retain_ds': retain_prompts,
                    'gk_prompts':gk_prompts,
                    'sensitive_tokens':sensitive_tokens,
                    'max_seq_length': max_seq_length,
                    'per_device_train_batch_size':per_device_train_batch_size,
                    'epochs': num_epochs,
                    'lr':learning_rate,
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
                            learning_rate=2e-6,
                            num_train_epochs=15,
                            save_strategy="steps",
                            save_steps=10,
                            logging_steps=10,
                            report_to="wandb",
                            remove_unused_columns=False,
                            fp16=False,
                            bf16=True,
                            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
                            ddp_find_unused_parameters=False, # Better performance for LoRA
                            save_total_limit=1,
                            save_only_model=True, 
                            average_tokens_across_devices = False,
                            eval_accumulation_steps = 1,
                            max_grad_norm=0.3,
                            optim="adamw_8bit",
                            warmup_ratio=0.1,
                            weight_decay=0.05
                            )
                    # training_args, combined_dataset = accelerator.prepare(training_args, combined_dataset)
                    trainer_obj.train(model, tokenizer, training_args, oracle_dataset, oracle_path)
                    trainer_obj.wandb_init(wandb_project, model_name, run_name, i)

                    print("########### ORACLE MODEL training is complete #########################################")
                    print(f'forget set {forget_prompts_list[i][:5]}')
                    eval_obj.check_model_health()
                    breakpoint()
                else:
                    trainer_obj.train_model_with_lora(mask_percent = saliency_mask)
                    end_time = time.time()
                    print(f"Training time: {end_time - start_time:.2f} seconds")
                    # Ensure all processes synchronize
                    if torch.distributed.is_initialized():
                            torch.distributed.barrier()
                    print(f"Model {model_id} training completed on {i}.")
                    print(f'forget set {forget_prompts_list[i][:5]}')
                    eval_obj.check_model_health()
                    breakpoint()

        if run_eval:
                eval_obj = EvaluationManager(model, tokenizer, device, forget_dataloader, fluency_ques_list, retain_dataloader)
                print(f"Perform Evaluation on {model_id} on TOFU {i}")
                fluency_check_ans = eval_obj.perform_final_audit()
                # with open(f'./results/fluency_check_{model_id}_{i}.txt', "w", encoding="utf-8") as f:
                #       json.dump(fluency_check_ans, f, default=list, indent=4, ensure_ascii=False)
                
                # print(f"Fluency results successfully dumped to: './results/fluency_check.txt'")

        del model, tokenizer, forget_dataloader
        gc.collect()
        torch.cuda.empty_cache()
        
        # Synchronize again to ensure GPU memory cleanup is complete
        if torch.distributed.is_initialized():
                torch.distributed.barrier()

        wandb.finish()

if __name__ == '__main__':
    main()