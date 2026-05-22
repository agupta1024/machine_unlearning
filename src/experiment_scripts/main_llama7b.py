import torch
from prepare_dataset.dataset import DatasetManager, CustomDataCollator
from model.prepare_model import ModelBuilder
from torch.utils.data import DataLoader, Dataset
from model.train2 import ModelTrainer
from analysis_scripts.evaluation import EvaluationManager
from analysis_scripts.memory_reports import AdapterReport
import time
import gc
import wandb
import json

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [
            # 'Qwen/Qwen2.5-0.5B', 
            #   'distilbert/distilgpt2',
            #   'meta-llama/Llama-3.2-1B',
              "locuslab/tofu_ft_llama2-7b"
            #   'HuggingFaceTB/SmolLM2-360M'
            ]
    lora_rank=256
    lora_config = dict(
            r=lora_rank,
            lora_alpha=16,
            layers_to_transform=list(range(4, 8)),
            target_modules=".*(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj|c_attn|c_fc|c_proj).*",
            lora_dropout=0.05,
            bias="none"
        )
    learning_rate = 3e-5
    num_epochs = 4
    lambda1= 0.7
    lambda2 = 0.2
    wandb.init(
        project="tofu-unlearning",
        name=f"ga-unlearn-{models[0]}",
        config={
            "learning_rate": learning_rate,
            "num_train_epochs": num_epochs,
            "lora_rank": lora_rank,
            "retain_loss_wt-lamba1" :lambda1,
            "gk_loss_lambda2": lambda2,
            "model_id": models[0],
        }
    )
    print('######## Load dataset ##########')
    per_device_train_batch_size = 4
    max_seq_length = 1024  # Maximum truncation size
    ds_manager_obj = DatasetManager(padding='longest',max_length=max_seq_length)
    forget_prompts_list, retain_prompts_list, sensitive_tokens = ds_manager_obj.get_raw_data('TOFU')
    gk_prompts = ds_manager_obj.prepare_general_knowledge_prompts()
    gk_prompts = gk_prompts[:1000]
    fluency_ques_list = ds_manager_obj.load_fluency_ques_bank()

    for i in range(len(forget_prompts_list)):
        for model_id in models:
            # Prepare model and tokenizer
            builder = ModelBuilder(model_id=model_id, lora_config=lora_config, device=device)
            model, tokenizer = builder.get_model_and_tokenizer()
            # model.to(device)
            print(f"Model {model_id} prepared with LoRA adapter. Ready for training and unlearning.")
            builder.print_trainable_parameters()
            AdapterReport(builder.model).calculate_memory()
            base_model_builder = ModelBuilder(model_id=model_id, use_lora=False, device=device)
            base_model = base_model_builder.model
            # base_model.to(device)
            base_model.eval()

            # Prepare dataset and dataloader
            ds_manager_obj.tokenizer = tokenizer
            forget_dataset = ds_manager_obj.tokenize_data(forget_prompts_list[i])
            collator = CustomDataCollator(tokenizer=tokenizer, mask_question=False)

        
            start_time = time.time()
            retain_prompts = retain_prompts_list[i]
            forget_dataloader = DataLoader(forget_dataset,
                                           batch_size=per_device_train_batch_size,
                                           shuffle=True, collate_fn=collator)
            

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
                'retain_loss_wt' : lambda1,
                'gk_loss_wt' : lambda2
            }
            trainer_obj = ModelTrainer(model_id, model_training_params)
            trainer_obj.model = model
            trainer_obj.base_model = base_model
            trainer_obj.tokenizer = tokenizer
            trainer_obj.train_model_with_lora()
            end_time = time.time()
            print(f"Training time: {end_time - start_time:.2f} seconds")
            # Ensure all processes synchronize
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            print(f"Model {model_id} training completed on {i}.")

            eval_obj = EvaluationManager(model, base_model, tokenizer, forget_dataloader, fluency_ques_list)
            fluency_check_ans = eval_obj.run_full_suite(step_name=f"Evalute {model_id} on TOFU {i}")
            with open(f'./results/fluency_check_{model_id}_{i}.txt', "w", encoding="utf-8") as f:
                json.dump(fluency_check_ans, f, default=list, indent=4, ensure_ascii=False)
            
            print(f"Fluency results successfully dumped to: './results/fluency_check.txt'")

            del model, tokenizer, base_model, forget_dataloader, fluency_check_ans, eval_obj
            gc.collect()
            torch.cuda.empty_cache()
            
            # Synchronize again to ensure GPU memory cleanup is complete
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

if __name__ == '__main__':
    main()
