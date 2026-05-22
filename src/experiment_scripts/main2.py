import torch
from prepare_dataset.dataset import DatasetManager, SpanAwareCollator
from model.builder import ModelBuilder
from torch.utils.data import DataLoader
from train import ModelTrainer
from analysis_scripts.evaluation import EvaluationManager
from analysis_scripts.memory_reports import AdapterReport
from transformers import TrainingArguments
import time
import gc
import wandb
import argparse, os
from datasets import builder, builder, concatenate_datasets
import warnings
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def main():
    torch.cuda.empty_cache()

    model_id = "meta-llama/Llama-3.2-1B"
    learning_rate = 2e-4
    unlearning_rate = 2e-4
    num_epochs_train_oracle = 3
    num_epochs = 3


    ds_manager_obj = DatasetManager(padding='longest',max_length=1024)
    forget_prompts_list, retain_prompts_list = ds_manager_obj.get_raw_data('TOFU')
    gk_prompts = ds_manager_obj.prepare_general_knowledge_prompts()
    gk_prompts = gk_prompts[:100]
    fluency_ques_list = ds_manager_obj.load_fluency_ques_bank()
    wandb_project = 'tofu-llama_unlearning'
    for i in range(len(forget_prompts_list)):
        if i != 1:
            continue
        unlearning_loss = 'kl_div_forget+kl_div_retain+kl_div_gk'
        model_metadata = {
            'model_name_or_path':model_id,
            'learning_rate':learning_rate,
            'unlearning_lr':unlearning_rate,
            'num_train_epochs':num_epochs_train_oracle,
            'num_unlr_epochs':num_epochs,
            'unlearning_loss': unlearning_loss,
        }
        model_short = '8bit'
        builder = ModelBuilder(model_id=model_id, load_in_8bit=True)
        model, tokenizer = builder.get_model_and_tokenizer()
        
        wandb_run_name = f"Llama-3.2-1B_{model_short}_dataset{i}"
        wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "unlearning_rate": unlearning_rate,
            "num_train_epochs": num_epochs,
            "lora_rank": 64,
            "forget_loss_wt-lamba0" :1.0,
            "retain_loss_wt-lamba1" :0.7,
            "gk_loss_lambda2": 1.2,
            "model_id": model_id,
            },
        reinit=True
        )
        ds_manager_obj.tokenizer = tokenizer
        num_forget_ex = 20
        oversampled_forget_list = forget_prompts_list[i][:num_forget_ex] * 20
        forget_dataset = ds_manager_obj.tokenize_data(oversampled_forget_list)
        retain_dataset = ds_manager_obj.tokenize_data(retain_prompts_list[i][:50])

        oracle_dataset = concatenate_datasets([forget_dataset, retain_dataset])
        sft_dataset = oracle_dataset.remove_columns(["pii_spans", "offset_mapping"])
        sft_dataset = sft_dataset.shuffle(seed=42)

        start_time = time.time()
        retain_prompts = retain_prompts_list[i][:50]
        custom_collator = SpanAwareCollator(tokenizer=tokenizer)
        forget_dataloader = DataLoader(forget_dataset,
                                    batch_size=4,
                                    shuffle=False, collate_fn=custom_collator,
                                    pin_memory=True,
                                    num_workers=4,
                                    prefetch_factor=2,
                                    persistent_workers=True)
        retain_dataloader = DataLoader(retain_dataset,
                                    batch_size=4,
                                    shuffle=True, collate_fn=custom_collator,
                                    pin_memory=True,
                                    num_workers=4,
                                    prefetch_factor=2,
                                    persistent_workers=True)
        tuned_model_path = './finetuned_adapter'
        trainer_obj = ModelTrainer(model_id, model, tokenizer)
        trainer_obj.train_oracle(sft_dataset, num_epochs_train_oracle, learning_rate)

        eval_obj = EvaluationManager(model, tokenizer, 'cuda', forget_dataloader, fluency_ques_list, gk_prompts, retain_dataloader)
        test_prompts = forget_prompts_list[i][:num_forget_ex]
        response_logs = eval_obj.check_model_health(test_prompts)
        eval_obj.dump_results_to_json(model_metadata, response_logs, filename_prefix=f"oracle_eval_Llama-3.2-1B_{model_short}_{i}")
        perplexity = eval_obj._get_perplexity(gk_prompts)
        print(f"Oracle model {model_id} {i} perplexity on GK prompts: {perplexity:.2f}")
        print(f"Perform Evaluation on ORACLE {model_id} on TOFU {i}")
        breakpoint()

if __name__ == "__main__":
    main()