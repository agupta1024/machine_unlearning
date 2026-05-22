"""Main entrypoint for TOFU unlearning experiments."""

# pylint: disable=too-many-locals,too-many-statements

import argparse
import os

from datasets import concatenate_datasets
import torch
from torch.utils.data import DataLoader
import wandb

from src.analysis.evaluation import EvaluationManager
from src.dataset.dataset import DatasetManager, SpanAwareCollator
from src.model.builder import ModelBuilder
from src.model.train import ModelTrainer

def main():
    """Run training, unlearning, and evaluation for one configured dataset split."""
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Machine Unlearning Runner")

    # Define the arguments that match your run_ablations script
    parser.add_argument(
        "--run_name", type=str, default="experiment", help="Name for W&B logging"
    )
    parser.add_argument(
        "--proj_name", type=str, default="tofu-unlearning", help="Name for W&B Project"
    )
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")

    # Arguments for training oracle configuration
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Whether to load the model in 4-bit precision",
    )

    # Arguments for un-learning training configuration
    parser.add_argument("--ulr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--lambda_forget", type=float, default=1.0, help="Weight for forget loss")
    parser.add_argument("--lambda_retain", type=float, default=0.7, help="Weight for retain loss")
    parser.add_argument(
        "--lambda_gk", type=float, default=1.2, help="Weight for general knowledge loss"
    )
    parser.add_argument("--ul_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--train_with_ga_loss",
        action="store_true",
        help="Train model with gradient ascent loss",
    )
    parser.add_argument(
        "--train_with_npo_loss",
        action="store_true",
        help="Train model with npo loss",
    )
    parser.add_argument(
        "--load_pretrained",
        action="store_true",
        help="Load pretrained oracle model"
    )
    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default='oracle',
        help="Load pretrained oracle/student model"
    )
    parser.add_argument("--run_unlearn", action="store_true", help="Train model to unlearn PII")
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Evaluate unlearned model for evaluation",
    )

    args = parser.parse_args()

    model_id = "meta-llama/Llama-3.2-1B"
    learning_rate = args.lr
    unlearning_rate = args.ulr
    num_epochs_train_oracle = args.num_epochs
    num_epochs_train_student = args.ul_epochs

    if os.getenv("GITHUB_ACTIONS") == "true":
        num_epochs_train_oracle = 1
        num_epochs_train_student = 1

    if os.getenv("GITHUB_ACTIONS") == "true":
        os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_MODE"] = "disabled"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_manager_obj = DatasetManager(padding="longest", max_length=1024)
    print("Fetching datasets...")
    forget_prompts_list, retain_prompts_list = ds_manager_obj.get_raw_data("TOFU")
    gk_prompts = ds_manager_obj.prepare_general_knowledge_prompts()
    gk_prompts = gk_prompts[:100]
    fluency_ques_list = ds_manager_obj.load_fluency_ques_bank()
    wandb_project = "tofu-llama_unlearning"
    for i, forget_prompts in enumerate(forget_prompts_list):
        if i != 1:
            continue
        if args.load_in_4bit:
            model_short = "4bit"
        else:
            model_short = "8bit"
        print(f"Loading model in {model_short} precision...")
        if args.train_with_npo_loss:
            tuned_model_path = "./finetuned_adapter/llama-3.2-1B-" + model_short + f"-{i}-npo_loss"
        else:
            tuned_model_path = "./finetuned_adapter/llama-3.2-1B-" + model_short + f"-{i}"
        tuned_model_path = "./Lora_model/TOFU/llama1b_ga_loss_finetuned_r64_3e-5_20ex_1"
        oracle_path= "./oracle_adapter_hf"
        if args.load_pretrained:
            builder = ModelBuilder(load_in_4bit=args.load_in_4bit, load_model=args.load_pretrained_model,
                                   oracle_path=oracle_path, tuned_model_path=tuned_model_path)
        else:
            builder = ModelBuilder(load_in_4bit=args.load_in_4bit)
        model, tokenizer = builder.get_model_and_tokenizer()

        wandb_run_name = f"Llama-3.2-1B_{model_short}_dataset{i}"
        loss = "kl_div_retain+kl_div_gk"
        if args.train_with_ga_loss:
            loss += "+ga_loss"
        elif args.train_with_npo_loss:
            loss += "+npo_loss"
        else:
            loss += "+kl_div_forget"
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_id": model_id,
                "model_short": model_short,
                "unlearning_rate": unlearning_rate,
                "num_train_epochs": num_epochs_train_oracle,
                "num_unlr_epochs": num_epochs_train_student,
                "lora_rank": args.lora_r,
                "forget_loss_wt-lamba0": args.lambda_forget,
                "retain_loss_wt-lamba1": args.lambda_retain,
                "gk_loss_lambda2": args.lambda_gk,
                "unlearning_loss": loss,
            },
            reinit=True,
        )
        ds_manager_obj.tokenizer = tokenizer
        if os.getenv("GITHUB_ACTIONS") == "true":
            print("Running in GitHub Actions environment - using smaller dataset for testing")
            num_forget_ex = 2
            num_retain_ex = 2
            oversampling_factor = 1
        else:
            num_forget_ex = min(len(forget_prompts), 20)
            num_retain_ex = min(len(retain_prompts_list[i]), 50)
            oversampling_factor = 20
        print("Preparing datasets...")
        oversampled_forget_list = forget_prompts[:num_forget_ex] * oversampling_factor
        # forget_dataset = ds_manager_obj.tokenize_data(oversampled_forget_list)
        # retain_dataset = ds_manager_obj.tokenize_data(retain_prompts_list[i][:num_retain_ex])

        # oracle_dataset = concatenate_datasets([forget_dataset, retain_dataset])
        # sft_dataset = oracle_dataset.remove_columns(["pii_spans", "offset_mapping"])
        # sft_dataset = sft_dataset.shuffle(seed=42)
        # retain_prompts = retain_prompts_list[i][:num_retain_ex]
        # custom_collator = SpanAwareCollator(tokenizer=tokenizer)
        # forget_dataloader = DataLoader(
        #     forget_dataset,
        #     batch_size=4,
        #     shuffle=False,
        #     collate_fn=custom_collator,
        #     pin_memory=True,
        #     num_workers=4,
        #     prefetch_factor=2,
        #     persistent_workers=True,
        # )
        # retain_dataloader = DataLoader(
        #     retain_dataset,
        #     batch_size=4,
        #     shuffle=True,
        #     collate_fn=custom_collator,
        #     pin_memory=True,
        #     num_workers=4,
        #     prefetch_factor=2,
        #     persistent_workers=True,
        # )

        # ul_training_params = {
        #     "forget_ds": forget_dataloader,
        #     "retain_ds": retain_prompts,
        #     "gk_prompts": gk_prompts,
        #     "ul_epochs": num_epochs_train_student,
        #     "ulr": unlearning_rate,
        #     "forget_loss_wt": args.lambda_forget,
        #     "retain_loss_wt": args.lambda_retain,
        #     "gk_loss_wt": args.lambda_gk,
        #     "finetuned_model_dir": tuned_model_path,
        # }
        # print("Training oracle and student models...")
        # trainer_obj = ModelTrainer(model, tokenizer, ul_training_params)
        # if not args.load_pretrained:
        #     trainer_obj.train_oracle(sft_dataset, num_epochs_train_oracle, learning_rate)
        #     builder.add_adapter_to_base_model()

        # if args.run_unlearn:
        #     trainer_obj.train_student(npo_loss=args.train_with_npo_loss, ga_loss=args.train_with_ga_loss)

        print("Evaluating on TOFU")
        tokenizer.padding_side = "left"
        eval_obj = EvaluationManager(
            model,
            tokenizer,
            None,
            fluency_ques_list,
            gk_prompts,
            None,
            device,
        )
        test_prompts = forget_prompts[:num_forget_ex]

        # response_logs = eval_obj.check_model_health(test_prompts)
        # oracle_response_logs = eval_obj.check_model_health(test_prompts, is_base_model=True)
        # eval_obj.dump_results_to_json(
        #     response_logs,
        #     oracle_response_logs,
        #     filename_prefix=f"Llama-3.2-1B_{model_short}_{i}",
        # )
        perplexity = eval_obj.get_perplexity(gk_prompts)
        # oracle_perplexity = eval_obj.get_perplexity(gk_prompts, is_base_model=True)
        # print(f"Oracle Model Perplexity on GK Prompts: {oracle_perplexity:.2f}")
        print(f"Model {model_id} perplexity on GK prompts: {perplexity:.2f}")

        curated_test = ds_manager_obj.load_text_prompts(f"./data/TOFU/text_prompts_forget05.txt")
        eval_obj.calculate_mean_metrics(curated_test, is_base_model=False)
        # eval_obj.calculate_mean_metrics(curated_test, is_base_model=True)

if __name__ == "__main__":
    main()
