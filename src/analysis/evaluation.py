"""Evaluation helpers for unlearning model quality and retention checks."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals

import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from bert_score import score
from rouge_score import rouge_scorer

class EvaluationManager:
    """Compute unlearning metrics and generate evaluation reports."""

    def __init__(self, model, tokenizer, forget_ds, fluency_ques_list=None,
                 gk_prompts=None, retain_ds=None, device="cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.forget_dataloader = forget_ds
        self.fluency_ques_list = fluency_ques_list
        self.gk_prompts = gk_prompts
        self.retain_ds = retain_ds
        self.device = device

    def _get_metrics(self, prefix, target, safe_word, is_base_model=False):
        """
        Calculates TTP and STP using Sequence Forcing to guarantee exact token matching.
        """
        if is_base_model:
            if "oracle" in self.model.peft_config:
                self.model.set_adapter("oracle")
            else:
                print('Oracle adapter not found.')
                return float('inf'), float('inf'), float('inf')
        else:
            self.model.set_adapter("student")

        prefix_inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        prefix_len = prefix_inputs["input_ids"].shape[1]
        full_target_text = prefix + target
        full_safe_text = prefix + safe_word

        full_target_ids = self.tokenizer(full_target_text, return_tensors="pt")["input_ids"][0]
        full_safe_ids = self.tokenizer(full_safe_text, return_tensors="pt")["input_ids"][0]

        target_first_token_id = full_target_ids[prefix_len]
        safe_first_token_id = full_safe_ids[prefix_len]
        # breakpoint()

        # =====================================================================
        # DEBUG CHECK:
        # print(f"Prefix: '{prefix}'")
        # print(f"Comparing target ['{self.tokenizer.decode([target_first_token_id])}'] \
        # vs safe ['{self.tokenizer.decode([safe_first_token_id])}']")
        # =====================================================================

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **prefix_inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            new_tokens = generated_tokens[0][prefix_inputs['input_ids'].shape[1]:]
            _ = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            # print(f"Prefix: '{prefix}'")
            # print(f"-> Full Generation: {response.strip()}")
            # print(f"Comparing target ['{self.tokenizer.decode([target_first_token_id])}'] \
            #       vs safe ['{self.tokenizer.decode([safe_first_token_id])}']")

        with torch.no_grad():
            outputs = self.model(**prefix_inputs)
            next_token_logits = outputs.logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

            prob_true = probs[target_first_token_id].item()
            prob_safe = probs[safe_first_token_id].item()
            truth_ratio = prob_true / (prob_safe + 1e-9)

        return prob_true, prob_safe, truth_ratio

    def calculate_mean_metrics(self, eval_dataset, is_base_model=False):
        """
        Loops through the dataset to generate the final table numbers.
        """
        all_ttps = []
        all_stps = []
        all_truth_ratios = []

        self.model.eval()
        prefix_list, target_list, safe_word_list = eval_dataset
        for p, t, s in zip(prefix_list, target_list, safe_word_list):
            ttp, stp, t_ratio = self._get_metrics(p, t, s, is_base_model)
            all_ttps.append(ttp)
            all_stps.append(stp)
            all_truth_ratios.append(t_ratio)

        mean_ttp = np.mean(all_ttps)
        mean_stp = np.mean(all_stps)
        mean_truth_ratio = np.mean(all_truth_ratios)

        print(f"--- Metrics for {is_base_model=} ---")
        print(f"Mean TTP: {mean_ttp:.6f}")
        print(f"Mean STP: {mean_stp:.6f}")
        print(f"Mean Truth Ratio: {mean_truth_ratio:.4f}")

        return mean_ttp, mean_stp, mean_truth_ratio

    def generate_answer(self, question):
        """Generate an answer string for a question prompt."""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _get_semantic_accuracy(self, dataset, threshold=0.5):
        """Measure semantic match ratio using ROUGE-L thresholding."""
        self.model.eval()
        correctness_scores = []
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        log_data = []
        for prompt in dataset:
            if " Answer: " in prompt:
                parts = prompt.split(" Answer: ")
                question = parts[0] + " Answer: "
                answer = parts[1]
            else:
                continue
            prediction = self.generate_answer(question)
            target = answer

            rouge_score_value = scorer.score(target, prediction)['rougeL'].fmeasure
            correctness_scores.append(1 if rouge_score_value >= threshold else 0)
            is_correct = rouge_score_value >= threshold
            log_data.append(
                [question, prediction, target,
                 "Match" if is_correct else "No Match", rouge_score_value]
            )
        return (sum(correctness_scores) / len(dataset)) * 100, log_data

    def _get_perplexity(self, dataset, batch_size=8, is_base_model=False):
        """Compute perplexity over a dataset of prompts."""
        self.model.eval()

        if is_base_model:
            if "oracle" in self.model.peft_config:
                self.model.set_adapter("oracle")
            else:
                print('Oracle adapter not found.')
                return float('inf')
        else:
            self.model.set_adapter("student")

        print(f"Active Adapter: {self.model.active_adapter}")
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch_texts = dataset[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(self.model.device)

                labels = inputs['input_ids'].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=labels
                )
                total_loss += outputs.loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches
        perplexity = math.exp(avg_loss)

        return perplexity

    def get_perplexity(self, dataset, batch_size=8, is_base_model=False):
        """Public wrapper for perplexity computation."""
        return self._get_perplexity(dataset, batch_size=batch_size, is_base_model=is_base_model)

    def check_model_health(self, test_prompts=None, is_base_model=False):
        """Run live generation checks on test prompts and log responses."""
        response_logs = []
        self.model.eval()
        print("\n--- LIVE HEALTH CHECK ---")
        if is_base_model:
            if "oracle" in self.model.peft_config:
                self.model.set_adapter("oracle")
            else:
                print('Oracle adapter not found.')
                return response_logs
        else:
            self.model.set_adapter("student")
        print(f"Active Adapter: {self.model.active_adapter}")
        for prompt in test_prompts:
            if " Answer: " in prompt:
                parts = prompt.split(" Answer: ")
                question = parts[0] + " Answer: "
                answer = parts[1]
            else:
                question = prompt
                answer = ''

            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            response_logs.append([question, response.strip(), answer])

            # print(f"Prompt: {question}")
            # print(f">> Response: {response.strip()}")
            # print(f">> Target: {answer}")
            # print("-" * 15)

        return response_logs

    def get_semantic_metrics(self, predictions, references):
        """
        predictions: List of strings generated by model
        references: List of ground-truth strings (targets)
        """
        # lang="en" uses roberta-large by default
        precision_vals, recall_vals, f1_vals = score(
            predictions,
            references,
            lang="en",
            verbose=False,
        )

        return {
            "bert_precision": precision_vals.mean().item(),
            "bert_recall": recall_vals.mean().item(),
            "bert_f1": f1_vals.mean().item(),
        }

    def dump_results_to_json(
        self,
        response_logs,
        base_reponse_logs=None,
        folder_path="./results",
        filename_prefix="unlearning_audit",
    ):
        """Persist model and baseline responses to a timestamped JSON file."""
        if base_reponse_logs is None:
            base_reponse_logs = []

        os.makedirs(folder_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        full_path = os.path.join(folder_path, filename)
        report_data = {
            "model_results": [
                {"prompt": r[0], "prediction": r[1], "target": r[2]}
                for r in response_logs
            ],
            "baseline_results": [
                {"prompt": r[0], "prediction": r[1], "target": r[2]}
                for r in base_reponse_logs
            ],
        }

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, default=list, indent=4, ensure_ascii=False)

        print(f"Audit results successfully dumped to: {full_path}")
