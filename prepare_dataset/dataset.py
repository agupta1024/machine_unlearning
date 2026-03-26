"""Dataset utilities for unlearning experiments."""
import os

import datasets
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = f"{dir_path}/../data"

class DatasetManager:
    """Loads, formats, and tokenizes forget/retain/general-knowledge datasets."""

    def __init__(self, padding=True, truncation=True, max_length=None):
        """Initialize dataset manager with tokenizer behavior settings."""
        self._tokenizer = PreTrainedTokenizerBase
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def get_remote_data(self, dataset_name: str):
        """Fetch remote HF datasets for the requested benchmark."""
        if dataset_name == 'TOFU':
            forget_01 = load_dataset("locuslab/TOFU", "forget01")
            forget_05 = load_dataset("locuslab/TOFU", "forget05")
            forget_10 = load_dataset("locuslab/TOFU", "forget10")
            retain_99 = load_dataset("locuslab/TOFU", "retain99")
            retain_95 = load_dataset("locuslab/TOFU", "retain95")
            retain_90 = load_dataset("locuslab/TOFU", "retain90")
            return {'forget': [forget_01, forget_05, forget_10],
                    'retain': [retain_99, retain_95, retain_90]}
        raise ValueError(f"Dataset {dataset_name} not supported.")

    def get_sensitive_tokens_path(self, dataset_name: str):
        """Return local paths of sensitive-token files for a dataset."""
        if dataset_name == "TOFU":
            target_path = [f"{DATA_DIR}/TOFU/sensitive_tokens_forget01.txt",
                           f"{DATA_DIR}/TOFU/sensitive_tokens_forget05.txt",
                           f"{DATA_DIR}/TOFU/sensitive_tokens_forget10.txt"]
            return target_path
        raise ValueError(f"Dataset {dataset_name} not supported.")

    @staticmethod
    def load_sensitive_words(file_path):
        """Load newline-delimited sensitive words from disk."""
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read lines and strip any extra whitespace characters
            sensitive_words = [line.strip() for line in file if line.strip()]
        return sensitive_words

    def get_raw_data(self, dataset_name: str):
        """Build formatted forget/retain prompt lists and token-path metadata."""
        forget_list = self.get_remote_data(dataset_name)['forget']
        retain_list = self.get_remote_data(dataset_name)['retain']
        forget_prompts_list = []
        for _, ds_item in enumerate(forget_list):
            train_dataset = ds_item["train"]
            questions = train_dataset["question"]
            answers = train_dataset["answer"]
            forget_prompts_list.append([f"Question: {q} Answer: {a}"
                                        for q, a in zip(questions, answers)])
            # forget_prompts_list.append([f"{question}: {answer}"
            #                             for question, answer in zip(questions, answers)])
        retain_prompts_list = []
        for _, ds_item in enumerate(retain_list):
            train_dataset = ds_item["train"]
            questions = train_dataset["question"]
            answers = train_dataset["answer"]
            retain_prompts_list.append([f"Question: {q} Answer: {a}"
                                        for q, a in zip(questions, answers)])
                # [f"{question}: {answer}"
                #                        for question, answer in zip(questions, answers)])

        target_path = self.get_sensitive_tokens_path(dataset_name)
        return [forget_prompts_list, retain_prompts_list, target_path]

    @property
    def tokenizer(self):
        """Return active tokenizer."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase):
        """Set active tokenizer instance."""
        self._tokenizer = tokenizer

    def tokenize_list_dataset(self, dataset_list):
        """Tokenize a nested prompt-list dataset into HF datasets."""
        tokenized_list = []
        for sublist in dataset_list:
            raw_ds = HFDataset.from_dict({"text": sublist})
            tokenized_ds = self.tokenize_data(raw_ds)
            tokenized_list.append(tokenized_ds)
        return tokenized_list

    def tokenize_data(self, raw_prompts):
        """Tokenize prompts into input tensors and labels."""
        if isinstance(raw_prompts, list):
            raw_prompts = HFDataset.from_dict({"text": raw_prompts})

        tokenized_dataset = raw_prompts.map(
            self._tokenize_function,
            batched=True,
            batch_size=128,
            num_proc=None,
            remove_columns=['text']
        )
        tokenized_dataset.set_format(type='torch')
        return tokenized_dataset

    def _tokenize_function(self, examples):
        """Tokenize batch examples and mask question/padding tokens in labels."""
        full_texts = [text + self._tokenizer.eos_token for text in examples['text']]
        # full_text = prompt['text'] + self._tokenizer.eos_token
        outputs = self._tokenizer(
            full_texts,
            truncation=self.truncation,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = outputs['input_ids'].clone()
        for i, text in enumerate(examples['text']):
            if " Answer: " in text:
                parts = text.split(" Answer: ")
                question_part = parts[0] + " Answer: "

                # tokenize question WITHOUT adding special tokens again
                # This prevents the double-BOS length error
                question_ids = self._tokenizer(
                    question_part,
                    add_special_tokens=False
                )["input_ids"]

                # If the tokenizer adds a BOS to the FULL text but not here,
                # we need to account for it.
                # Usually, Unsloth/Llama/Qwen use 1 BOS token.
                has_bos = 1 if outputs['input_ids'][i][0] == self._tokenizer.bos_token_id else 0
                question_len = len(question_ids) + has_bos

                labels[i, :question_len] = -100

        # Mask padding tokens using the attention mask
        labels[outputs['attention_mask'] == 0] = -100

        outputs['labels'] = labels
        return outputs

    def prepare_general_knowledge_prompts(self, verbose=False, min_len=50, max_len=700):
        """Load and filter generic knowledge prompts from EasyRAG mini-wikipedia."""
        gk_prompts = []
        gk_prompts = datasets.load_dataset(
            "philschmid/easyrag-mini-wikipedia",
            "documents",
            split="full"
        )['document']
        # Filter out texts that do not fall within the specified self.length range
        gk_prompts = [p[:max_len] for p in gk_prompts if len(p) > min_len]

        if verbose:
            print(f"Loaded {len(gk_prompts)} general knowledge prompts for dataset")
        return gk_prompts

    def load_fluency_ques_bank(self):
        """Load fluency question prompts from the local common-knowledge file."""
        ques_bank_file = f"{DATA_DIR}/common_knowledge.txt"
        with open(ques_bank_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        return questions

if __name__ == "__main__":
    pass
