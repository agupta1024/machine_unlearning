"""Dataset utilities for unlearning experiments."""

# pylint: disable=too-few-public-methods,too-many-instance-attributes,too-many-locals,attribute-defined-outside-init,multiple-statements,line-too-long

import os
import json
import datasets
import spacy
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from transformers import DataCollatorWithPadding
from spacy.util import is_package

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = f"{dir_path}/../../data"

class SpanAwareCollator(DataCollatorWithPadding):
    """Data collator that preserves per-sample PII span metadata."""

    def __call__(self, features):
        pii_spans = [feature.pop("pii_spans", []) for feature in features]
        batch = super().__call__(features)
        batch["pii_spans"] = pii_spans
        return batch

class DatasetManager:
    """Loads, formats, and tokenizes forget/retain/general-knowledge datasets."""

    def __init__(self, padding=True, truncation=True, max_length=None):
        """Initialize dataset manager with tokenizer behavior settings."""
        self._tokenizer = PreTrainedTokenizerBase
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.category_subs = {
            "PERSON": "someone",
            "GPE": "location",    # Countries, Cities, States
            "LOC": "area",        # Non-GPE locations (mountains, rivers)
            "ORG": "group",
            "DATE": "sometime",
            "NORP": "culture"     # Nationalities, Religions, Political groups
        }
        self.protected_words = {"the", "a", "an", "in", "to", "for", "of", "and", "is", "by",
                                "with", ',', '.', '?', '!', ':', ';', '-', '_', '(', ')',
                                '[', ']', '{', '}', '"', "'"}
        model_name = "en_core_web_md"
        if not is_package(model_name):
            print(f"Downloading {model_name}...")
            spacy.cli.download(model_name)
        self.nlp = spacy.load("en_core_web_md")

    def get_remote_data(self, dataset_name: str):
        """Fetch remote HF datasets for the requested benchmark."""
        if dataset_name == 'TOFU':
            forget_01 = load_dataset("locuslab/TOFU", "forget01")
            forget_05 = load_dataset("locuslab/TOFU", "forget05")
            forget_10 = load_dataset("locuslab/TOFU", "forget10")
            retain_99 = load_dataset("locuslab/TOFU", "retain99")
            retain_95 = load_dataset("locuslab/TOFU", "retain95")
            retain_90 = load_dataset("locuslab/TOFU", "retain90")
            self.forget_datasets = [forget_01, forget_05, forget_10]
            self.retain_datasets = [retain_99, retain_95, retain_90]
            return
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
        self.get_remote_data(dataset_name)
        forget_list = self.forget_datasets
        retain_list = self.retain_datasets
        forget_prompts_list = []
        for _, ds_item in enumerate(forget_list):
            train_dataset = ds_item["train"]
            questions = train_dataset["question"]
            answers = train_dataset["answer"]
            forget_prompts_list.append([f"Question: {q} Answer: {a}"
                                        for q, a in zip(questions, answers)])

        retain_prompts_list = []
        for _, ds_item in enumerate(retain_list):
            train_dataset = ds_item["train"]
            questions = train_dataset["question"]
            answers = train_dataset["answer"]
            retain_prompts_list.append([f"Question: {q} Answer: {a}"
                                        for q, a in zip(questions, answers)])
        return [forget_prompts_list, retain_prompts_list]

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
        self.span_text_dict = {}
        if isinstance(raw_prompts, list):
            raw_prompts = HFDataset.from_dict({"text": raw_prompts})

        tokenized_dataset = raw_prompts.map(
            self._tokenize_function,
            batched=True,
            batch_size=128,
            num_proc=None,
            remove_columns=['text']
        )

        with open("./pii_mapped_entities.json", "w", encoding='utf-8') as f:
            json.dump(self.span_text_dict, f, indent=4)
        return tokenized_dataset

    def _tokenize_function(self, examples):
        """Tokenize batch examples and mask question/padding tokens in labels."""
        full_texts = [text + self._tokenizer.eos_token for text in examples['text']]
        outputs = self._tokenizer(
            full_texts,
            truncation=self.truncation,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        outputs["pii_spans"] = self.get_pii_scans(examples, outputs)

        labels = outputs['input_ids'].clone()
        # separator = " Answer: "
        # sep_ids = self._tokenizer(separator, add_special_tokens=False).input_ids

        for i, text in enumerate(full_texts):
            input_id_row = outputs['input_ids'][i]
            if " Answer: " in text:
                char_idx = text.find(" Answer: ") + len(" Answer: ")
                token_idx = outputs.char_to_token(i, char_idx)

                if token_idx is not None:
                    labels[i, :token_idx] = -100
                else:
                    colon_positions = (input_id_row == 25).nonzero(as_tuple=True)[0]
                    if len(colon_positions) > 0:
                        labels[i, :colon_positions[0] + 1] = -100

        labels[outputs['attention_mask'] == 0] = -100
        outputs['labels'] = labels
        return outputs

    def get_sensitive_token_ids(self, sensitive_word_file):
        """Map sensitive words to tokenizer IDs, excluding protected tokens."""
        sensitive_ids = set()
        with open(sensitive_word_file, 'r', encoding='utf-8') as f:
            sensitive_words = [line.strip() for line in f if line.strip()]
        for word in sensitive_words:
            tokens = self._tokenizer(" " + word.strip(), add_special_tokens=False).input_ids

            for t_id in tokens:
                decoded_t = self._tokenizer.decode([t_id]).strip().lower()
                if decoded_t not in self.protected_words and len(decoded_t) > 0:
                    sensitive_ids.add(t_id)
        return list(sensitive_ids)

    def get_pii_scans(self, examples, tokenized_outputs):
        """Build token span annotations for detected named entities in each prompt."""
        all_spans = []
        for i, doc in enumerate(self.nlp.pipe(examples["text"])):
            current_sentence_spans = []
            offsets = tokenized_outputs["offset_mapping"][i]

            for ent in doc.ents:
                if ent.label_ in self.category_subs:
                    start_char, end_char = ent.start_char, ent.end_char
                    t_start, t_end = None, None

                    for idx, (s, e) in enumerate(offsets):
                        if s == 0 and e == 0 and idx != 0: continue
                        if s <= start_char < e: t_start = idx
                        if s < end_char <= e: t_end = idx

                    if t_start is not None and t_end is not None:
                        safe_word = self.category_subs[ent.label_]
                        safe_id = self._tokenizer.encode(" " + safe_word, add_special_tokens=False)[-1]
                        current_sentence_spans.append([t_start, t_end, safe_id])
                        decoded_safe = self._tokenizer.decode([safe_id])
                        self.span_text_dict[ent.text] = decoded_safe
                        # print(f"Mapping '{ent.text}' to '{decoded_safe}'")
                        # print(f"Mapping '{ent.text}' to '{decoded_safe}' at token positions {t_start}-{t_end} with safe ID {safe_id}")
                    else:
                        print(f"Warning: Could not find token positions for entity '{ent.text}' in prompt. Skipping.")

            all_spans.append(current_sentence_spans)
        return all_spans

    def prepare_general_knowledge_prompts(self, verbose=False, min_len=50, max_len=700):
        """Load and filter generic knowledge prompts from EasyRAG mini-wikipedia."""
        gk_prompts = []
        gk_prompts = datasets.load_dataset(
            "philschmid/easyrag-mini-wikipedia",
            "documents",
            split="full"
        )['document']
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

    def load_text_prompts(self, file_path):
        """Load newline-delimited text prompts from disk."""
        with open(file_path, 'r', encoding='utf-8') as file:
            prompts = [line.strip() for line in file if line.strip()]
            prompt_parts = [p.split('>') for p in prompts]
            prefix = [p[0] for p in prompt_parts]
            target = [p[1] for p in prompt_parts]
            safe_word = [p[2] for p in prompt_parts]
        return prefix, target, safe_word

if __name__ == "__main__":
    pass
