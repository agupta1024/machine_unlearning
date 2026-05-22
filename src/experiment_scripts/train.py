import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import DataCollatorWithPadding

model_id = "meta-llama/Llama-3.2-1B"
outdir = "./oracle_adapter_hf"

class ModelTrainer:
    def __init__(self, model_id: str, model, tokenizer, training_params=None):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.training_params = training_params
        self._model = None
        self._tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_oracle(self, dataset, num_epochs, lr):

        # ==========================================
        # 5. THE DATA COLLATOR (Llama-3 ID Hack)
        # ==========================================
        # Why it crashed before: Llama-3's BPE tokenizer merges text weirdly. 
        # String matching often fails. We pass the raw Token IDs instead to guarantee it finds the split.

        response_template = " Answer:" # Change this if your prompt ends differently
        # Encode the template and slice off the automatic BOS token (the [1:])
        response_template_ids = self.tokenizer.encode(response_template, add_special_tokens=False)[1:]

        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids, 
            tokenizer=self.tokenizer
        )

        # ==========================================
        # 6. TRAINER & EXECUTION
        # ==========================================
        # training_args = TrainingArguments(
        #     output_dir=outdir,
        #     per_device_train_batch_size=4,
        #     gradient_accumulation_steps=4,
        #     learning_rate=lr,
        #     logging_steps=10,
        #     max_steps=200, # Adjust to your epochs
        #     optim="paged_adamw_8bit",
        #     fp16=False,
        #     bf16=True, # Use bf16 for Llama 3
        #     save_strategy="no", # We save manually at the end
        # )
        training_args = SFTConfig(
            output_dir=outdir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            max_steps=500, 
            optim="paged_adamw_8bit",
            fp16=False,
            bf16=True, 
            save_strategy="no", 
            
            # --- NEW SFT-SPECIFIC ARGS MOVED HERE ---
            max_seq_length=1024,
            dataset_text_field=None,
            packing=False,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset, # Your loaded dataset
            args=training_args,
            processing_class=self.tokenizer,
            # data_collator=collator, # NOW THE PROMPT IS PROPERLY HIDDEN!
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )

        print("Starting standard HF Oracle training...")
        trainer.train()

        # ==========================================
        # 7. THE BULLETPROOF SAVE ROUTINE
        # ==========================================
        if trainer.is_world_process_zero():
            # Save the weights
            trainer.model.save_pretrained(outdir)
            # CRITICAL: Save the tokenizer state we configured in Step 1
            self.tokenizer.save_pretrained(outdir)
            print(f"Oracle successfully saved to {outdir}")