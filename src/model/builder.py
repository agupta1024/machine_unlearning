"""Model builder to load base/oracle/student models and attach LoRA adapters."""

# pylint: disable=too-many-arguments,too-many-positional-arguments

import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)


class ModelBuilder:
    """Helper class to build and manage the base and student models with LoRA adapters."""
    def __init__(self, load_in_4bit=False, load_model:str = 'base',
                oracle_path="./oracle_adapter_hf", tuned_model_path=None,
                for_training=True):

        model_id = "meta-llama/Llama-3.2-1B"
        if os.getenv("GITHUB_ACTIONS") == "true":
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # ==========================================
        # TOKENIZER SETUP
        # ==========================================
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_bos_token = False
        if for_training:
            tokenizer.padding_side = "right"
        else:
            tokenizer.padding_side = "left"

        # ==========================================
        # QUANTIZATION CONFIG
        # ==========================================
        has_cuda = torch.cuda.is_available()
        bnb_config = None

        if has_cuda:
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        target_device = {"": torch.cuda.current_device()} if has_cuda else {"": "cpu"}

        # ==========================================
        # LOAD BASE MODEL & PREPARE
        # ==========================================
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=target_device,
            torch_dtype=torch.bfloat16
        )
        model = prepare_model_for_kbit_training(model)

        # ==========================================
        # ATTACH LoRA
        # ==========================================
        self.peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head"
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )

        if load_model == 'base':
            model = get_peft_model(model, self.peft_config, adapter_name="oracle")
        elif load_model == 'student':
            model = PeftModel.from_pretrained(model, f"{tuned_model_path}/student",
                                              adapter_name="student")
        else:
            model = PeftModel.from_pretrained(model, oracle_path, adapter_name=load_model)
            if for_training:
                model.add_adapter("student", self.peft_config)
                model.load_adapter(oracle_path, adapter_name="student")
        model.print_trainable_parameters()
        self.model = model
        self.tokenizer = tokenizer

    def get_model_and_tokenizer(self):
        """Return the prepared model and tokenizer."""
        return self.model, self.tokenizer

    def add_adapter_to_base_model(self):
        """Add a new LoRA adapter for the student model and initialize it with oracle weights."""
        self.model.add_adapter("student", self.model.peft_config['oracle'])
        for _, module in self.model.base_model.named_modules():
            if hasattr(module, "lora_A") and "oracle" in module.lora_A:
                with torch.no_grad():
                    module.lora_A["student"].weight.copy_(module.lora_A["oracle"].weight)
                    module.lora_B["student"].weight.copy_(module.lora_B["oracle"].weight)
                    if hasattr(module, "scaling"):
                        module.scaling["student"] = module.scaling["oracle"]
