# Machine Unlearning Project

This project contains code for machine unlearning experiments using Llama-based language models, LoRA adapters, and probability-based generation analysis.

## Project Structure

- `main.py` — training / evaluation entry point
- `generation.py` — model loading, generation, and probability checks
- `src/` — core project modules
- `prepare_dataset/` — dataset preparation utilities
- `Lora_model/` — fine-tuned adapter checkpoints

## Setup

1. Create and activate a Python environment.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure model paths in the scripts match your local files.

## Usage

### Run the main workflow
```sh
python main.py
```

### Run generation and analysis
```sh
python generation.py
```

## Notes

- The scripts assume access to a compatible Hugging Face model checkpoint.
- `generation.py` uses an 8-bit quantized base model and a LoRA adapter.
- Update paths such as `./Lora_model/...` and `./oracle_adapter_hf` as needed.

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- PEFT
- Datasets
- bitsandbytes
- wandb