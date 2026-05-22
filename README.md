# Machine Unlearning Project

This repository contains a machine unlearning workflow built around the TOFU benchmark. The code trains a LoRA-augmented causal language model, performs unlearning on sensitive prompts, and writes evaluation outputs to disk.

**Quickstart**

1. Clone Repository

```bash
git clone https://github.com/UCL-ELEC0141-26/assignment-new-agupta1024.git path/to/project
cd path/to/project
```

2. Create and activate a Python environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or if you use conda
# conda env create -f environment.yml
# conda activate <env-name>
```
3. Run the entry point:

```bash
python3 main.py
```

## Project layout

```
assignment-new-agupta1024/
│
├── main.py                          # primary entry point for training, unlearning, and evaluation
├── requirements.txt                 # Python package dependencies
├── environment.yml                  # Conda environment specification
│
├── src/
│   ├── model/                       # model construction and training helpers
│   │	├── builder.py
│   │	└── train.py
│   ├── dataset                      # dataset loading, tokenisation, and collators
│   │	└── dataset.py
│   └── analysis/                    # evaluation, generation, and reporting utilities
│   	├── evaluation.py
│   	└── generation.py
├── oracle_adapter_hf/               # Oracle LoRA weights
├── student_model/                   # Student LoRA weights
└── data/TOFU                        # Downloaded dataset   
```

## Notes

- The repository is designed to be reproducible and script-driven.
- Evaluation outputs are saved under `results/`.
- The code targets the TOFU unlearning setup used in the assignment.
