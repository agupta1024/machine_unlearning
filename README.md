[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mbjnNZjl)
# ELEC0141: Deep Learning for Natural Language Processing

## Assignment

## 1. General Overview

This assignment requires the development of a complete machine learning system addressing a real-world research competition challenge in computer vision.

Each student must:

- Select one public-domain competition (Kaggle or TopCoder).
- Develop and test a machine learning solution.
- Propose and validate a clear research hypothesis.
- Report findings in the format of a TMLR-style conference paper.

The project must demonstrate:

- Sound model design
- Proper training/validation/testing methodology
- Experimental analysis and ablation studies
- Reproducibility
- Balanced complexity vs performance trade-offs

The goal is not to achieve leaderboard dominance, but to demonstrate strong reasoning, engineering design, and experimental validation.

---

## 2. Challenge Selection

You must select **one competition** from:

- Kaggle (past 3 years recommended)  
  https://www.kaggle.com/competitions  

- TopCoder challenges  
  https://www.topcoder.com/challenges?bucket=allPast&tab=details  

### Allowed Task Types (Computer Vision Only)

You must choose one of:

- Image classification
- Image segmentation
- Image inpainting / super resolution
- [Advanced] Conditional generation / multimodality
- [Advanced] Image generation

⚠ NLP-only competitions are not allowed.

If selecting an **advanced generative project**, you must:

- Justify feasibility
- Specify dataset size
- Specify model size
- Estimate compute requirements
- Use compact models or parameter-efficient fine-tuning
- Avoid training large-scale generative models from scratch
- Get it approved by the team beforehand

---

## 3. Research Hypothesis

Your project must be structured around a clearly defined hypothesis, such as:

- Architectural comparison / Inductive bias comparison (e.g., CNNs vs ViT, Attention in Segmentation, etc.)
- Training Strategies & Meta-Learning (e.g., transfer learning vs training from scratch, data augmentation, regularization, etc.)
- Objective functions (e.g., auxiliary losses, various training losses, etc.)
- Label smoothing
- Class imbalance handling
- Robustness to noise
- Ensemble methods

Your experiments must test this hypothesis through empirical evaluation.

---

## 4. Constraints

- No paid services.
- Use only free/public datasets and infrastructure.
- No external database services.
  - Spawn a local database if needed.
  - If remote, it must remain accessible for 2 months after submission.
- Plain Python only.
- No notebooks.
- No Makefiles.
- Deterministic execution (fixed seeds).
- No interactive input.
- All plots saved to disk.
- Training must be feasible on limited compute.

### AI Usage Disclosure

This assignment follows UCL Category 2 GenAI usage.

- Undeclared GenAI use will be penalised.
- Reports and code not using GenAI are rewarded.
- If used, clearly disclose usage in the report.

---

## 5. Deliverables

### Report (80%)

- Max 8 pages (excluding references + optional appendix).
- Must use TMLR template (provided on Moodle).
- Must be submitted in PDF format.
- File naming format:

  Report_NLP_25-26_SNXXXXXXXX.pdf

- Include:
  - Student number
  - GitHub repo URL
- Do NOT include your name.

### Code (20%)

The repository must:

- Produce all experimental evidence presented in the report.
- Contain a single entry point: `main.py` located in the root directory.
- Execute the complete experimental workflow automatically when running:

  ```bash
  python main.py

- Include `environment.yml` in root.
- Be fully reproducible.
- Require no manual intervention.

Autograding will:

1. Install `environment.yml`
2. Run `python main.py`

Any manual README instructions will be ignored.

---

## 6. Marking Scheme

### REPORT — 80%

| Section | Weight |
|----------|--------|
| Abstract | 5% |
| Introduction | 5% |
| Literature Review | 10% |
| Model Design & Methodology | 20% |
| Implementation Details | 20% |
| Experimental Results, Analysis & Conclusion | 20% |

### CODE — 20%

| Component | Weight |
|------------|--------|
| Reproducibility | 7% |
| Code quality & documentation | 7% |
| Code organisation | 2.5% |
| Git & GitHub usage | 2.5% |

Performance alone does not determine marks.  
Clarity, reasoning, experimental validation, and engineering design matter most.
