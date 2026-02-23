# Battle of BERTS - FOMC Hawkish-Dovish Stance Classification
### DSA4265 Take-Home Assignment 1  
this readme and code cleanup was kindly written with claude ai

-----

## Project Overview

As part of my DSA4265 assignment, this project compares **FinBERT** (110M params, financial domain pre-training) against **RoBERTa-large** (355M params, general-purpose) on the task of classifying FOMC (Federal Open Market Committee) sentences as **Hawkish**, **Dovish**, or **Neutral**.

Building on the [Trillion Dollar Words](https://aclanthology.org/2023.acl-long.368/) paper by Shah et al. (ACL 2023), the project replicates their finding that RoBERTa-large outperforms domain-specialized models, and contributes an interpretive analysis of *why* financial domain pre-training fails on central bank text — a register mismatch between corporate finance language and FOMC communication.

## Key Results

| Metric | FinBERT (110M) | RoBERTa (355M) | Claude Zero-Shot |
|--------|---------------|-----------------|------------------|
| Accuracy | 0.651 | 0.726 | 0.686 |
| F1 (Macro) | 0.633 | 0.710 | 0.658 |
| AUC-ROC (Macro) | 0.820 | 0.870 | — |

## Repository Structure

```
DSA4265/
├── README.md                           ← This file
├── report.pdf                          ← Report detailing results, shortcomings and future improvements
├── FOMC_Step1_Data_Preparation.ipynb   ← Data loading, EDA, LLM labeling
├── FOMC_Step2_Training.ipynb           ← Model training & evaluation
│
├── Data/
│   ├── train.csv                      ← Training split (1,683 sentences)
│   ├── val.csv                        ← Validation split (297 sentences)
│   ├── test.csv                       ← Test split (496 sentences)
│   └── llm_labels_fomc.json           ← Claude zero-shot labels for all sentences
│
├── Models/
│   ├── finbert_model/                 ← Saved FinBERT fine-tuned model
│   └── roberta_model/                 ← Saved RoBERTa fine-tuned model
│
└── Figures/
    ├── class_distribution.png
    ├── sentence_lengths.png
    ├── temporal_distribution.png
    ├── llm_vs_human_confusion.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── training_curves.png
    └── performance_by_era.png
```

> **Note:** Adjust the folder structure above to match your actual Google Drive layout. Not all files may be present if Colab storage was cleared during training.

## How to Run

### Prerequisites
- Google Colab (free tier with T4 GPU)
- Anthropic API key (for LLM labeling in Step 1)

### Step 1: Data Preparation
1. Open `FOMC_Step1_Data_Preparation.ipynb` in Google Colab
2. Input your Anthropic API key where requested.
3. Run all cells — this will:
   - Load the dataset from HuggingFace (`gtfintechlab/fomc_communication`)
   - Perform exploratory data analysis
   - Run Claude zero-shot labeling on all 2,476 sentences (~15 min, ~$1)
   - Compare LLM labels with human annotations
   - Create train/val/test splits and save to Google Drive

### Step 2: Model Training
1. Open `FOMC_Step2_Training.ipynb` in Google Colab
2. Ensure GPU runtime is selected (Runtime → Change runtime type → T4 GPU)
3. Run all cells — this will:
   - Train FinBERT (~2 min) and RoBERTa-large (~16 min)
   - Evaluate both models on the test set
   - Generate all comparison figures and error analysis
   - Save results to Google Drive

### Important Notes
- **Storage:** RoBERTa checkpoints are 7-8GB each. On Colab free tier, set `load_best_model_at_end=False` to avoid exceeding the 15GB limit.
- **Reproducibility:** Both notebooks use `seed=42` for deterministic results, though minor variations may occur across different GPU hardware.

## Dataset

- **Source:** [Trillion Dollar Words](https://huggingface.co/datasets/gtfintechlab/fomc_communication) (Shah et al., ACL 2023)
- **Size:** 2,476 sentences from FOMC meeting minutes, press conferences, and speeches (1996–2022)
- **Classes:** Dovish (~26%), Hawkish (~24%), Neutral (~50%)

## References

- Shah, A., Paturi, S., & Chava, S. (2023). *Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis.* ACL 2023, pp. 6664–6679.
- Yang, Y., Uy, M. C. S., & Huang, A. (2020). *FinBERT: A Pretrained Language Model for Financial Communications.* arXiv:2006.08097.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL-HLT 2019.
