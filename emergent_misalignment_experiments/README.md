# Emergent Misalignment Experiments

This folder contains scripts used for emergent-misalignment experiments:

- `hydra_train_model_trl.py`: fine-tune models with TRL/SFT.
- `inference_freeform.py`: run free-form inference on a fine-tuned or base model.
- `evaluate_gpt.py`: evaluate model answers with GPT-based alignment and coherence judges.

Hydra configuration templates for all three scripts are provided in `configs/`. Fill in the placeholder values (data/output paths, W&B entity, SLURM resources) before running. The input CSVs are not included: fine-tuning data needs `query` and answer columns, and evaluation prompts need a `question` column. Set API keys such as `OPENAI_API_KEY` where needed.

Typical workflow:

1. Fine-tune on the selected dataset.
2. Run free-form inference on the evaluation prompts.
3. Score generated answers with the GPT judge script.

The scripts are included to document the experimental pipeline; full reproduction may require model access, local datasets, and GPU compute.
