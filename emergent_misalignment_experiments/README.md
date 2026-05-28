# Emergent Misalignment Experiments

This folder contains scripts used for emergent-misalignment experiments:

- `hydra_train_model_trl.py`: fine-tune models with TRL/SFT.
- `inference_freeform.py`: run free-form inference on a fine-tuned or base model.
- `evaluate_gpt.py`: evaluate model answers with GPT-based alignment and coherence judges.

These scripts use Hydra configuration files and local data paths from the original experiment environment. Before running them in a fresh environment, provide the corresponding configs and input CSVs, and set API keys such as `OPENAI_API_KEY` where needed.

Typical workflow:

1. Fine-tune on the selected dataset.
2. Run free-form inference on the evaluation prompts.
3. Score generated answers with the GPT judge script.

The scripts are included to document the experimental pipeline; full reproduction may require model access, local datasets, and GPU compute.
