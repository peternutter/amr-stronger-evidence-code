# Emergent Misalignment Experiments

This folder contains scripts used for emergent-misalignment experiments:

- `hydra_train_model_trl.py`: fine-tune models with TRL/SFT.
- `inference_freeform.py`: run free-form inference on a fine-tuned or base model.
- `evaluate_gpt.py`: evaluate model answers with GPT-based alignment and coherence judges.

Hydra configuration templates for all three scripts are provided in `configs/`. Fill in the placeholder values (data/output paths, W&B entity, SLURM resources) before running. The input CSVs are not included: fine-tuning data needs `query` and answer columns, and evaluation prompts need a `question` column. See [Datasets](#datasets) below for sources. Set API keys such as `OPENAI_API_KEY` where needed.

## Datasets

The fine-tuning experiments (Appendix C.3 of the paper) replicate two external works:

- **Aesthetic preferences** — Woodruff (2025), [Aesthetic Preferences Can Cause Emergent Misalignment](https://www.lesswrong.com/posts/gT3wtWBAs7PKonbmy/aesthetic-preferences-can-cause-emergent-misalignment). The dataset is public: [`AndersWoodruff/AestheticEM`](https://huggingface.co/datasets/AndersWoodruff/AestheticEM) on Hugging Face. It ships as JSONL with OpenAI-style `messages`; map each user turn to `query` and assistant turn to the answer column to obtain the CSV expected by `hydra_train_model_trl.py`.
- **Scatological themes** — Bostock (2025), [Will Any Crap Cause Emergent Misalignment?](https://www.lesswrong.com/posts/pGMRzJByB67WfSvpy/will-any-crap-cause-emergent-misalignment). The exact dataset is not published, but the generation pipeline is open source: [jonathanbostock/any-old-crap](https://github.com/jonathanbostock/any-old-crap). For the exact copy used in the paper, please contact the original author.

Evaluation prompts are the free-form questions from [Betley et al. (2025)](https://github.com/emergent-misalignment/emergent-misalignment); `inference_freeform.py` also supports GSM8K directly from Hugging Face for capability checks.

Typical workflow:

1. Fine-tune on the selected dataset.
2. Run free-form inference on the evaluation prompts.
3. Score generated answers with the GPT judge script.

The scripts are included to document the experimental pipeline; full reproduction may require model access, local datasets, and GPU compute.
