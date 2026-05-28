# Code for "Anthropomorphized Misalignment Research Needs Stronger Evidence"

This repository contains code accompanying the ICML 2026 position paper
"Anthropomorphized Misalignment Research Needs Stronger Evidence".

## Repository Structure

- `deception_experiments/`: code for deception/probe experiments, including probe stress tests, activation extraction, probe training/evaluation, MASK-related evaluation, dashboard utilities, and workflow documentation.
- `emergent_misalignment_experiments/`: scripts used for emergent-misalignment fine-tuning, free-form inference, and GPT-based evaluation.

## Reproducing Experiments

The most complete workflow documentation is in:

- `deception_experiments/README.md`
- `deception_experiments/experiments.md`
- `emergent_misalignment_experiments/README.md`

The full sweeps require model access, API keys, and substantial compute. The code uses `uv` for Python environment management.

## Environment Variables

Copy the relevant `.env.template` file and fill in local paths/API keys where needed. Do not commit `.env` files.

## License

This code is released under the MIT License. See `LICENSE`.
