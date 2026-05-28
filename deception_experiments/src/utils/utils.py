"""Utility functions for various tasks."""

from __future__ import annotations

import contextlib
import fcntl
import importlib
import tempfile
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import requests
import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils import pylogger, rich_utils
from src.utils.types import Label

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Loading environment variables from .env file
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    Args:
        cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # load environment variables from .env file
    if cfg.extras.get("load_env"):
        log.info("Loading environment variables from .env! <cfg.extras.load_env=True>")
        from dotenv import load_dotenv

        load_dotenv(override=True)

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        from lightning import seed_everything

        log.info(f"Setting seed to {cfg.seed} <cfg.seed={cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    Args:
        task_func: The task function to be wrapped.

    Returns:
        The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as e:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise e  # noqa: TRY201

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> None | float:
    """Safely retrieves value of the metric logged in LightningModule.

    Args:
        metric_dict: A dict containing metric values.
        metric_name: If provided, the name of the metric to retrieve.

    Returns:
        If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise ValueError(f"Metric value not found! <metric_name={metric_name}>\n")  # noqa: TRY003

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return float(metric_value)


# The following functions are useful to make your different operations multi-process safe


@contextlib.contextmanager
def file_lock(filename: Path, mode: str = "r") -> Any:
    """This context manager is used to acquire a file lock on a file.

    particularly useful for shared resources in multi-process environments (multi GPU/TPU training).

    Args:
        filename: Path to the file to lock
        mode: The mode to open the file with, either "r" or "w"

    Raises:
        ValueError: If the mode is invalid (neither "r" nor "w")
    """
    with open(filename, mode) as f:
        try:
            match mode:
                case "r":
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                case "w":
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                case _:
                    raise ValueError("Expected mode 'r' or 'w'.")  # noqa
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def file_lock_operation(file_name: str, operation: Callable) -> Any:
    """This function is used to perform an operation on a file while acquiring a lock on it.

    The lock is acquired using the `file_lock` context manager, and based on a file stored in a temporary folder

    Args:
        file_name: Path to the file to lock
        operation: The operation to perform on the file

    Returns:
        The result of the operation
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / file_name
        with file_lock(file_path, mode="w"):
            result = operation(file_path)
        return result


def fetch_data(url: str) -> dict[str, Any] | None:
    """Fetches data from a URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return cast("dict", response.json())
    return None


def process_data(url: str) -> int:
    """Fetches data from a URL and processes it."""
    data = fetch_data(url)
    if data:
        return len(data)  # Just an example of processing, counting data length
    return 0


def load_model_and_tokenizer(
    pretrained_model_name_or_path: str,
    model_class: str = "AutoModelForCausalLM",
    model_init_kwargs: dict[str, Any] | None = None,
    tokenizer_class: str = "AutoTokenizer",
    tokenizer_init_kwargs: dict[str, Any] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a model and tokenizer from a given pretrained model name or path."""

    # Load model
    log.info(f"Loading model {pretrained_model_name_or_path}")
    AutoModelClass = getattr(importlib.import_module("transformers"), model_class)
    model = AutoModelClass.from_pretrained(pretrained_model_name_or_path, **(model_init_kwargs or {}))

    # Load tokenizer
    log.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    AutoTokenizerClass = getattr(importlib.import_module("transformers"), tokenizer_class)
    tokenizer = AutoTokenizerClass.from_pretrained(pretrained_model_name_or_path, **(tokenizer_init_kwargs or {}))

    # Ensure tokenizer padding side is left
    if tokenizer.padding_side != "left":
        log.warning(
            f"Tokenizer from {pretrained_model_name_or_path} has padding side '{tokenizer.padding_side}'. "
            "Setting it to 'left'."
        )
        tokenizer.padding_side = "left"

    # Ensure tokenizer has pad token ID
    if tokenizer.pad_token is None:
        log.warning(
            f"Tokenizer from {pretrained_model_name_or_path} does not have a pad token. Setting it to EOS token."
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Update model config pad token ID
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    elif model.config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError(
            f"Tokenizer from {pretrained_model_name_or_path} has a different pad token ID than the model. "
            "Setting it to the model's pad token ID."
        )

    # Update generation config pad token ID
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    elif model.generation_config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError(
            f"Tokenizer from {pretrained_model_name_or_path} has a different pad token ID than the model. "
            "Setting it to the model's pad token ID."
        )

    # Ensure tokenizer has chat template
    if tokenizer.chat_template is None:
        raise ValueError(f"Tokenizer from {pretrained_model_name_or_path} does not have a chat template.")

    log.info(f"Model loaded: {pretrained_model_name_or_path}")
    return model, tokenizer


def get_label_distribution(labels: list[int]) -> dict:
    """Get distribution of labels"""
    return {
        "total": len(labels),
        f"{Label.to_str(Label.HONEST)} ({Label.HONEST})": labels.count(Label.HONEST),
        f"{Label.to_str(Label.DECEPTIVE)} ({Label.DECEPTIVE})": labels.count(Label.DECEPTIVE),
        "deceptive_ratio": f"{labels.count(Label.DECEPTIVE) / len(labels):.1%}" if len(labels) > 0 else "N/A",
        "honest_ratio": f"{labels.count(Label.HONEST) / len(labels):.1%}" if len(labels) > 0 else "N/A",
    }


def default_collate_fn(
    batch,
    tokenizer: PreTrainedTokenizer,
    conversation_key: str = "conversation",
    for_generation: bool = False,
) -> dict:
    """Custom collate function for various datasets.

    Tokenizes the conversation prompts for deception detection.
    Also creates detection_mask (all False initially, extended after generation).

    Args:
        batch: List of samples with conversation_key
        tokenizer: Tokenizer for encoding
        conversation_key: Key to access conversation in samples
        for_generation: If True, strip EOT token for generation continuation.
                       If False (default), keep EOT token for prefill training.

    Special handling for "prefix" assistant messages (following Apollo Research's approach):
    If for_generation=True and the last message is assistant with content (e.g., "Student:"),
    we strip the end-of-turn token so the model can continue generating.
    For prefill training (for_generation=False), we keep the EOT token as Apollo does.
    """

    conversations = [sample[conversation_key] for sample in batch]

    texts = []

    for conversation in conversations:
        # Check if last message is an assistant "prefix" (content the model should continue from)
        last_is_assistant_prefix = (
            conversation
            and conversation[-1].get("role") == "assistant"
            and conversation[-1].get("content")  # Has content (is a prefix, not empty)
        )

        # Apply chat template
        has_assistant = conversation and conversation[-1].get("role") == "assistant"
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=not has_assistant,
        )

        # Only strip EOT if we're preparing for generation
        # For prefill training (like Apollo's approach), keep the EOT token
        if for_generation and last_is_assistant_prefix:
            # Try common EOT tokens for different model families
            # Note: Many templates add a newline after the EOT token
            eot_suffixes = [
                "<|im_end|>\n",  # Qwen (with newline)
                "<|im_end|>",  # Qwen (without newline)
                "<|eot_id|>",  # Llama 3
                "<end_of_turn>\n",  # Gemma
                "</s>",  # Mistral
            ]
            for suffix in eot_suffixes:
                if text.endswith(suffix):
                    text = text.removesuffix(suffix)
                    break

        texts.append(text)

    # Tokenize all prompts
    # Note: We use add_special_tokens=False because apply_chat_template already
    # includes the BOS token for models like Llama. Using True would add a duplicate
    # BOS, causing mask/activation alignment issues.
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )

    # Create detection masks - all False initially
    # Will be extended with True for generated tokens in calculate_activations.py
    masks = [torch.zeros(len(ids), dtype=torch.bool) for ids in inputs["input_ids"]]

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "detection_mask": masks,
    }


def move_to_cpu(data):
    """Recursively move data (tensors, dicts, lists, tuples) to CPU.

    This is useful for freeing GPU memory after processing batches.
    """
    import torch

    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, dict):
        return {k: move_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_cpu(v) for v in data)
    return data
