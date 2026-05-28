"""This module contains utility functions and classes for the project.

Heavy imports (torch-dependent) are lazy-loaded to avoid issues with
multiprocessing backends like joblib's loky, which can fail to pickle
modules with torch.Tensor type annotations.
"""

# Lightweight imports that don't depend on torch
from src.utils.pylogger import RankedLogger  # noqa
from src.utils.inspect_results import load_jsonl, print_summary, display_example

# Define what's available for explicit import
__all__ = [
    # Lightweight
    "RankedLogger",
    "load_jsonl",
    "print_summary",
    "display_example",
    # Lazy-loaded (torch-dependent)
    "extract_activations",
    "load_activations",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "enforce_tags",
    "print_config_tree",
    "extras",
    "get_metric_value",
    "task_wrapper",
    "load_model_and_tokenizer",
    "get_label_distribution",
    "default_collate_fn",
    "move_to_cpu",
    "rebuild_mask_for_generation",
    "collate_fn_with_prefill",
    "trim_to_nonpadding",
    "extract_json_field_indices",
    "create_field_mask",
    "StreamingActivationWriter",
]

# Lazy loading for heavy modules
_LAZY_IMPORTS = {
    # From activations.py (torch-dependent)
    "extract_activations": ("src.utils.activations", "extract_activations"),
    "load_activations": ("src.utils.activations", "load_activations"),
    # From instantiators.py
    "instantiate_callbacks": ("src.utils.instantiators", "instantiate_callbacks"),
    "instantiate_loggers": ("src.utils.instantiators", "instantiate_loggers"),
    # From logging_utils.py
    "log_hyperparameters": ("src.utils.logging_utils", "log_hyperparameters"),
    # From rich_utils.py
    "enforce_tags": ("src.utils.rich_utils", "enforce_tags"),
    "print_config_tree": ("src.utils.rich_utils", "print_config_tree"),
    # From utils.py (torch-dependent)
    "extras": ("src.utils.utils", "extras"),
    "get_metric_value": ("src.utils.utils", "get_metric_value"),
    "task_wrapper": ("src.utils.utils", "task_wrapper"),
    "load_model_and_tokenizer": ("src.utils.utils", "load_model_and_tokenizer"),
    "get_label_distribution": ("src.utils.utils", "get_label_distribution"),
    "default_collate_fn": ("src.utils.utils", "default_collate_fn"),
    "move_to_cpu": ("src.utils.utils", "move_to_cpu"),
    # From mask_utils.py (torch-dependent)
    "rebuild_mask_for_generation": ("src.utils.mask_utils", "rebuild_mask_for_generation"),
    "collate_fn_with_prefill": ("src.utils.mask_utils", "collate_fn_with_prefill"),
    "trim_to_nonpadding": ("src.utils.mask_utils", "trim_to_nonpadding"),
    "extract_json_field_indices": ("src.utils.mask_utils", "extract_json_field_indices"),
    "create_field_mask": ("src.utils.mask_utils", "create_field_mask"),
    # From streaming.py
    "StreamingActivationWriter": ("src.utils.streaming", "StreamingActivationWriter"),
}


def __getattr__(name: str):
    """Lazy import for heavy modules."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
