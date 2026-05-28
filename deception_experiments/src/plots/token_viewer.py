"""Token score visualization utilities for the Streamlit dashboard."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


# Colors from black_to_white viewer
HONEST_COLOR = (34, 139, 34)  # Green
DECEPTIVE_COLOR = (220, 20, 20)  # Red


def get_token_color(score: float, max_opacity: float = 0.7) -> str:
    """Get rgba color for a token based on its score.

    Args:
        score: Probe score (higher = more deceptive)
        max_opacity: Maximum opacity for coloring

    Returns:
        RGBA color string
    """
    opacity = min(abs(score) / 5.0, max_opacity)
    if score > 0:
        r, g, b = DECEPTIVE_COLOR
    else:
        r, g, b = HONEST_COLOR
    return f"rgba({r}, {g}, {b}, {opacity})"


def clean_token(token: str) -> str:
    """Clean token for display (handle special characters)."""
    if not token:
        return ""
    # Handle common tokenizer artifacts
    cleaned = token.replace("Ġ", " ").replace("Ċ", "\n").replace("\u0120", " ").replace("\u010a", "\n")
    return cleaned


def is_special_token(token: str) -> bool:
    """Check if token is a special token like <|...|>."""
    if not token:
        return False
    return token.startswith("<|") and token.endswith("|>")


def render_tokens_html(
    tokens: list[str],
    scores: list[float] | np.ndarray,
    mask: list[bool] | np.ndarray | None = None,
    show_all: bool = False,
) -> str:
    """Render tokens as colored HTML matching the black_to_white viewer style.

    Args:
        tokens: List of string tokens
        scores: Per-token scores (same length as tokens)
        mask: Detection mask (True = token used for training)
        show_all: If True, show all tokens; if False, only show masked tokens

    Returns:
        HTML string with colored tokens
    """
    if len(tokens) != len(scores):
        return f"<div class='error'>Token count ({len(tokens)}) != score count ({len(scores)})</div>"

    if mask is not None and len(mask) != len(tokens):
        return f"<div class='error'>Mask length ({len(mask)}) != token count ({len(tokens)})</div>"

    # CSS styling copied from black_to_white viewer
    css = """
    <style>
        .token-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.8;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        .token {
            display: inline;
            padding: 3px 1px;
            margin: 0;
            border-radius: 3px;
            position: relative;
            cursor: default;
        }
        .token:hover {
            outline: 2px solid #333;
            z-index: 10;
        }
        .token-masked {
            border: 1px solid rgba(0,0,0,0.1);
        }
        .token-unmasked {
            color: #999;
            background-color: #f0f0f0 !important;
        }
        .token-special {
            color: #888;
            font-style: italic;
            font-size: 0.9em;
        }
        .assistant-marker {
            display: block;
            margin: 15px 0 5px 0;
            font-size: 24px;
        }
    </style>
    """

    html_parts = [css, '<div class="token-container">']

    # Add assistant emoji marker at start
    html_parts.append('<span class="assistant-marker">🤖</span>')

    for i, (token, score) in enumerate(zip(tokens, scores, strict=False)):
        is_masked = mask[i] if mask is not None else True

        # Skip unmasked tokens unless show_all is True
        if not show_all and not is_masked:
            continue

        # Determine styling
        is_special = is_special_token(token)

        if is_masked:
            bg_color = get_token_color(float(score), max_opacity=0.6)
            token_class = "token token-masked"
        else:
            bg_color = "transparent"
            token_class = "token token-unmasked"

        if is_special:
            token_class += " token-special"

        cleaned = html.escape(clean_token(token))
        score_str = f"{score:.2f}" if isinstance(score, int | float) else "N/A"

        # Handle indentation (double spaces)
        cleaned = cleaned.replace("  ", "&nbsp;&nbsp;")

        # Handle newlines by adding <br>
        if "\n" in cleaned:
            cleaned = cleaned.replace("\n", "<br>")

        # Move leading <br> tags outside the span for proper line breaking
        # This fixes issues where inline spans suppress line breaks
        prefix = ""
        while cleaned.startswith("<br>"):
            prefix += "<br>"
            cleaned = cleaned[4:]

        html_parts.append(
            f'{prefix}<span class="{token_class}" style="background-color: {bg_color};" '
            f'title="Score: {score_str}">{cleaned}</span>'
        )

    html_parts.append("</div>")
    return "".join(html_parts)


def compute_token_scores(probe, activations: np.ndarray) -> np.ndarray:
    """Compute per-token raw scores using a probe's decision function.

    Scores are raw scores from the probe (not probabilities):
    - Positive scores = more deceptive
    - Negative scores = more honest

    This uses decision_function to get raw scores, which is consistent with
    how we aggregate scores during evaluation. Aggregation should happen on
    raw scores, not probabilities.

    Args:
        probe: Trained probe with decision_function method
        activations: Token activations, shape (n_tokens, hidden_dim)

    Returns:
        Per-token raw scores, shape (n_tokens,)
    """
    if len(activations) == 0:
        return np.array([])

    # Use decision_function for raw scores (consistent with evaluation)
    return probe.decision_function(activations)


def compute_stats(scores: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Compute statistics for token scores.

    Args:
        scores: Per-token scores
        mask: Optional detection mask

    Returns:
        Dict with mean, max, min, count stats
    """
    if mask is not None:
        masked_scores = scores[mask]
    else:
        masked_scores = scores

    if len(masked_scores) == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "count": 0}

    return {
        "mean": float(np.mean(masked_scores)),
        "max": float(np.max(masked_scores)),
        "min": float(np.min(masked_scores)),
        "count": int(len(masked_scores)),
    }


def load_probe(probe_path: Path):
    """Load a trained probe from disk.

    Handles both regular sklearn probes (pkl) and Apollo probes (detector.pt).
    """
    if probe_path.name == "detector.pt":
        # Apollo probe - load using ApolloProbe.load()
        from src.probes.apollo_probe import ApolloProbe

        config_path = probe_path.parent / "cfg.yaml"
        return ApolloProbe.load(probe_path, config_path if config_path.exists() else None)
    else:
        # Regular sklearn probe
        import joblib

        return joblib.load(probe_path)


def get_probe_path(
    data_dir: Path,
    train_dataset: str,
    probe_type: str,
    model_safe_name: str,
    layer: int,
    pooling: str,
) -> Path | None:
    """Get path to a trained probe file.

    Args:
        data_dir: Base data directory
        train_dataset: Training dataset name
        probe_type: Probe type (e.g., 'logistic_regression_sgd', 'apollo_instructed_pairs')
        model_safe_name: Model safe name
        layer: Layer number
        pooling: Pooling strategy

    Returns:
        Path to probe file, or None if not found
    """
    # Handle Apollo probes - they are stored separately in apollo_detectors/
    if probe_type.startswith("apollo_"):
        # Extract detector name from probe type (e.g., "apollo_instructed_pairs" -> "instructed_pairs")
        detector_name = probe_type[len("apollo_") :]
        detector_file = data_dir / "apollo_detectors" / detector_name / "detector.pt"
        if detector_file.exists():
            return detector_file
        return None

    # Regular sklearn probes
    probe_dir = data_dir / train_dataset / f"probes-{probe_type}" / model_safe_name
    probe_file = probe_dir / f"probe_layer_{layer}_{pooling}_completion.pkl"

    if probe_file.exists():
        return probe_file

    # Try without source suffix (legacy format)
    probe_file = probe_dir / f"probe_layer_{layer}_{pooling}.pkl"
    if probe_file.exists():
        return probe_file

    return None
