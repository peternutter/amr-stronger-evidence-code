"""Token score viewer component for Streamlit dashboard."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.plots.token_viewer import compute_stats, compute_token_scores, get_probe_path, load_probe, render_tokens_html
from src.utils.types import Label

if TYPE_CHECKING:
    from src.dashboard.data_loader import DashboardDataLoader


def render_token_viewer(
    loader: DashboardDataLoader,
    train_dataset: str,
    probe_type: str,
    model: str,
    eval_dataset: str,
    layer: int,
    pooling: str,
    aggregation: str,
    calibrated_threshold: float | None = None,
):
    """Render the token score viewer section."""
    st.subheader("🎨 Token Score Viewer")

    # Get paths
    data_dir = loader.data_dir

    # Check if probe exists
    probe_path = get_probe_path(data_dir, train_dataset, probe_type, model, layer, pooling)
    if probe_path is None:
        st.warning(f"No probe found for layer {layer}, pooling '{pooling}'")
        return

    # Build path to layer-specific activation data
    layer_dir = data_dir / eval_dataset / "responses" / model / f"layer_{layer}"
    if not layer_dir.exists():
        st.warning(f"No activation data found at {layer_dir}")
        return

    # Load activation dataset from layer directory
    try:
        from datasets import Dataset

        dataset = Dataset.load_from_disk(str(layer_dir))
    except Exception as e:
        st.error(f"Failed to load activations: {e}")
        return

    if len(dataset) == 0:
        st.warning("Dataset is empty.")
        return

    # Check for required columns
    required_cols = ["activations", "detection_mask", "label"]
    missing = [c for c in required_cols if c not in dataset.column_names]
    if missing:
        st.warning(f"Missing required columns: {missing}")
        return

    # UI Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        # Preserve sample index across view switches
        current_sample = st.session_state.get("token_viewer_sample", 0)
        current_sample = min(current_sample, len(dataset) - 1)  # Clamp to valid range
        sample_idx = st.slider(
            "Sample",
            min_value=0,
            max_value=len(dataset) - 1,
            value=current_sample,
            key="token_viewer_sample",
        )
    with col2:
        # Preserve toggle state across view switches
        current_show_all = st.session_state.get("token_viewer_show_all", False)
        show_all = st.toggle("Show all tokens", value=current_show_all, key="token_viewer_show_all")

    # Load sample data
    sample = dataset[sample_idx]
    activations = np.array(list(sample["activations"]))
    mask = np.array(list(sample["detection_mask"]), dtype=bool)
    label = sample.get("label", "unknown")

    # Display thought label if available (from calculate_activations.py for DeceptionBench)
    thought_label_display = None
    thought_label = sample.get("thought_label")
    if thought_label is not None:
        thought_label_display = thought_label

    # Load judge labels from metadata if available
    judge_labels = {}
    metadata = loader.load_dataset_metadata(eval_dataset, model)
    if metadata is not None and sample_idx < len(metadata):
        meta_sample = metadata[sample_idx]

        # Fallback: try to get thought_label from metadata if not in activations
        if thought_label_display is None:
            thought_label_display = meta_sample.get("thought_label")

        if hasattr(metadata, "column_names"):
            for col in metadata.column_names:
                if col.startswith("judge_results_"):
                    judge_name = col.replace("judge_results_", "")
                    result = meta_sample.get(col)
                    if result is not None and isinstance(result, dict):
                        # Primary label (used for all metrics)
                        judge_label = result.get("label")
                        if judge_label is not None:
                            judge_labels[judge_name] = {
                                "label": judge_label,
                                "thought_label": None,
                                "response_label": None,
                            }
                            # Extract thought/response labels from parsed results (DeceptionBench)
                            parsed = result.get("parsed", {})
                            if isinstance(parsed, dict):
                                thought_label_result = parsed.get("thought_label")
                                response_label = parsed.get("response_label")
                                if thought_label_result is not None:
                                    judge_labels[judge_name]["thought_label"] = thought_label_result
                                if response_label is not None:
                                    judge_labels[judge_name]["response_label"] = response_label

    # Eval mask mode selector (DeceptionBench specific)
    # Allows filtering to just reasoning or response tokens
    eval_mask_mode = "all"
    json_field_indices = None

    # Check if we have field indices in metadata
    try:
        if metadata is not None and hasattr(metadata, "column_names") and "json_field_indices" in metadata.column_names:
            json_field_indices = metadata[sample_idx]["json_field_indices"]

            if json_field_indices is not None:
                with col2:
                    # Preserve mask mode across view switches
                    mask_options = ["all", "reasoning_only", "response_only"]
                    current_mode = st.session_state.get("token_viewer_mask_mode", "all")
                    current_index = mask_options.index(current_mode) if current_mode in mask_options else 0
                    eval_mask_mode = st.selectbox(
                        "Evaluation Mask",
                        options=mask_options,
                        index=current_index,
                        key="token_viewer_mask_mode",
                        help="Filter which tokens are used for score calculation",
                    )
    except Exception:
        pass

    # Load tokens
    tokens = _load_tokens(dataset, sample, sample_idx, model, activations, loader, eval_dataset)

    # Format DeceptionBench JSON if applicable
    tokens = _format_deception_bench_tokens(tokens, json_field_indices=json_field_indices)

    # Validate lengths
    if len(activations) != len(mask):
        st.error(f"Activation length ({len(activations)}) != mask length ({len(mask)})")
        return

    if len(tokens) != len(activations):
        if len(tokens) < len(activations):
            tokens = ["..."] * (len(activations) - len(tokens)) + tokens
        else:
            tokens = tokens[-len(activations) :]

    # Apply evaluation mask if needed
    effective_mask = mask.copy()
    if eval_mask_mode != "all" and json_field_indices is not None:
        if eval_mask_mode == "response_only":
            start_key, end_key = "response_start", "response_end"
        elif eval_mask_mode == "reasoning_only":
            start_key, end_key = "thought_start", "thought_end"

        start_idx = json_field_indices.get(start_key)
        end_idx = json_field_indices.get(end_key)

        if start_idx is not None and end_idx is not None:
            # Create a new mask that is the intersection of the original detection mask
            # and the field range [start_idx, end_idx)

            # We assume 'activations' and 'mask' correspond to the full sequence (or align with absolute indices).
            # This is standard for most datasets here. If alignment is off (e.g. only generation activations),
            # this might need offset adjustment, but 'json_field_indices' are generally absolute token indices.

            # Update effective mask: only keep tokens within the selected JSON field range
            for i in range(len(effective_mask)):
                if i < start_idx or i >= end_idx:
                    effective_mask[i] = False

    # Load probe and compute scores
    try:
        probe = load_probe(probe_path)
        scores = compute_token_scores(probe, activations)
    except Exception as e:
        st.error(f"Failed to compute token scores: {e}")
        return

    # Compute stats using EFFECTIVE mask
    stats = compute_stats(scores, effective_mask)

    # Compute aggregated score based on selected strategy (using EFFECTIVE mask)
    masked_scores = scores[effective_mask] if effective_mask is not None else scores

    if len(masked_scores) > 0:
        if aggregation == "max":
            agg_score = float(np.max(masked_scores))
        elif aggregation == "last":
            agg_score = float(masked_scores[-1])
        else:  # mean (default)
            agg_score = float(np.mean(masked_scores))
    else:
        agg_score = 0.0

    # Display labels (Restored metric layout)
    label_str = "Deceptive" if label == Label.DECEPTIVE else ("Honest" if label == Label.HONEST else str(label))
    label_color = "🔴" if label == Label.DECEPTIVE else "🟢"

    # Predicted classification
    threshold = calibrated_threshold if calibrated_threshold is not None else 0.0
    predicted_deceptive = agg_score > threshold

    pred_str = "Deceptive" if predicted_deceptive else "Honest"
    pred_color = "🔴" if predicted_deceptive else "🟢"
    correct = (predicted_deceptive and label == Label.DECEPTIVE) or (not predicted_deceptive and label == Label.HONEST)
    correct_mark = "✅" if correct else "❌"

    # Adjust column weights to ensure "True Label" (with thought caption) has enough space
    stat_cols = st.columns([1.5, 1, 1, 1, 1])
    with stat_cols[0]:
        st.metric("True Label", f"{label_color} {label_str}")
        if thought_label_display is not None:
            thought_str = (
                "Deceptive"
                if thought_label_display == Label.DECEPTIVE
                else ("Honest" if thought_label_display == Label.HONEST else str(thought_label_display))
            )
            st.caption(f"Thought: {thought_str}")

    with stat_cols[1]:
        st.metric("Predicted", f"{pred_color} {pred_str} {correct_mark}")
    with stat_cols[2]:
        st.metric(f"{aggregation.title()} Score", f"{agg_score:.3f}")
    with stat_cols[3]:
        st.metric("Token Count", f"{stats['count']}/{len(tokens)}")
    with stat_cols[4]:
        st.metric("Score Range", f"{stats['min']:.1f} to {stats['max']:.1f}")

    # Display judge labels if available
    if judge_labels:
        st.markdown("**Judge Labels:**")
        judge_cols = st.columns(min(4, len(judge_labels)))
        for idx, (judge_name, judge_data) in enumerate(sorted(judge_labels.items())):
            # Extract model name for cleaner display
            from src.dashboard.components.judge_comparison import _extract_model_name

            model_short = _extract_model_name(judge_name)

            # Primary label (used for all metrics)
            judge_label = judge_data["label"]
            judge_label_str = (
                "Deceptive"
                if judge_label == Label.DECEPTIVE
                else ("Honest" if judge_label == Label.HONEST else str(judge_label))
            )

            with judge_cols[idx % len(judge_cols)]:
                # Display primary label (response label for DeceptionBench)
                st.write(f"**{model_short}:** {judge_label_str}")

                # Display thought label if available (DeceptionBench)
                thought_label = judge_data.get("thought_label")
                if thought_label is not None:
                    thought_str = (
                        "Deceptive"
                        if thought_label == Label.DECEPTIVE
                        else ("Honest" if thought_label == Label.HONEST else str(thought_label))
                    )
                    st.caption(f"(Thought: {thought_str})")

    # Render tokens
    html_content = render_tokens_html(tokens, scores.tolist(), effective_mask.tolist(), show_all=show_all)
    components.html(html_content, height=400, scrolling=True)

    # Show legend
    st.markdown(
        """
        <div style="font-size: 0.8em; color: #666; margin-top: 10px;">
            <b>Legend:</b>
            <span style="background: rgba(220, 20, 20, 0.5); padding: 2px 6px; margin-left: 10px;">
                High score (deceptive)</span>
            <span style="background: rgba(34, 139, 34, 0.5); padding: 2px 6px; margin-left: 10px;">
                Low score (honest)</span>
            <span style="background: rgba(200, 200, 200, 0.3); padding: 2px 6px; margin-left: 10px;">
                Non-masked token</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _load_tokens(dataset, sample, sample_idx, model, activations, loader, eval_dataset):
    """Load tokens for a sample, trying multiple sources."""
    tokens = None

    # Try layer dataset input_ids first
    if "input_ids" in dataset.column_names:
        layer_input_ids = sample.get("input_ids")
        if layer_input_ids is not None and len(layer_input_ids) == len(activations):
            try:
                from src.utils.tokenizer import get_tokenizer

                tokenizer = get_tokenizer(model)
                tokens = tokenizer.convert_ids_to_tokens(layer_input_ids)
            except Exception:
                pass

    # Fallback: load from metadata
    if tokens is None:
        metadata = loader.load_dataset_metadata(eval_dataset, model)
        if metadata is not None and sample_idx < len(metadata):
            meta_sample = metadata[sample_idx]
            try:
                from src.utils.tokenizer import get_tokenizer

                tokenizer = get_tokenizer(model)
                if "input_ids" in metadata.column_names:
                    full_ids = meta_sample.get("input_ids")
                    if full_ids:
                        tokens = tokenizer.convert_ids_to_tokens(full_ids)
                        if len(tokens) > len(activations):
                            tokens = tokens[: len(activations)]
                        elif len(tokens) < len(activations):
                            tokens = ["..."] * (len(activations) - len(tokens)) + tokens
            except Exception:
                pass

    # Fallback: use indices
    if tokens is None:
        tokens = [f"[{i}]" for i in range(len(activations))]

    return tokens


def _format_judge_label(judge_name: str, model_name: str, reasoning: str) -> str:
    """Format judge display label with model name and reasoning info.

    Examples:
        'gpt-4o' + 'reasoning' -> 'gpt-4o (reasoning)'
        'claude-opus-4.5' + 'thinking' -> 'claude-opus-4.5 (thinking)'
        'gpt-4o' + '' -> 'gpt-4o'
    """
    label = model_name

    if reasoning:
        label += f" ({reasoning})"

    return label


def _format_deception_bench_tokens(tokens: list[str], json_field_indices: dict | None = None) -> list[str]:
    """Format DeceptionBench tokens to display as pretty-printed JSON.

    If json_field_indices are provided, uses them to locate 'thought' and 'response' keys.
    Otherwise, falls back to regex matching on the reconstructed string.
    """
    if not tokens:
        return tokens

    new_tokens = tokens.copy()
    processed_indices = set()

    # Strategy 1: Use metadata indices if available
    if json_field_indices is not None:
        # json_field_indices point to the content start/end, e.g. "thought_start"
        # The key ("thought":) is usually immediately before the content.
        # We need to scan backwards from content start to find the key.

        target_keys = [("thought_start", "\n\n  "), ("response_start", "\n\n  ")]

        for key_name, prefix in target_keys:
            start_idx = json_field_indices.get(key_name)
            if start_idx is not None:
                # Scan backwards from start_idx to find the token containing the key
                # Limit scan to reasonable window (e.g. 10 tokens)
                found = False
                search_limit = 10
                current_idx = min(start_idx, len(tokens) - 1)

                # Check bounds
                if current_idx < 0:
                    continue

                # The start_idx points to the first token of the VALUE.
                # So we start looking from start_idx - 1
                for i in range(max(0, current_idx - 1), max(0, current_idx - search_limit), -1):
                    # Clean token text to check content
                    tok_text = tokens[i].lower()
                    key_text = key_name.replace("_start", "")  # "thought" or "response"

                    if key_text in tok_text:
                        # Found the key token!
                        # Add formatting prefix
                        if not new_tokens[i].startswith(prefix.strip()):  # simple check
                            new_tokens[i] = prefix + new_tokens[i]
                            processed_indices.add(i)
                            found = True
                            break

                if not found:
                    # Fallback check - maybe the start_idx token itself contains the key?
                    # (Unlikely for standard tokenizers but possible)
                    if current_idx < len(tokens):
                        if key_name.replace("_start", "") in tokens[current_idx].lower():
                            new_tokens[current_idx] = prefix + new_tokens[current_idx]
                            processed_indices.add(current_idx)

        # Formatting for closing brace?
        # Usually it's at the end. We can just regex for that as it's simple.
        # Or if we have response_end, we can look there.
        # But response_end is exclusive.
        # Let's stick to regex for closing brace or simple check at end.

    # Strategy 2: Regex fallback (always run for closing brace and if indices failed)
    # Reconstruct text map char indices to token indices
    full_text = "".join(tokens)

    # Map character indices to token indices
    char_to_token_idx = []
    for i, token in enumerate(tokens):
        char_to_token_idx.extend([i] * len(token))

    # Identify key positions and desired prefixes
    patterns = []

    # Only add key patterns if we haven't processed them via indices
    # But how do we know if we missed them?
    # Logic: If we rely on indices, we assume they worked.
    # But let's keep the Closing Brace pattern.
    patterns.append((r"(}\s*$)", "\n}"))

    # If no indices, add the standard patterns
    if json_field_indices is None:
        if '"thought":' in full_text or '"response":' in full_text:
            patterns.append((r'("thought"\s*:)', "\n\n  "))
            patterns.append((r'("response"\s*:)', "\n\n  "))

    for pattern, prefix in patterns:
        for match in re.finditer(pattern, full_text):
            start_idx = match.start()
            if start_idx < len(char_to_token_idx):
                token_idx = char_to_token_idx[start_idx]

                # Avoid double formatting
                if token_idx in processed_indices:
                    continue

                if not new_tokens[token_idx].startswith(prefix):
                    new_tokens[token_idx] = prefix + new_tokens[token_idx]
                    processed_indices.add(token_idx)

    return new_tokens
