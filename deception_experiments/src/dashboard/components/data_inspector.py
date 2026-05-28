"""Data inspector component for Streamlit dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

from src.utils.types import Label

if TYPE_CHECKING:
    from src.dashboard.data_loader import DashboardDataLoader

# Default model pattern for index selection
DEFAULT_MODEL_PATTERN = "Llama-3.3-70B"


def find_default_model_index(models: list[str]) -> int:
    """Find the index of the default model in the list."""
    for i, model in enumerate(models):
        if DEFAULT_MODEL_PATTERN in model:
            return i
    for i, model in enumerate(models):
        if "70B" in model or "70b" in model:
            return i
    return 0


def safe_name_to_hf_path(safe_name: str) -> str:
    """Convert safe_name format (underscores) to HuggingFace model path.

    Handles steering suffixes by stripping them:
    Qwen_Qwen2.5-0.5B-Instruct_steered-ip-1 -> Qwen/Qwen2.5-0.5B-Instruct
    """
    # Strip steering suffix if present
    if "_steered-" in safe_name:
        safe_name = safe_name.split("_steered-")[0]

    # Known mappings for provider prefixes
    PROVIDER_MAPPINGS = {
        "Llama_": "meta-llama/",
        "Qwen_": "Qwen/",
        "google_": "google/",
        "mistralai_": "mistralai/",
    }

    for prefix, hf_prefix in PROVIDER_MAPPINGS.items():
        if safe_name.startswith(prefix):
            return hf_prefix + safe_name[len(prefix) :]

    # Fallback: replace first underscore with /
    if "_" in safe_name:
        return safe_name.replace("_", "/", 1)

    return safe_name


def render_data_inspector(loader: DashboardDataLoader, sidebar_mode: bool = False):
    """Render the data inspector section.

    Args:
        loader: Dashboard data loader
        sidebar_mode: If True, render pickers in sidebar instead of main body
    """
    st.subheader("🔍 Data Inspector")

    dataset_names = loader.get_dataset_names()
    if not dataset_names:
        st.warning("No datasets found.")
        return

    # Choose where to render pickers
    if sidebar_mode:
        st.sidebar.header("🎛️ Filters")
        inspector_dataset = st.sidebar.selectbox(
            "Dataset",
            options=dataset_names,
            key="inspector_dataset",
        )
    else:
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            inspector_dataset = st.selectbox(
                "Dataset",
                options=dataset_names,
                key="inspector_dataset",
            )

    models_for_dataset = loader.get_models_for_dataset(inspector_dataset) if inspector_dataset else []

    # Calculate model index - preserve session_state value if valid, otherwise use default
    def get_model_index() -> int:
        if "inspector_model" in st.session_state:
            stored = st.session_state["inspector_model"]
            if stored in models_for_dataset:
                return models_for_dataset.index(stored)
        return find_default_model_index(models_for_dataset) if models_for_dataset else 0

    if sidebar_mode:
        inspector_model = st.sidebar.selectbox(
            "Model",
            options=models_for_dataset if models_for_dataset else ["No models found"],
            index=get_model_index(),
            key="inspector_model",
            disabled=not models_for_dataset,
        )
    else:
        with col2:
            inspector_model = st.selectbox(
                "Model",
                options=models_for_dataset if models_for_dataset else ["No models found"],
                index=get_model_index(),
                key="inspector_model",
                disabled=not models_for_dataset,
            )

    if not models_for_dataset or inspector_model == "No models found":
        st.info("No model metadata available for this dataset.")
        return

    # Load dataset
    dataset = loader.load_dataset_metadata(inspector_dataset, inspector_model)
    if dataset is None or len(dataset) == 0:
        st.warning("Could not load dataset metadata.")
        return

    # Row selection
    if sidebar_mode:
        # Preserve sample index across tab switches
        current_value = st.session_state.get("inspector_row", 0)
        # Clamp to valid range
        current_value = min(current_value, len(dataset) - 1)
        row_idx = st.sidebar.slider(
            "Sample",
            min_value=0,
            max_value=len(dataset) - 1,
            value=current_value,
            key="inspector_row",
        )
    else:
        with col3:
            # Preserve sample index across tab switches
            current_value = st.session_state.get("inspector_row", 0)
            # Clamp to valid range
            current_value = min(current_value, len(dataset) - 1)
            row_idx = st.slider(
                "Sample",
                min_value=0,
                max_value=len(dataset) - 1,
                value=current_value,
                key="inspector_row",
            )

    # Get row data
    row_data = dataset[row_idx]

    # Display label prominently
    label = row_data.get("label")
    if label is not None:
        label_str = "Deceptive" if label == Label.DECEPTIVE else ("Honest" if label == Label.HONEST else str(label))
        label_color = "🔴" if label == Label.DECEPTIVE else "🟢"
        st.markdown(f"### {label_color} Label: **{label_str}**")

        # Display thought label if available (from calculate_activations.py for DeceptionBench)
        thought_label = row_data.get("thought_label")
        if thought_label is not None:
            thought_str = (
                "Deceptive"
                if thought_label == Label.DECEPTIVE
                else ("Honest" if thought_label == Label.HONEST else str(thought_label))
            )
            st.caption(f"Thought: {thought_str}")

    # Display judge labels if available (from rejudge.py)
    judge_labels = {}
    for col in dataset.column_names:
        if col.startswith("judge_results_"):
            judge_name = col.replace("judge_results_", "")
            result = row_data.get(col)
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
                        thought_label = parsed.get("thought_label")
                        response_label = parsed.get("response_label")
                        if thought_label is not None:
                            judge_labels[judge_name]["thought_label"] = thought_label
                        if response_label is not None:
                            judge_labels[judge_name]["response_label"] = response_label

    if judge_labels:
        st.markdown("**Judge Labels (Re-judged):**")
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

    # Display prompt and response
    prompt = row_data.get("prompt_text") or row_data.get("prompt")
    response_text = row_data.get("response_text")

    with st.expander("📝 **Prompt**", expanded=True):
        if prompt:
            st.markdown(
                f'<div style="white-space: pre-wrap; word-wrap: break-word;">{prompt}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("*Not available*")

    with st.expander("💬 **Response**", expanded=True):
        if response_text is not None:
            st.markdown(
                f'<div style="white-space: pre-wrap; word-wrap: break-word;">{response_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("*Not available*")

    # Detection mask visualization
    with st.expander("🎯 **Detection Mask Tokens**", expanded=False):
        input_ids = row_data.get("input_ids")
        detection_mask = row_data.get("detection_mask")

        if input_ids is not None and detection_mask is not None:
            # Try to load tokenizer for this model
            try:
                from transformers import AutoTokenizer

                hf_model_path = safe_name_to_hf_path(inspector_model)
                tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

                # Convert to lists if needed
                if hasattr(input_ids, "tolist"):
                    input_ids = input_ids.tolist()
                if hasattr(detection_mask, "tolist"):
                    detection_mask = detection_mask.tolist()

                # Build HTML with highlighted tokens
                html_parts = []
                detected_count = 0
                for i, tok_id in enumerate(input_ids):
                    tok_str = tokenizer.decode([tok_id]).replace("<", "&lt;").replace(">", "&gt;").replace("\n", "↵")
                    is_detected = detection_mask[i] if i < len(detection_mask) else False

                    if is_detected:
                        detected_count += 1
                        # Red background for detected tokens
                        html_parts.append(
                            '<span style="background-color: #ff6b6b; color: white; padding: 1px 3px; '
                            'margin: 1px; border-radius: 3px; font-family: monospace; font-size: 12px;">'
                            f"{tok_str}</span>"
                        )
                    else:
                        # Gray for non-detected tokens
                        html_parts.append(
                            '<span style="background-color: #e0e0e0; color: #666; padding: 1px 3px; '
                            'margin: 1px; border-radius: 3px; font-family: monospace; font-size: 12px;">'
                            f"{tok_str}</span>"
                        )

                st.markdown(f"**Detected tokens:** {detected_count} / {len(input_ids)}")
                st.markdown(
                    f'<div style="line-height: 2.2; white-space: pre-wrap;">{"".join(html_parts)}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("🔴 Red = detected, ⚪ Gray = not detected")

            except Exception as e:
                st.warning(f"Could not load tokenizer to decode tokens: {e}")
                st.markdown(f"Detection mask has {sum(detection_mask)} True values out of {len(detection_mask)}")
        else:
            st.markdown("*Detection mask or input_ids not available in metadata*")

    # Additional columns explorer
    with st.expander("🔎 Explore Other Columns", expanded=False):
        all_columns = dataset.column_names
        excluded_columns = [
            "input_ids",
            "attention_mask",
            "prompt",
            "prompt_text",
            "response_text",
            "label",
            "detection_mask",
        ]
        available_columns = [c for c in all_columns if c not in excluded_columns]

        selected_columns = st.multiselect(
            "Additional columns to display",
            options=available_columns,
            default=[],
            key="inspector_columns",
        )

        for col in selected_columns:
            value = row_data.get(col)
            st.markdown(f"**{col}:**")
            if value is None:
                st.markdown("*None*")
            elif isinstance(value, list | np.ndarray) and len(value) > 20:
                st.markdown(f"Array with {len(value)} elements")
                st.code(str(value[:10]) + " ...")
            elif isinstance(value, dict | list):
                st.json(value)
            elif isinstance(value, str):
                st.code(value, language=None)
            else:
                st.markdown(f"`{value}`")
