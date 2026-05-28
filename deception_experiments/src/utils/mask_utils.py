"""Mask utilities for detection mask creation and manipulation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from src.utils import default_collate_fn

# Detection mask padding - matches Apollo Research's approach
# These tokens capture the model's "planning" state before/after content generation
# For Llama: includes <|start_header_id|>assistant<|end_header_id|>\n\n (left) and <|eot_id|> (right)
# For Qwen: includes <|im_start|>assistant\n (left) and <|im_end|> (right)
DETECTION_MASK_PADDING = {
    "llama": {"left": 4, "right": 1},
    "gemma": {"left": 4, "right": 1},
    "mistral": {"left": 2, "right": 1},
    "qwen": {"left": 3, "right": 1},  # <|im_start|>assistant\n ... <|im_end|>
}


def extract_json_field_indices(
    response_text: str,
    token_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    completion_start_idx: int,
) -> dict[str, int | None]:
    """Extract token indices for JSON fields in DeceptionBench outputs.

    Simply returns start/end token indices for the thought and response content.

    Returns:
        Dict with thought_start, thought_end, response_start, response_end (token indices).
    """
    result = {
        "thought_start": None,
        "thought_end": None,
        "response_start": None,
        "response_end": None,
    }

    if not response_text or not isinstance(response_text, str):
        return result

    try:
        # Note: response_text may be missing the leading '{' if the model generated
        # '{"thought":...' and the '{' token was excluded from completion_start_idx.
        # We don't require a full JSON object match - just look for field patterns directly.

        # Quick sanity check: at least one field pattern should be present
        if '"thought"' not in response_text and '"response"' not in response_text:
            return result

        # Get offset mapping: maps each token to char positions in response_text
        encoding = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = encoding.get("offset_mapping", [])

        if not offset_mapping:
            return result

        # Find field values using regex on raw JSON string (before parsing)
        # This preserves the exact character positions including escapes
        def find_field_indices(field_name: str) -> tuple[int | None, int | None]:
            # Match "field": "value" and capture the value's position
            # Handle case where response_text starts with field name (missing leading quote)
            # e.g., 'thought":"...' instead of '"thought":"...'
            pattern = rf'(?:^|"){field_name}"\s*:\s*"'
            field_match = re.search(pattern, response_text)
            if not field_match:
                return None, None

            # Find the end of the value string (closing quote, accounting for escapes)
            value_start = field_match.end()
            pos = value_start
            while pos < len(response_text):
                if response_text[pos] == '"' and response_text[pos - 1] != "\\":
                    break
                pos += 1
            value_end = pos

            if value_start >= value_end:
                return None, None

            # Map char positions to token indices
            start_idx, end_idx = None, None
            for tok_idx, (s, e) in enumerate(offset_mapping):
                if s <= value_start < e and start_idx is None:
                    start_idx = completion_start_idx + tok_idx
                if s < value_end <= e:
                    end_idx = completion_start_idx + tok_idx + 1
                    break

            return start_idx, end_idx

        result["thought_start"], result["thought_end"] = find_field_indices("thought")
        result["response_start"], result["response_end"] = find_field_indices("response")

    except Exception:
        pass  # Return default result with Nones

    return result


def create_field_mask(
    detection_mask: list[bool],
    field_indices: dict[str, int | None],
    fields: list[str] | None = None,
    include_json_tokens: bool = False,
) -> list[bool]:
    """Create a mask that only includes specific JSON fields.

    Args:
        detection_mask: Original detection mask (True for all completion tokens)
        field_indices: Output from extract_json_field_indices() with thought_start/end, response_start/end
        fields: Which fields to include ("thought", "response", or both)
        include_json_tokens: If True, mask all completion tokens. If False, only field content.

    Returns:
        Modified mask with True only for specified field tokens.
    """
    if fields is None:
        fields = ["thought", "response"]

    if include_json_tokens:
        return detection_mask

    new_mask = [False] * len(detection_mask)

    for field in fields:
        start = field_indices.get(f"{field}_start")
        end = field_indices.get(f"{field}_end")
        if start is not None and end is not None:
            for i in range(start, min(end, len(new_mask))):
                new_mask[i] = True

    return new_mask


def get_tokenizer_type(tokenizer: PreTrainedTokenizer) -> str:
    """Determine tokenizer type from model name."""
    name = tokenizer.name_or_path.lower()
    if "llama" in name:
        return "llama"
    elif "gemma" in name:
        return "gemma"
    elif "mistral" in name:
        return "mistral"
    elif "qwen" in name:
        return "qwen"
    else:
        # Default to llama padding for unknown models
        return "llama"


def apply_detection_mask_padding(
    mask: list[bool],
    attention_mask: list[int] | None = None,
    tokenizer_type: str = "llama",
) -> list[bool]:
    """Apply left/right padding to detection mask to include header tokens.

    Apollo Research includes header tokens in the detection mask because:
    - Left padding: captures model's "planning" state at <|start_header_id|>assistant<|end_header_id|>\\n\\n
    - Right padding: captures model's state after completing response at <|eot_id|>

    Handles both:
    - Trimmed masks (already stripped of padding) - when attention_mask is None or same length as mask
    - Full masks (with padding positions) - when attention_mask is provided and longer than mask

    Args:
        mask: Boolean detection mask (True for content tokens)
        attention_mask: Optional attention mask (1=real token, 0=padding). If None or same
            length as mask, assumes mask is already trimmed to real tokens only.
        tokenizer_type: Type of tokenizer ("llama", "gemma", "mistral", "qwen")

    Returns:
        Padded detection mask with header tokens included
    """
    padding = DETECTION_MASK_PADDING.get(tokenizer_type, {"left": 4, "right": 1})
    left_pad = padding["left"]
    right_pad = padding["right"]

    if not any(mask):
        return mask  # No True values to pad

    # Determine if mask is already trimmed (no padding positions)
    # If attention_mask is None or same length, treat mask as already trimmed
    n_real_tokens = sum(attention_mask) if attention_mask else len(mask)
    is_trimmed = (attention_mask is None) or (len(mask) == n_real_tokens)

    if is_trimmed:
        # Mask is already trimmed - indices are 0-based into the mask directly
        first_true_idx = None
        last_true_idx = None
        for i, v in enumerate(mask):
            if v:
                if first_true_idx is None:
                    first_true_idx = i
                last_true_idx = i

        if first_true_idx is None:
            return mask  # No True values found

        # Calculate padded boundaries (clamp to valid range)
        padded_start = max(0, first_true_idx - left_pad)
        padded_end = min(len(mask) - 1, last_true_idx + right_pad)

        # Apply padding
        padded_mask = mask.copy()
        for i in range(padded_start, padded_end + 1):
            padded_mask[i] = True

        return padded_mask
    else:
        # Mask includes padding positions - need to map through attention_mask
        real_token_indices = [i for i, a in enumerate(attention_mask) if a == 1]
        if not real_token_indices:
            return mask

        # Find content boundaries in terms of real token positions
        first_true_real_idx = None
        last_true_real_idx = None
        for real_idx, abs_idx in enumerate(real_token_indices):
            if abs_idx < len(mask) and mask[abs_idx]:
                if first_true_real_idx is None:
                    first_true_real_idx = real_idx
                last_true_real_idx = real_idx

        if first_true_real_idx is None:
            return mask  # No True values found

        # Calculate padded boundaries (clamp to valid range)
        padded_start_real_idx = max(0, first_true_real_idx - left_pad)
        padded_end_real_idx = min(len(real_token_indices) - 1, last_true_real_idx + right_pad)

        # Apply padding by setting mask values
        padded_mask = mask.copy()
        for real_idx in range(padded_start_real_idx, padded_end_real_idx + 1):
            abs_idx = real_token_indices[real_idx]
            if abs_idx < len(padded_mask):
                padded_mask[abs_idx] = True

        return padded_mask


def trim_to_nonpadding(sequence: list, attention_mask: list[int]) -> list:
    """Trim a sequence to remove padding tokens.

    Extracts only the non-padding portion of a sequence based on attention mask.
    Handles left-padding by finding first and last non-padding positions.

    Args:
        sequence: Any list-like sequence (mask, tokens, etc.)
        attention_mask: Attention mask (1=real token, 0=padding)

    Returns:
        Trimmed sequence without padding
    """
    non_padded_indices = [i for i, a in enumerate(attention_mask) if a == 1]
    if not non_padded_indices:
        return []

    start_idx = non_padded_indices[0]
    end_idx = non_padded_indices[-1] + 1
    return sequence[start_idx:end_idx]


def create_mask_from_prefix_len(attention_mask: list[int], prefix_len: int) -> list[bool]:
    """Create a boolean mask where True indicates tokens after the prefix.

    Handles padding by counting real tokens.

    Args:
        attention_mask: Attention mask (1=real token, 0=padding)
        prefix_len: Number of real tokens in the prefix

    Returns:
        Boolean mask (True for tokens after prefix, False otherwise)
    """
    mask = []
    real_token_count = 0
    for attn in attention_mask:
        if attn == 1:
            real_token_count += 1
            mask.append(real_token_count > prefix_len)
        else:
            mask.append(False)
    return mask


def rebuild_mask_for_generation(
    prompt_attention_mask: list[int],
    completion_attention_mask: list[int],
    input_ids: list[int] | None = None,
    eos_token_id: int | None = None,
    tokenizer_type: str = "llama",
    apply_padding: bool = False,
    prompt_detection_mask: list[bool] | None = None,
) -> list[bool]:
    """Rebuild detection mask for generated tokens.

    Handles left-padding by counting real tokens rather than using positions.
    Returns trimmed mask (padding removed) to match activation extraction.

    Optionally applies Apollo-style padding to include header tokens (e.g., assistant header)
    which contain important "planning" activations.

    Args:
        prompt_attention_mask: Attention mask for prompt (1=real, 0=padding)
        completion_attention_mask: Attention mask for completion (1=real, 0=padding)
        input_ids: Optional full sequence token IDs (for EOS detection)
        eos_token_id: Optional EOS token ID to exclude from mask
        tokenizer_type: Type of tokenizer for padding config ("llama", "gemma", "mistral", "qwen")
        apply_padding: Whether to apply Apollo-style header token padding (default: False
            for backward compatibility; set True for Apollo probe evaluation)
        prompt_detection_mask: Optional mask for the prompt portion. If provided,
            True values from this mask will be carried over to the generated mask.

    Returns:
        Trimmed mask where:
        - False for prompt tokens unless they were True in prompt_detection_mask
        - True for generated tokens (excluding EOS, plus header padding if enabled)
        Length matches trimmed activations (padding removed)
    """
    prompt_real_tokens = sum(prompt_attention_mask)

    # Build mask for full sequence (including padding)
    mask = create_mask_from_prefix_len(completion_attention_mask, prompt_real_tokens)

    # If prompt detection mask is provided, carry over True values
    if prompt_detection_mask is not None:
        # Map prompt mask values to their real token index
        # We need to handle potential padding in prompt_detection_mask
        # Assuming prompt_detection_mask aligns with prompt_attention_mask
        tokens_to_mask = set()
        real_token_idx = 0
        for i, attn in enumerate(prompt_attention_mask):
            if attn == 1:
                # If this real token is marked in prompt mask, remember its index
                if i < len(prompt_detection_mask) and prompt_detection_mask[i]:
                    tokens_to_mask.add(real_token_idx)
                real_token_idx += 1

        # Apply to full mask
        real_token_idx = 0
        for i, attn in enumerate(completion_attention_mask):
            if attn == 1:
                if real_token_idx in tokens_to_mask:
                    mask[i] = True
                real_token_idx += 1

    # Exclude EOS tokens from the mask if we can detect them
    if input_ids is not None and eos_token_id is not None:
        for i, tok_id in enumerate(input_ids):
            if tok_id == eos_token_id:
                mask[i] = False

    # Apply Apollo-style padding to include header tokens
    if apply_padding:
        mask = apply_detection_mask_padding(mask, completion_attention_mask, tokenizer_type)

    # Trim to match activation extraction (remove padding)
    return trim_to_nonpadding(mask, completion_attention_mask)


def collate_fn_with_prefill(
    batch: list[dict],
    tokenizer: PreTrainedTokenizer,
    prefix_messages: int = -1,
    apply_padding: bool = False,
    for_generation: bool = False,
) -> dict:
    """Collate function that creates detection masks for prefill (assistant completion) scenarios.

    Masks the assistant's response content tokens for probe training.
    Optionally applies Apollo-style padding to include header tokens.
    Returns trimmed masks (padding removed) to match activation extraction.

    Args:
        batch: List of samples with 'conversation' key
        tokenizer: Tokenizer for encoding
        prefix_messages: Number of messages to treat as prefix (prompt).
                        -1 means all but last message (default for assistant response).
        apply_padding: Whether to apply Apollo-style header token padding (default: False
            for backward compatibility; set True for Apollo probe evaluation)
        for_generation: If True, strip EOT token for generation continuation.
                       If False (default), keep EOT token for prefill training.

    Returns:
        Dict with input_ids, attention_mask, and detection_mask
        Masks are trimmed to match activation lengths (padding removed)
    """
    # Pass for_generation to control EOT stripping
    out = default_collate_fn(batch, tokenizer, for_generation=for_generation)
    tokenizer_type = get_tokenizer_type(tokenizer)

    masks = []

    for i, sample in enumerate(batch):
        conversation = sample["conversation"]

        # Split conversation into prefix and completion
        if prefix_messages == -1:
            prefix_conversation = conversation[:-1]  # All but last (assistant) message
            completion_message = conversation[-1]
        else:
            prefix_conversation = conversation[:prefix_messages]
            completion_message = conversation[prefix_messages] if prefix_messages < len(conversation) else None

        # Get the full formatted text for char_to_token mapping
        full_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize the full text to get char_to_token mapping
        full_encoding = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        # Tokenize prefix to find where completion starts
        prefix_text = tokenizer.apply_chat_template(
            prefix_conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_len_chars = len(prefix_text)

        # Get content info - mask all assistant content tokens
        if completion_message and completion_message.get("content"):
            content = completion_message["content"]
            content_start_char = prefix_len_chars
            mask_end_char = content_start_char + len(content)
        else:
            content_start_char = prefix_len_chars
            mask_end_char = prefix_len_chars

        # Get attention mask for this sample (handles left-padding)
        attention_mask = out["attention_mask"][i].tolist()

        # Use offset mapping to create the mask
        # The offset mapping tells us which characters each token covers
        offset_mapping = full_encoding.get("offset_mapping", [])

        # Create mask based on character positions
        mask = []
        token_idx = 0  # Counter for real tokens in full_encoding
        for attn in attention_mask:
            if attn == 0:
                # Padding token
                mask.append(False)
            else:
                # Real token - get its character range from full_encoding
                if token_idx < len(offset_mapping):
                    start_char, end_char = offset_mapping[token_idx]
                    # Token is masked if it's within the content range we want to detect
                    # A token is "in content" if any part of it falls within the masked range
                    in_masked_content = start_char >= content_start_char and start_char < mask_end_char
                    mask.append(in_masked_content)
                    token_idx += 1
                else:
                    # Should not happen if lengths match, but handle safely
                    mask.append(False)

        # Apply Apollo-style padding to include header tokens
        if apply_padding:
            mask = apply_detection_mask_padding(mask, attention_mask, tokenizer_type)

        # Trim mask to match activation extraction (remove padding)
        trimmed_mask = trim_to_nonpadding(mask, attention_mask)
        masks.append(torch.tensor(trimmed_mask, dtype=torch.bool))

    out["detection_mask"] = masks
    return out
