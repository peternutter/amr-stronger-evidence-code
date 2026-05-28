import os
import shutil

# CRITICAL: Set CUDA memory allocation config BEFORE importing torch
# This MUST happen before any CUDA context is created
# Use both old and new variable names for compatibility
# expandable_segments helps with variable tensor sizes
# max_split_size_mb prevents large block splitting that causes fragmentation during generation
# garbage_collection_threshold triggers GC earlier when memory pressure builds
_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.6"
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _CUDA_ALLOC_CONF
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = _CUDA_ALLOC_CONF

import time
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.activation_metadata import ActivationMetadataStore
from src.data.activation_schema import ActivationSchemaError, validate_dataset
from src.models.local_model import LocalModel, clear_gpu_memory
from src.utils import RankedLogger, extras, rebuild_mask_for_generation, trim_to_nonpadding
from src.utils.mask_utils import get_tokenizer_type
from src.utils.streaming import StreamingActivationWriter

if TYPE_CHECKING:
    from lightning import LightningDataModule

    from src.models.remote_model import RemoteModel

log = RankedLogger(__name__, rank_zero_only=True)

LAYER_PASSTHROUGH_COLUMNS = {"label", "detection_mask"}


def parse_layers(cfg_layer, num_layers: int) -> list[int]:
    """Parse layer config to list of layer indices."""
    if cfg_layer == "all":
        return list(range(num_layers))
    elif hasattr(cfg_layer, "__iter__") and not isinstance(cfg_layer, str):
        layers = list(cfg_layer)
    else:
        layers = [cfg_layer]
    # Handle negative indices
    return [(layer + num_layers) % num_layers for layer in layers]


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="calculate_activations",
)
def main(cfg: DictConfig):
    """Script to calculate activations, responses and their labels.

    This version uses hook-based layer extraction to avoid OOM on large models.
    Instead of using output_hidden_states=True (which gathers all layers on GPU 0),
    we register forward hooks for only the requested layers and move activations
    to CPU immediately.

    Memory optimization: Activations are streamed to disk per-batch using async I/O
    to avoid accumulating large tensors in memory.

    DATA FLOW:
    ==========
    Both PREFILL and GENERATION modes produce the same output structure:

    1. Full sequence (prompt + completion) is processed
    2. Activations are extracted for the full trimmed sequence
    3. detection_mask marks which tokens are completion (True) vs prompt (False)
    4. completion_start_index indicates where completion begins

    SAVED DATA:
    ===========
    Metadata (output_dir/metadata/):
        - sample_index: Sequential ID
        - input_ids: Full trimmed token IDs (matches activation length)
        - attention_mask: Full trimmed attention mask
        - completion_start_index: Where completion starts
        - detection_mask: Boolean mask for prompt/completion
        - prompt_text, response_text, conversation, label, etc.

    Layer datasets (output_dir/layer_N/):
        - sample_index: Sequential ID
        - activations: Full trimmed sequence [seq_len, hidden_dim]
        - completion_start_index: Where completion starts
        - detection_mask: Boolean mask (for convenience)
        - label: Ground truth (if available)

    INVARIANTS:
    ===========
    For each sample i:
        len(metadata.input_ids[i]) == len(metadata.detection_mask[i])
        len(layer.activations[i]) == len(layer.detection_mask[i])
        len(metadata.input_ids[i]) == len(layer.activations[i])  # CRITICAL
    """
    torch.set_float32_matmul_precision("medium")

    # Initialize extras (logging, seeding, etc.)
    extras(cfg)

    # Log memory configuration for debugging OOM issues
    cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "not set")
    alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "not set")
    log.info(f"PYTORCH_CUDA_ALLOC_CONF={cuda_alloc_conf}")
    log.info(f"PYTORCH_ALLOC_CONF={alloc_conf}")

    # Verify CUDA allocator is configured correctly after torch is imported
    if torch.cuda.is_available():
        try:
            # Get allocator config to verify expandable_segments is working
            allocator_backend = torch.cuda.get_allocator_backend()
            log.info(f"CUDA allocator backend: {allocator_backend}")

            # Log initial memory state
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log.info(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.2f} GB")
        except Exception as e:
            log.warning(f"Could not get CUDA allocator info: {e}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LocalModel = hydra.utils.instantiate(cfg.model, generation_kwargs=cfg.get("generation_kwargs"))

    # If model has a tokenizer and datamodule needs one, set it
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
    if hasattr(datamodule, "tokenizer") and datamodule.tokenizer is None:
        datamodule.tokenizer = tokenizer

    # Log model memory usage
    model.log_model_memory_usage()

    # Log device map for multi-GPU verification
    if hasattr(model.model, "hf_device_map"):
        log.info(f"Model device map: {model.model.hf_device_map}")
    else:
        log.info(f"Model on single device: {model.device}")

    # Dead Salmon baseline: shuffle weights if seed is set in data config
    randomize_seed = cfg.data.get("randomize_weights_seed", None)
    if randomize_seed is not None:
        log.info("=" * 80)
        log.info(f"DEAD SALMON MODE ENABLED (seed={randomize_seed})")
        log.info("Shuffling transformer weights to create null hypothesis baseline")
        log.info("=" * 80)
        model.shuffle_transformer_weights(seed=randomize_seed, keep_embeddings=True)

    # Setup steering if enabled
    from src.utils.steering import setup_steering

    steering_hook_handle, steering_config = setup_steering(cfg, model)

    # Initialize judge if provided

    judge = None
    if hasattr(cfg, "judge") and cfg.judge is not None:
        judge: RemoteModel = hydra.utils.instantiate(cfg.judge)

    log.info(f"Datamodule has completions: {datamodule.has_completions}")
    log.info(f"Datamodule has labels: {datamodule.has_labels}")
    log.info(f"Datamodule has judge prompts: {datamodule.has_judge_prompts}")

    # Setup datamodule for prediction
    datamodule.prepare_data()
    datamodule.setup(stage="predict")

    # Determine which layers to extract BEFORE running forward passes
    # This is crucial for memory efficiency - we only capture what we need
    num_layers = model.num_layers
    log.info(f"Model has {num_layers} transformer layers")

    save_activations = getattr(datamodule, "save_activations", True)
    if save_activations:
        layers = parse_layers(cfg.layer, num_layers)
        log.info(f"Will extract activations for layers: {layers}")
    else:
        layers = []
        log.info("Skipping activation extraction (save_activations=False)")

    # Get dataloader
    dataloader = datamodule.predict_dataloader()

    # Setup output directory
    output_dir = Path(cfg.paths.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize streaming writer for activations
    streaming_writer = StreamingActivationWriter(output_dir, layers) if layers else None

    # =========================================================================
    # PHASE 1: EXTRACTION
    # Process batches, stream activations to disk, capture sequence info
    # =========================================================================
    log.info("Beginning activation extraction with streaming to disk...")

    # Store per-batch data needed for metadata (lightweight - just tokens, not activations)
    batched_full_sequences = []  # Full sequence input_ids/attention_mask (matches activation length)
    batched_detection_masks = []  # Detection masks for slicing prompt vs completion

    is_generation_mode = not datamodule.has_completions
    desc = "Generating + extracting" if is_generation_mode else "Extracting activations"

    # Check if Apollo-style mask padding is enabled
    # Priority: 1) Config override if explicitly set, 2) Datamodule's default
    # Datamodules like InstructedPairs, Alpaca, StressTest default to True (matching Apollo)
    # Datamodules like RolePlaying default to False (Apollo uses 0/0 padding for it)
    cfg_apollo_padding = cfg.get("apollo_mask_padding", None)
    datamodule_apollo_padding = getattr(datamodule, "use_apollo_padding", False)
    use_apollo_padding = cfg_apollo_padding if cfg_apollo_padding is not None else datamodule_apollo_padding
    tok_type = get_tokenizer_type(tokenizer) if tokenizer else "llama"
    if use_apollo_padding:
        source = "config" if cfg_apollo_padding is not None else "datamodule"
        log.info(f"Apollo mask padding enabled for tokenizer type: {tok_type} (from {source})")
        from src.utils.mask_utils import apply_detection_mask_padding
    else:
        log.info(f"Apollo mask padding disabled (from {'config' if cfg_apollo_padding is not None else 'datamodule'})")

    for _, batch in enumerate(tqdm(dataloader, desc=desc)):
        if datamodule.has_completions:
            # -----------------------------------------------------------------
            # PREFILL PATH: Data already has completions, just extract
            # Input: Full sequence (prompt + completion from dataset)
            # Output: Activations for full sequence
            # -----------------------------------------------------------------
            outputs = (
                model.extract_activations_for_layers(batch, layers)
                if layers
                else {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "layer_activations": {},
                }
            )
            # Detection mask comes from collate_fn (already trimmed to match sequence)
            batch_masks = [m.tolist() for m in batch["detection_mask"]]

            # Apply Apollo-style padding if enabled
            # Note: masks from collate_fn are already trimmed, so pass None for attention_mask
            if use_apollo_padding:
                batch_masks = [apply_detection_mask_padding(mask, None, tok_type) for mask in batch_masks]

        else:
            # -----------------------------------------------------------------
            # GENERATION PATH: Generate completions, then extract
            # Input: Prompt only
            # Output: Activations for full sequence (prompt + generated completion)
            # -----------------------------------------------------------------
            prompt_attention_mask = batch["attention_mask"].clone()

            # Generate and extract activations in one pass
            outputs = model.generate_with_activations_for_layers(batch, layers)

            # Build detection mask from prompt vs generated lengths
            # Pass input_ids and eos_token_id to exclude EOS from mask
            # Apollo-style padding is applied via rebuild_mask_for_generation if enabled
            eos_token_id = tokenizer.eos_token_id if tokenizer else None
            batch_masks = [
                rebuild_mask_for_generation(
                    prompt_attention_mask[i].tolist(),
                    outputs["attention_mask"][i].tolist(),
                    input_ids=outputs["input_ids"][i].tolist() if eos_token_id else None,
                    eos_token_id=eos_token_id,
                    tokenizer_type=tok_type,
                    apply_padding=use_apollo_padding,
                )
                for i in range(len(prompt_attention_mask))
            ]

        # ---------------------------------------------------------------------
        # COMMON: Save sequence info before deleting outputs
        # This is used later for metadata - must match activation lengths
        # ---------------------------------------------------------------------
        batched_full_sequences.append(
            {
                "input_ids": outputs["input_ids"].cpu(),
                "attention_mask": outputs["attention_mask"].cpu(),
            }
        )
        batched_detection_masks.append(batch_masks)

        # Stream activations to disk immediately (async write, non-blocking)
        if streaming_writer is not None:
            streaming_writer.write_batch(outputs, batch_masks)

        # Free activation memory immediately since it's been written to disk
        del outputs
        clear_gpu_memory(reset_stats=True)

    # Wait for all async writes to complete
    if streaming_writer is not None:
        log.info("Waiting for async disk writes to complete...")
        streaming_writer.wait_for_pending()

    clear_gpu_memory()
    log.info("Completed activation extraction")

    # =========================================================================
    # PHASE 2: LABELING & METADATA
    # Build metadata dict with aligned data
    # =========================================================================

    # Initialize metadata columns from dataset + extra columns we add
    metadata_dict = {key: [] for key in datamodule.dataset.column_names}

    # Check if datamodule has JSON fields (e.g., DeceptionBench with thought/response)
    has_json_fields = getattr(datamodule, "has_json_fields", False)

    extra_keys = [
        "sample_index",
        "input_ids",  # Full sequence token IDs (matching activations length)
        "attention_mask",  # Full sequence attention mask
        "completion_start_index",  # Where completion starts in the sequence
        "prompt_text",
        "response_text",
        "judge_prompt",
        "judge_response_raw",
        "judge_response_parsed",
        "detection_mask",
    ]

    # Add JSON field indices for DeceptionBench-style datasets
    if has_json_fields:
        extra_keys.extend(
            [
                "json_field_indices",  # Dict with thought/response token boundaries
            ]
        )

    for extra_key in extra_keys:
        metadata_dict[extra_key] = []

    sample_index = 0
    log.info("Beginning response labeling and metadata extraction...")

    for batch_idx, full_seq in enumerate(
        tqdm(batched_full_sequences, desc="Labeling", total=len(batched_full_sequences))
    ):
        batch_size = full_seq["input_ids"].shape[0]
        batch_masks = batched_detection_masks[batch_idx]

        for i in range(batch_size):
            sample = datamodule.dataset[sample_index]
            metadata_dict["sample_index"].append(sample_index)
            metadata_dict["detection_mask"].append(batch_masks[i])

            # Copy original dataset columns
            for key, value in sample.items():
                metadata_dict[key].append(value)

            # Build prompt text from conversation
            prompt_text = ""
            if tokenizer is not None and sample.get("conversation"):
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        sample["conversation"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as e:
                    log.warning(f"Failed to build prompt text for sample {sample_index}: {e}")
            metadata_dict["prompt_text"].append(prompt_text)

            # -----------------------------------------------------------------
            # CRITICAL: Use full sequence from outputs (not prompt-only)
            # This ensures metadata.input_ids aligns with layer.activations
            # -----------------------------------------------------------------
            raw_ids = full_seq["input_ids"][i].cpu().tolist()
            raw_attn = full_seq["attention_mask"][i].cpu().tolist()

            # Trim to valid range (remove padding) - matches activation extraction
            full_ids = trim_to_nonpadding(raw_ids, raw_attn)
            full_attn = trim_to_nonpadding(raw_attn, raw_attn)

            # Calculate completion_start_index from detection mask (count of False values)
            completion_start_idx = sum(1 for m in batch_masks[i] if not m)

            metadata_dict["input_ids"].append(full_ids)
            metadata_dict["attention_mask"].append(full_attn)
            metadata_dict["completion_start_index"].append(completion_start_idx)

            # Get response text
            if datamodule.has_completions:
                # Prefill: get from sample data
                response_str = sample.get("completion", "")
                if not response_str and sample.get("conversation"):
                    for msg in reversed(sample["conversation"]):
                        if msg.get("role") == "assistant":
                            response_str = msg.get("content", "")
                            break
            else:
                # Generation: decode generated tokens (completion portion)
                if tokenizer is not None and completion_start_idx < len(full_ids):
                    try:
                        completion_ids = full_ids[completion_start_idx:]
                        response_str = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    except Exception as e:
                        log.warning(f"Failed to decode response for sample {sample_index}: {e}")
                        response_str = ""
                else:
                    response_str = ""
            metadata_dict["response_text"].append(response_str)

            # Extract JSON field indices for DeceptionBench-style datasets
            # This enables flexible masking (thought-only, response-only, or both)
            if has_json_fields and tokenizer is not None:
                from src.utils.mask_utils import extract_json_field_indices

                json_indices = extract_json_field_indices(
                    response_text=response_str,
                    token_ids=full_ids,
                    tokenizer=tokenizer,
                    completion_start_idx=completion_start_idx,
                )
                metadata_dict["json_field_indices"].append(json_indices)

            # Judge evaluation (if no ground truth labels)
            judge_prompt, judge_response_raw, judge_response_parsed = "", "", ""

            if not datamodule.has_labels:
                # Use judge to evaluate response
                conversation, parse_fn = datamodule.get_judge_prompt(
                    sample_index=sample_index,
                    response=response_str,
                )
                judge_prompt = conversation
                for attempt in range(cfg.max_judge_retries):
                    try:
                        judge_result = judge.generate(conversation)
                        # Extract text from GenerationResult
                        judge_response_raw = judge_result.text if hasattr(judge_result, "text") else str(judge_result)
                        labels = parse_fn(judge_response_raw)
                        judge_response_parsed = labels
                        break
                    except Exception as e:
                        log.warning(
                            f"[Attempt {attempt + 1}/{cfg.max_judge_retries}] Error during OpenAI evaluation. {e}"
                        )
                        time.sleep(cfg.judge_retry_delay)
                else:
                    raise RuntimeError("Max retries exceeded for OpenAI evaluation.")

                # Add results to metadata
                for key, value in labels.items():
                    if key not in metadata_dict:
                        if sample_index > 0:
                            raise ValueError("All samples must have the same set of label keys.")
                        metadata_dict[key] = []
                    metadata_dict[key].append(value)
            metadata_dict["judge_prompt"].append(judge_prompt)
            metadata_dict["judge_response_raw"].append(judge_response_raw)
            metadata_dict["judge_response_parsed"].append(judge_response_parsed)

            sample_index += 1

    log.info("Response processing complete.")

    # =========================================================================
    # PHASE 3: VALIDATION & SAVING
    # Validate alignment, save metadata and layer datasets
    # =========================================================================

    log.info(f"Collected metadata for {len(metadata_dict['sample_index'])} samples")

    # Validate alignment before saving
    if len(metadata_dict["input_ids"]) > 0:
        sample_ids_len = len(metadata_dict["input_ids"][0])
        sample_mask_len = len(metadata_dict["detection_mask"][0])
        if sample_ids_len != sample_mask_len:
            log.error(
                f"ALIGNMENT ERROR: input_ids[0] length={sample_ids_len} "
                f"!= detection_mask[0] length={sample_mask_len}"
            )
            raise ValueError("Metadata alignment check failed!")
        log.info(f"✓ Metadata alignment check passed (sample 0: {sample_ids_len} tokens)")

    # Add steering config to metadata if steering was enabled
    if steering_config is not None:
        metadata_dict["steering_config"] = [steering_config] * len(metadata_dict["sample_index"])
        log.info("✓ Steering config added to metadata")

    metadata_store = ActivationMetadataStore(output_dir)
    metadata_store.save_dict(metadata_dict)

    # Cleanup steering hook if registered
    if steering_hook_handle is not None:
        steering_hook_handle.remove()
        log.info("Steering hook removed")

    # Check if we should save activations
    if not save_activations:
        log.info("Skipping activation saving as save_activations is False.")
        if streaming_writer is not None:
            streaming_writer.cleanup()
        return None

    log.info("Beginning activation consolidation...")
    log.info(f"Consolidating activations for layers: {layers}")

    # Clean up stale layer directories from previous runs
    existing_layer_dirs = list(output_dir.glob("layer_*"))
    layers_to_save_set = set(layers)
    for existing_dir in existing_layer_dirs:
        try:
            layer_num = int(existing_dir.name.replace("layer_", ""))
            if layer_num not in layers_to_save_set:
                log.warning(f"Removing stale layer directory: {existing_dir.name}")
                shutil.rmtree(existing_dir)
        except ValueError:
            # Not a valid layer directory name, skip
            pass

    # Build base metadata for layer datasets
    layer_base = {"sample_index": list(metadata_dict["sample_index"])}
    for column in LAYER_PASSTHROUGH_COLUMNS:
        if column in metadata_dict:
            layer_base[column] = list(metadata_dict[column])

    # Consolidate each layer from temp files to HuggingFace Dataset
    for layer in tqdm(layers, desc="Consolidating layers"):
        log.info(f"Consolidating layer {layer}")

        dataset = streaming_writer.consolidate_layer(layer, layer_base)

        # Validate schema before saving
        try:
            validate_dataset(dataset, check_all=False)
        except ActivationSchemaError as e:
            log.error(f"Schema validation failed for layer {layer}: {e}")
            raise

        # Cross-validate: layer activations length should match metadata input_ids length
        if len(metadata_dict["input_ids"]) > 0:
            layer_acts_len = len(dataset[0]["activations"])
            meta_ids_len = len(metadata_dict["input_ids"][0])
            if layer_acts_len != meta_ids_len:
                log.error(
                    f"CROSS-ALIGNMENT ERROR: layer_{layer} activations[0] length={layer_acts_len} "
                    f"!= metadata input_ids[0] length={meta_ids_len}"
                )
                raise ValueError(f"Layer/metadata alignment check failed for layer {layer}!")
            log.info(f"✓ Layer {layer} cross-alignment check passed ({layer_acts_len} tokens)")

        dataset.set_format(type="torch")
        dataset.save_to_disk(output_dir / f"layer_{layer}")
        log.info(f"✓ Layer {layer} saved with valid schema")

    # Cleanup temp files
    streaming_writer.cleanup()
    log.info("Streaming activation extraction complete.")

    return None


if __name__ == "__main__":
    main()
