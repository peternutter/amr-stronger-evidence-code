from __future__ import annotations

import gc
from collections.abc import Callable

import torch

from src.utils import load_model_and_tokenizer, pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def log_gpu_memory(prefix: str = "") -> None:
    """Log GPU memory usage for all available GPUs."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        log.info(f"{prefix}GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")


def log_gpu_memory_detailed(prefix: str = "", hypothesis_id: str = "") -> None:
    """Log detailed GPU memory for OOM debugging - includes peak and nvidia-smi style info."""
    if not torch.cuda.is_available():
        return
    import subprocess

    log.info(f"{'=' * 60}")
    log.info(f"{prefix} [Hypothesis: {hypothesis_id}]")

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free_in_reserved = reserved - allocated

        # Peak stats may not be available in all PyTorch builds
        try:
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(i) / 1024**3
            peak_str = f"peak_alloc={max_allocated:.2f}GB, peak_reserved={max_reserved:.2f}GB, "
        except (AttributeError, RuntimeError):
            peak_str = ""

        log.info(
            f"  GPU {i}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, "
            f"{peak_str}total={total:.2f}GB, free_in_reserved={free_in_reserved:.2f}GB"
        )

    # Try to get nvidia-smi data to see non-PyTorch memory usage
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            log.info(f"  nvidia-smi (MB): {result.stdout.strip().replace(chr(10), ' | ')}")
    except Exception as e:
        log.info(f"  nvidia-smi: unavailable ({e})")
    log.info(f"{'=' * 60}")


def clear_gpu_memory(reset_stats: bool = False) -> None:
    """Aggressively clear GPU memory - sync, collect, and empty cache.

    Args:
        reset_stats: If True, also reset peak memory stats for accurate tracking.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if reset_stats:
            torch.cuda.reset_peak_memory_stats()


class LocalModel:
    """HuggingFace CausalLM model with generation and memory-efficient activation extraction.

    Provides hook-based layer extraction to avoid OOM on large models by extracting
    only requested layers and moving activations to CPU immediately.
    """

    def __init__(
        self,
        model_name_or_path: str,
        safe_name: str,
        model_init_kwargs: dict | None = None,
        tokenizer_init_kwargs: dict | None = None,
        generation_kwargs: dict | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.safe_name = safe_name
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path,
            model_init_kwargs=model_init_kwargs,
            tokenizer_init_kwargs=tokenizer_init_kwargs,
        )
        self.generation_kwargs = generation_kwargs or {}

        # Log attention implementation being used
        attn_impl = getattr(self.model.config, "_attn_implementation", "unknown")
        log.info(f"Model loaded with attention implementation: {attn_impl}")

        # Log GPU state after model load for debugging
        self.log_gpu_processes()

    @property
    def device(self) -> torch.device:
        """Get the device of the model (first parameter's device for sharded models)."""
        return next(self.model.parameters()).device

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def num_layers(self) -> int:
        """Get the number of transformer layers in the model."""
        return len(self._get_transformer_layers())

    def _get_transformer_layers(self) -> list:
        """Get the list of transformer layers from the model.

        Supports common architectures: Llama, Qwen, GPT-2, etc.
        """
        model = self.model

        # Try common layer access patterns
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Llama, Qwen, Mistral style
            return list(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-2, GPT-Neo style
            return list(model.transformer.h)
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            # GPT-NeoX style
            return list(model.gpt_neox.layers)
        else:
            raise ValueError(
                f"Could not find transformer layers for model type {type(model).__name__}. "
                "Please add support for this architecture."
            )

    def _get_embedding_layer(self):
        """Get the embedding layer from the model."""
        model = self.model
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte
        return None

    def shuffle_transformer_weights(self, seed: int = 42, keep_embeddings: bool = True) -> None:
        """Shuffle transformer block weights (Dead Salmon baseline).

        This destroys learned structure while preserving weight statistics.
        Used as a null hypothesis for probing experiments to verify that
        probes are detecting learned representations, not input artifacts.

        Args:
            seed: Random seed for reproducibility.
            keep_embeddings: If True, keep embedding layer fixed (recommended
                for high-level tasks per the Dead Salmon paper Appendix A.1).
        """
        torch.manual_seed(seed)
        log.info(f"[DEAD SALMON] Shuffling transformer weights (seed={seed}, keep_embeddings={keep_embeddings})")

        layers = self._get_transformer_layers()
        params_shuffled = 0

        for _, layer in enumerate(layers):
            for _, param in layer.named_parameters():
                if param.requires_grad:
                    # Shuffle weights along flattened dimension
                    flat = param.data.view(-1)
                    perm = torch.randperm(flat.size(0), device=param.device)
                    param.data = flat[perm].view(param.shape)
                    params_shuffled += 1

        log.info(f"[DEAD SALMON] Shuffled {params_shuffled} parameters across {len(layers)} transformer layers")

    def log_gpu_processes(self) -> None:
        """Log all processes using GPU memory to detect multi-tenancy."""
        import subprocess

        log.info("=" * 60)
        log.info("[OOM Debug - Hypothesis A/B] Checking for other GPU processes...")
        try:
            # Get process list from nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                log.info(f"  GPU Processes: {result.stdout.strip().replace(chr(10), ' | ')}")
            else:
                log.info("  No GPU processes found or nvidia-smi unavailable")

            # Also get overall GPU memory usage
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free,memory.total,utilization.memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split(", ")
                    if len(parts) >= 4:
                        idx, used, free, total = parts[0], parts[1], parts[2], parts[3]
                        used_gb = float(used) / 1024
                        free_gb = float(free) / 1024
                        total_gb = float(total) / 1024
                        log.info(
                            f"  nvidia-smi GPU {idx}: used={used_gb:.2f}GB, free={free_gb:.2f}GB, "
                            f"total={total_gb:.2f}GB"
                        )
        except Exception as e:
            log.warning(f"  Failed to query nvidia-smi: {e}")
        log.info("=" * 60)

    def log_model_memory_usage(self) -> None:
        """Log the memory usage of the model parameters."""
        total_params = 0
        total_params_bytes = 0
        for p in self.model.parameters():
            total_params += p.numel()
            total_params_bytes += p.numel() * p.element_size()

        total_buffers_bytes = 0
        for b in self.model.buffers():
            total_buffers_bytes += b.numel() * b.element_size()

        total_gb = (total_params_bytes + total_buffers_bytes) / 1024**3
        log.info(f"[Model Memory] Total Params: {total_params / 1e9:.2f}B")
        log.info(f"[Model Memory] Size: {total_gb:.2f} GB (Params + Buffers)")

        # Log distribution across devices
        device_usage = {}
        for p in self.model.parameters():
            dev = str(p.device)
            device_usage[dev] = device_usage.get(dev, 0) + p.numel() * p.element_size()

        for dev, usage in device_usage.items():
            log.info(f"[Model Memory] Device {dev}: {usage / 1024**3:.2f} GB")

    @torch.no_grad()
    def generate(self, batch: dict, **kwargs) -> torch.Tensor:
        """Generate responses for conversations in batch.

        Args:
            batch: Input batch containing "input_ids" and "attention_mask".
            kwargs: Additional generation kwargs passed to `model.generate()`.

        Returns:
            Result of `model.generate()`
        """
        input_ids = batch["input_ids"]  # shape (batch_size, seq_length)
        attention_mask = batch["attention_mask"]  # shape (batch_size, seq_length)

        # Combine stored and passed kwargs, with passed kwargs taking precedence
        gen_kwargs = self.generation_kwargs.copy()
        gen_kwargs.update(kwargs)

        return self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    @torch.no_grad()
    def extract_activations(self, batch: dict) -> dict:
        """Extract activations for a batch (already tokenized by datamodule).

        This is used for non-generation mode (forward pass only).

        Args:
            batch: Input batch containing "input_ids" and "attention_mask".

        Returns:
            A dictionary with input_ids, attention_mask, and hidden_states.
        """
        # Move batch to model device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Move hidden states to CPU immediately to free GPU memory
        hidden_states = tuple(layer_hidden.detach().cpu() for layer_hidden in outputs.hidden_states)

        # Move input/output tensors to CPU
        result = {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "hidden_states": hidden_states,
        }

        # Explicitly delete GPU tensors and clear cache
        del outputs
        del input_ids
        del attention_mask
        clear_gpu_memory()
        log_gpu_memory("[extract_activations] After cleanup: ")

        return result

    @torch.no_grad()
    def extract_activations_for_layers(
        self,
        batch: dict,
        layers: list[int],
        include_embeddings: bool = False,
    ) -> dict[int, torch.Tensor]:
        """Extract activations for specific layers only using forward hooks.

        This is more memory-efficient than output_hidden_states=True because
        it only stores the requested layers and moves them to CPU immediately.

        Args:
            batch: Input batch containing "input_ids" and "attention_mask".
            layers: List of layer indices to extract (0-indexed).
            include_embeddings: If True, also extract embedding layer (index -1 in result).

        Returns:
            Dictionary mapping layer index to activation tensor on CPU.
            Shape of each tensor: (batch_size, seq_length, hidden_size)
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Log batch info and pre-forward memory state
        import time as _time

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        log.info(f"[extract_activations] Batch: size={batch_size}, seq_len={seq_len}, extracting {len(layers)} layers")
        log.info(f"[extract_activations] Input device: {input_ids.device}, Model device: {self.device}")
        log_gpu_memory_detailed("BEFORE forward pass", hypothesis_id="peak-memory")
        try:
            torch.cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            pass
        _forward_start = _time.perf_counter()

        activations: dict[int, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Get transformer layers
        transformer_layers = self._get_transformer_layers()
        num_layers = len(transformer_layers)

        # Normalize negative indices
        normalized_layers = [(layer + num_layers) % num_layers for layer in layers]

        def make_hook(layer_idx: int) -> Callable:
            def hook(module, input, output):
                # Handle different output formats (some return tuples)
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output
                # Transfer to CPU immediately to avoid holding all 80 layers on GPU
                # This reduces peak GPU memory from ~58GB to ~33GB (model weights only)
                activations[layer_idx] = hidden_state.detach().cpu()

            return hook

        # Register hooks for requested layers
        for layer_idx in normalized_layers:
            if layer_idx < 0 or layer_idx >= num_layers:
                log.warning(f"Layer {layer_idx} out of range [0, {num_layers}), skipping")
                continue
            hook = transformer_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        # Optionally capture embeddings
        if include_embeddings:
            embed_layer = self._get_embedding_layer()
            if embed_layer is not None:

                def embed_hook(module, input, output):
                    # Keep on GPU during forward pass to avoid sync overhead
                    activations[-1] = output.detach()

                hooks.append(embed_layer.register_forward_hook(embed_hook))

        try:
            # Forward pass without storing all hidden states
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,  # Don't store all layers
            )

            # Log peak memory after forward pass
            _forward_end = _time.perf_counter()
            log.info(f"[extract_activations] Forward pass took {_forward_end - _forward_start:.2f}s")
            log_gpu_memory_detailed("AFTER forward pass (PEAK)", hypothesis_id="peak-memory")

            # Explicitly delete outputs (contains logits on GPU) to prevent memory leak
            del outputs

            # Batch transfer all activations to CPU (avoids per-layer sync overhead)
            _transfer_start = _time.perf_counter()
            for layer_idx in activations:
                activations[layer_idx] = activations[layer_idx].cpu()
            _transfer_end = _time.perf_counter()
            elapsed = _transfer_end - _transfer_start
            log.info(f"[extract_activations] CPU transfer took {elapsed:.2f}s for {len(activations)} layers")
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

        # Store input metadata
        result = {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "layer_activations": activations,
        }

        # Clear GPU memory
        del input_ids
        del attention_mask
        clear_gpu_memory()

        # Log memory after cleanup
        _total_end = _time.perf_counter()
        log.info(f"[extract_activations] Total batch time: {_total_end - _forward_start:.2f}s")
        log_gpu_memory_detailed("AFTER cleanup", hypothesis_id="peak-memory")

        return result

    @torch.no_grad()
    def generate_with_activations_for_layers(self, batch: dict, layers: list[int], **kwargs) -> dict:
        """Generate responses and extract activations for specific layers using hooks.

        This is the memory-efficient version that avoids output_hidden_states=True,
        which would gather all layer activations on GPU 0 causing OOM.

        Args:
            batch: Input batch containing "input_ids" and "attention_mask".
            layers: List of layer indices to extract.
            kwargs: Additional generation kwargs passed to `model.generate()`.

        Returns:
            A dictionary with input_ids, attention_mask, and layer_activations dict.
        """
        # Move batch to model device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        gpu_batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        input_seq_len = input_ids.shape[-1]
        log.info(f"[generate_with_activations_for_layers] Input sequence length: {input_seq_len}")

        # Generate responses
        generated_ids = self.generate(gpu_batch, **kwargs)

        # Log generated sequence stats
        gen_seq_len = generated_ids.shape[-1]
        new_tokens = gen_seq_len - input_seq_len
        log.info(
            f"[generate_with_activations_for_layers] Generated: {gen_seq_len} tokens "
            f"(input={input_seq_len}, new={new_tokens})"
        )

        # Clear memory after generation
        del gpu_batch
        del input_ids
        del attention_mask
        clear_gpu_memory(reset_stats=True)

        # If no layers requested, skip the second forward pass entirely
        # This is a major memory optimization for generation-only mode
        if not layers:
            log.info("[generate_with_activations_for_layers] No layers requested, skipping second forward pass")
            result_ids = generated_ids.cpu()
            result_attention = (result_ids != self.tokenizer.pad_token_id).int()
            del generated_ids
            clear_gpu_memory(reset_stats=True)
            return {
                "input_ids": result_ids,
                "attention_mask": result_attention,
                "layer_activations": {},
            }

        log_gpu_memory("[generate_with_activations_for_layers] After generation: ")

        # Build batch for extraction with generated attention mask
        # Note: Using (generated_ids != pad_token_id) handles batch right-padding correctly
        # The prompt formatting fix (stripping EOT token) ensures proper generation
        result_attention = (generated_ids != self.tokenizer.pad_token_id).int()

        extraction_batch = {
            "input_ids": generated_ids,
            "attention_mask": result_attention,
        }

        # Reuse extract_activations_for_layers for the forward pass with hooks
        extraction_result = self.extract_activations_for_layers(extraction_batch, layers)

        log_gpu_memory("[generate_with_activations_for_layers] After cleanup: ")

        return extraction_result

    def register_steering_hook(
        self,
        layer: int,
        direction: torch.Tensor,
        strength: float,
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a steering hook on a specific transformer layer.

        The hook modifies activations during forward passes:
        - strength > 0: Add direction (steer towards probe's positive class)
        - strength < 0: Subtract direction (steer away from positive class)
        - strength == 0: Project out direction (remove component entirely)

        Args:
            layer: Layer index to apply steering (0-indexed)
            direction: Unit direction vector [1, hidden_dim]
            strength: Steering coefficient

        Returns:
            RemovableHandle to remove the hook when done
        """
        from src.utils.steering import create_steering_hook

        # Get transformer layers
        transformer_layers = self._get_transformer_layers()
        num_layers = len(transformer_layers)

        # Normalize negative indices
        normalized_layer = (layer + num_layers) % num_layers

        if normalized_layer < 0 or normalized_layer >= num_layers:
            raise ValueError(f"Layer {layer} out of range [0, {num_layers})")

        # Move direction to model device
        direction = direction.to(self.device)

        # Create the hook
        hook_fn, operation = create_steering_hook(direction, strength, normalized_layer)

        # Register hook
        handle = transformer_layers[normalized_layer].register_forward_hook(hook_fn)

        log.info(
            f"Registered steering hook on layer {normalized_layer}: " f"operation={operation}, strength={strength}"
        )

        return handle
