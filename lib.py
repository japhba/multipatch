import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# --- Layer/Token Specification ---

def get_layer_range(n_layers: int, frac_patch_layers: float) -> tuple[int, int, int]:
    """
    Get layer range anchored at the TOP (highest layers).

    Small fractions patch only the uppermost layers; larger fractions
    extend downward toward layer 0.

    Args:
        n_layers: Total layers in model
        frac_patch_layers: Fraction of layers to patch (0.0 to 1.0)
            e.g., 0.1 → only top 10% of layers
            e.g., 0.5 → top 50% of layers
            e.g., 1.0 → all layers

    Returns:
        (start_layer, end_layer, center_layer) as indices
    """
    n_patch = max(1, int(round(frac_patch_layers * n_layers)))
    end_layer = n_layers - 1  # Always the top layer
    start_layer = n_layers - n_patch  # Extends downward
    center_layer = (start_layer + end_layer) // 2

    return start_layer, end_layer, center_layer


def get_token_indices(n_tokens: int, frac_patch_tokens: float) -> list[int]:
    """
    Get token indices for the last `frac_patch_tokens` fraction of tokens.

    Args:
        n_tokens: Total number of tokens
        frac_patch_tokens: Fraction of tokens to use (0.0 to 1.0)
            e.g., 0.5 → last 50% of tokens

    Returns:
        List of token indices
    """
    n_keep = max(1, int(round(frac_patch_tokens * n_tokens)))
    return list(range(n_tokens - n_keep, n_tokens))


# Deprecated - use get_layer_range instead
def get_layer_indices(n_layers: int, f_start: float, f_end: float, n_extract: int) -> np.ndarray:
    """
    [DEPRECATED] Get unique layer indices from fractional specification.
    Use get_layer_range(n_layers, frac_patch_layers) instead.
    """
    fractions = np.linspace(f_start, f_end, n_extract)
    indices = np.round(fractions * (n_layers - 1)).astype(int)
    return np.unique(indices)


# --- Model Loading ---

def load_gemma(device="cuda", use_flash_attn: bool = True, compile_model: bool = True):
    """Load google/gemma-3-1b-it with Flash Attention 2 and torch.compile if available."""
    model_id = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Try Flash Attention 2, fall back to SDPA, then default
    attn_impl = None
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print(f"[MODEL] Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"  # PyTorch 2.0+ scaled dot product attention
            print(f"[MODEL] Flash Attention not installed, using SDPA")

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device,
    }
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Compile model for faster inference (PyTorch 2.0+)
    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[MODEL] Compiled with torch.compile (reduce-overhead mode)")
        except Exception as e:
            print(f"[MODEL] torch.compile failed: {e}, using eager mode")

    return model, tokenizer


# --- Activation Extraction ---

def get_layers(model):
    """Get the layers list from a model, handling different architectures."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'layers'):
        return model.layers
    else:
        raise ValueError(f"Cannot find layers in model: {type(model)}")


def get_activations(model, tokenizer, text: str, layer_indices: list) -> dict:
    """
    Extract activations using output_hidden_states=True.

    Args:
        model: The model
        tokenizer: The tokenizer
        text: Input text
        layer_indices: Which layers to return (0-indexed, after that layer)

    Returns:
        Dict mapping layer_idx -> tensor of shape (seq_len, hidden_dim)
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is tuple of (n_layers + 1) tensors: embeddings + each layer output
    # hidden_states[0] = embeddings, hidden_states[i] = output of layer i-1
    hidden_states = outputs.hidden_states

    result = {}
    for idx in layer_indices:
        # hidden_states[idx + 1] is the output after layer idx
        # Clone to avoid CUDA graph issues when sharing across threads
        result[idx] = hidden_states[idx + 1][0].detach().clone()  # Remove batch dim
    return result


def get_activations_batched(
    model,
    tokenizer,
    texts: list[str],
    layer_indices: list,
    batch_size: int = 8,
) -> list[dict]:
    """
    Extract activations for multiple texts using batched forward passes.

    Args:
        model: The model
        tokenizer: The tokenizer
        texts: List of input texts
        layer_indices: Which layers to return (0-indexed, after that layer)
        batch_size: Number of texts to process in each batch

    Returns:
        List of dicts, each mapping layer_idx -> tensor of shape (seq_len, hidden_dim)
        Note: Each text may have different seq_len
    """
    all_results = []

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        # Tokenize with padding
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        attention_mask = inputs.attention_mask

        # Extract per-example activations (removing padding)
        for i in range(len(batch_texts)):
            # Find actual sequence length (non-padded)
            seq_len = attention_mask[i].sum().item()

            result = {}
            for idx in layer_indices:
                # Get activations for this example, excluding padding
                # Clone to avoid CUDA graph issues when sharing across threads
                acts = hidden_states[idx + 1][i, :seq_len].detach().clone()
                result[idx] = acts
            all_results.append(result)

    return all_results


@dataclass
class CachedActivations:
    """Pre-computed activations for a set of source texts."""
    texts: list[str]
    activations: list[dict]  # List of {layer_idx: tensor}
    layer_indices: list[int]
    _device_copies: dict = field(default_factory=dict)  # Cache of {device: list[dict]}

    def __len__(self):
        return len(self.texts)

    def get(self, text: str, device: str = None) -> Optional[dict]:
        """Get cached activations for a text, optionally on a specific device."""
        try:
            idx = self.texts.index(text)
            if device is None:
                return self.activations[idx]
            # Use device-local copy if available
            if device in self._device_copies:
                return self._device_copies[device][idx]
            return self.activations[idx]
        except ValueError:
            return None

    def to_device(self, device: str) -> "CachedActivations":
        """
        Get a view with activations copied to the specified device.
        Caches the copy for reuse.
        """
        if device in self._device_copies:
            return self  # Already cached

        # Copy all activations to the target device
        device_activations = []
        for acts_dict in self.activations:
            device_acts = {
                layer: tensor.to(device)
                for layer, tensor in acts_dict.items()
            }
            device_activations.append(device_acts)

        self._device_copies[device] = device_activations
        return self


def precompute_activations(
    model,
    tokenizer,
    texts: list[str],
    frac_patch_layers: float = 1.0,
    batch_size: int = 8,
) -> CachedActivations:
    """
    Pre-compute activations for all texts at all layers that might be needed.

    Call this once before your grid sweep, then use patch_and_query_cached.

    Args:
        model: The model
        tokenizer: The tokenizer
        texts: All source texts to cache
        frac_patch_layers: Max fraction of layers you'll use (1.0 = all layers)
        batch_size: Batch size for extraction

    Returns:
        CachedActivations object to pass to patch_and_query_cached
    """
    n_layers = model.config.num_hidden_layers

    # Get the full layer range we might need
    start_layer, end_layer, _ = get_layer_range(n_layers, frac_patch_layers)
    layer_indices = list(range(start_layer, end_layer + 1))

    print(f"[CACHE] Extracting activations for {len(texts)} texts, layers {start_layer}-{end_layer}...")
    activations = get_activations_batched(model, tokenizer, texts, layer_indices, batch_size)
    print(f"[CACHE] Done. Cached {len(activations)} activation sets.")

    return CachedActivations(
        texts=texts,
        activations=activations,
        layer_indices=layer_indices,
    )


# --- Activation Injection ---

def inject_and_generate_multilayer(
    model,
    tokenizer,
    input_ids: torch.Tensor,  # (1, seq_len) - already includes prepended dummy tokens
    attention_mask: torch.Tensor,
    activations_by_layer: dict,  # {layer_idx: tensor of shape (n_tokens, hidden_dim)}
    patch_positions: list,  # Which positions to patch (e.g., [1, 2, 3, ...] after BOS)
    max_new_tokens: int = 10,  # Reduced default - "Yes"/"No" needs few tokens
    alpha: float = 0.5,
    debug: bool = False,  # Changed default to False for speed
) -> str:
    """
    Run model with convex-combination activation injection at multiple layers.

    h' = (1 - alpha) * h + alpha * v

    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: Token IDs with prepended dummy tokens
        attention_mask: Attention mask
        activations_by_layer: Dict mapping layer_idx -> activations tensor
        patch_positions: Which token positions to patch
        max_new_tokens: Max tokens to generate
        alpha: Interpolation weight (0 = keep original, 1 = full replacement)
        debug: Print injection info

    Returns:
        Generated text response
    """
    input_len = input_ids.shape[1]

    def make_hook(layer_acts, layer_idx):
        def injection_hook(module, input, output):
            hidden = output[0]  # (batch, seq, hidden)
            seq_len = hidden.shape[1]
            # Only inject during prefill (exact match)
            if seq_len != input_len:
                return output

            if debug:
                print(f"[INJECT] Layer {layer_idx}: seq_len={seq_len}, {len(patch_positions)} positions")

            for i, pos in enumerate(patch_positions):
                if i < layer_acts.shape[0] and pos < seq_len:
                    h = hidden[0, pos]
                    v = layer_acts[i].to(h.device, h.dtype)
                    if debug and i == 0:
                        h_norm_before = h.norm().item()
                    hidden[0, pos] = (1 - alpha) * h + alpha * v
                    if debug and i == 0:
                        h_norm_after = hidden[0, pos].norm().item()
                        print(f"  pos={pos}: norm {h_norm_before:.1f} → {h_norm_after:.1f}")
            return (hidden,) + output[1:]
        return injection_hook

    layers = get_layers(model)
    handles = []
    for layer_idx, acts in activations_by_layer.items():
        handle = layers[layer_idx].register_forward_hook(make_hook(acts, layer_idx))
        handles.append(handle)

    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    finally:
        for h in handles:
            h.remove()

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- Convenience ---

def patch_and_query(
    model,
    tokenizer,
    source_text: str,
    question: str,
    frac_patch_layers: float = 0.5,
    frac_patch_tokens: float = 1.0,
    alpha: float = 0.5,
    debug: bool = True,
) -> str:
    """
    End-to-end: extract activations from source → inject into oracle → generate.

    Patches MULTIPLE layers simultaneously (centered at 50% depth).
    Prepends dummy tokens to the question, then injects source activations at those positions.

    Args:
        model: The language model (used as both target and oracle)
        tokenizer: The tokenizer
        source_text: Text to extract activations from
        question: Question to ask about the activations
        frac_patch_layers: Fraction of layers to patch (0.0 to 1.0)
            e.g., 0.5 → patch layers from 25% to 75% depth
        frac_patch_tokens: Fraction of tokens to use from the end (0.0 to 1.0)
            e.g., 0.5 → last 50% of tokens
        alpha: Convex combination weight (0.5 = 50% blend)
        debug: Print injection info

    Returns:
        Generated response
    """
    n_layers = model.config.num_hidden_layers

    # Get layer range to patch (centered at 50%)
    start_layer, end_layer, _ = get_layer_range(n_layers, frac_patch_layers)
    patch_layers = list(range(start_layer, end_layer + 1))

    # Extract activations from all layers we'll patch
    activations = get_activations(model, tokenizer, source_text, patch_layers)

    # Select tokens based on frac_patch_tokens
    sample_acts = activations[patch_layers[0]]
    n_tokens = sample_acts.shape[0]
    token_indices = get_token_indices(n_tokens, frac_patch_tokens)
    n_patch = len(token_indices)

    # Filter activations to selected tokens
    activations_filtered = {
        layer: acts[token_indices] for layer, acts in activations.items()
    }

    # Tokenize the question
    question_tokens = tokenizer(question, return_tensors="pt", add_special_tokens=False)
    question_ids = question_tokens.input_ids[0]  # (seq_len,)

    # Get a dummy token ID (use pad token, or unk, or a common token)
    if tokenizer.pad_token_id is not None:
        dummy_id = tokenizer.pad_token_id
    elif tokenizer.unk_token_id is not None:
        dummy_id = tokenizer.unk_token_id
    else:
        dummy_id = 0  # Fallback

    # Build input: [BOS] + [dummy]*n_patch + [question tokens]
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    dummy_ids = torch.full((n_patch,), dummy_id, dtype=torch.long)

    if bos_id is not None:
        input_ids = torch.cat([
            torch.tensor([bos_id]),
            dummy_ids,
            question_ids
        ]).unsqueeze(0).to(model.device)
        # Patch positions are right after BOS: [1, 2, ..., n_patch]
        patch_positions = list(range(1, n_patch + 1))
    else:
        input_ids = torch.cat([dummy_ids, question_ids]).unsqueeze(0).to(model.device)
        # Patch positions are at the start: [0, 1, ..., n_patch-1]
        patch_positions = list(range(n_patch))

    attention_mask = torch.ones_like(input_ids)

    if debug:
        print(f"[SETUP] n_patch={n_patch}, patch_layers={patch_layers[0]}-{patch_layers[-1]}")
        print(f"  Input shape: {input_ids.shape}, patch_positions: {patch_positions[:5]}...")

    return inject_and_generate_multilayer(
        model, tokenizer, input_ids, attention_mask,
        activations_filtered, patch_positions, alpha=alpha, debug=debug
    )


def patch_and_query_cached(
    model,
    tokenizer,
    cached: CachedActivations,
    source_text: str,
    question: str,
    frac_patch_layers: float = 0.5,
    frac_patch_tokens: float = 1.0,
    alpha: float = 0.5,
    debug: bool = False,
) -> str:
    """
    Like patch_and_query, but uses pre-computed activations from cache.

    This is much faster when sweeping over hyperparameters with the same source texts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        cached: CachedActivations from precompute_activations()
        source_text: Text to get activations for (must be in cache)
        question: Question to ask about the activations
        frac_patch_layers: Fraction of layers to patch (0.0 to 1.0)
        frac_patch_tokens: Fraction of tokens to use from the end (0.0 to 1.0)
        alpha: Convex combination weight (0.5 = 50% blend)
        debug: Print injection info

    Returns:
        Generated response
    """
    # Get cached activations for this text (on model's device if available)
    model_device = str(next(model.parameters()).device)
    all_activations = cached.get(source_text, device=model_device)
    if all_activations is None:
        raise ValueError(f"Text not found in cache: {source_text[:50]}...")

    n_layers = model.config.num_hidden_layers

    # Get layer range to patch
    start_layer, end_layer, _ = get_layer_range(n_layers, frac_patch_layers)
    patch_layers = list(range(start_layer, end_layer + 1))

    # Filter to only the layers we need
    activations = {layer: all_activations[layer] for layer in patch_layers if layer in all_activations}

    # Select tokens based on frac_patch_tokens
    sample_acts = activations[patch_layers[0]]
    n_tokens = sample_acts.shape[0]
    token_indices = get_token_indices(n_tokens, frac_patch_tokens)
    n_patch = len(token_indices)

    # Filter activations to selected tokens
    activations_filtered = {
        layer: acts[token_indices] for layer, acts in activations.items()
    }

    # Tokenize the question
    question_tokens = tokenizer(question, return_tensors="pt", add_special_tokens=False)
    question_ids = question_tokens.input_ids[0]

    # Get a dummy token ID
    if tokenizer.pad_token_id is not None:
        dummy_id = tokenizer.pad_token_id
    elif tokenizer.unk_token_id is not None:
        dummy_id = tokenizer.unk_token_id
    else:
        dummy_id = 0

    # Build input: [BOS] + [dummy]*n_patch + [question tokens]
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    dummy_ids = torch.full((n_patch,), dummy_id, dtype=torch.long)

    if bos_id is not None:
        input_ids = torch.cat([
            torch.tensor([bos_id]),
            dummy_ids,
            question_ids
        ]).unsqueeze(0).to(model.device)
        patch_positions = list(range(1, n_patch + 1))
    else:
        input_ids = torch.cat([dummy_ids, question_ids]).unsqueeze(0).to(model.device)
        patch_positions = list(range(n_patch))

    attention_mask = torch.ones_like(input_ids)

    if debug:
        print(f"[SETUP] n_patch={n_patch}, patch_layers={patch_layers[0]}-{patch_layers[-1]}")
        print(f"  Input shape: {input_ids.shape}, patch_positions: {patch_positions[:5]}...")

    return inject_and_generate_multilayer(
        model, tokenizer, input_ids, attention_mask,
        activations_filtered, patch_positions, alpha=alpha, debug=debug
    )


def evaluate_batched(
    model,
    tokenizer,
    cached_pos: CachedActivations,
    cached_neg: CachedActivations,
    question: str,
    parse_fn,
    frac_layers: float,
    frac_tokens: float,
    alpha: float,
    n_examples: Optional[int] = None,
) -> float:
    """
    Evaluate accuracy using cached activations.

    Args:
        model: The model
        tokenizer: The tokenizer
        cached_pos: Cached activations for positive examples
        cached_neg: Cached activations for negative examples
        question: The question to ask
        parse_fn: Function to parse yes/no from response
        frac_layers: Fraction of layers to patch
        frac_tokens: Fraction of tokens to patch
        alpha: Blending weight
        n_examples: Max examples per class (None = all)

    Returns:
        Accuracy as a float
    """
    pos_texts = cached_pos.texts[:n_examples] if n_examples else cached_pos.texts
    neg_texts = cached_neg.texts[:n_examples] if n_examples else cached_neg.texts

    correct = 0
    total = len(pos_texts) + len(neg_texts)

    for text in pos_texts:
        resp = patch_and_query_cached(
            model, tokenizer, cached_pos, text, question,
            frac_patch_layers=frac_layers,
            frac_patch_tokens=frac_tokens,
            alpha=alpha, debug=False
        )
        if parse_fn(resp) == True:
            correct += 1

    for text in neg_texts:
        resp = patch_and_query_cached(
            model, tokenizer, cached_neg, text, question,
            frac_patch_layers=frac_layers,
            frac_patch_tokens=frac_tokens,
            alpha=alpha, debug=False
        )
        if parse_fn(resp) == False:
            correct += 1

    return correct / total


# --- Batched Generation (single GPU, multiple examples) ---

def patch_and_query_batched(
    model,
    tokenizer,
    cached: CachedActivations,
    source_texts: list[str],
    question: str,
    frac_patch_layers: float = 0.5,
    frac_patch_tokens: float = 1.0,
    alpha: float = 0.5,
    max_new_tokens: int = 10,
    debug: bool = False,
) -> list[str]:
    """
    Process multiple examples in a single batched generate() call.

    This is much faster than sequential calls because:
    - Single kernel launch overhead
    - Better GPU utilization
    - Parallelized attention computation

    Args:
        model: The language model
        tokenizer: The tokenizer
        cached: CachedActivations from precompute_activations()
        source_texts: List of texts to process (must all be in cache)
        question: Question to ask (same for all)
        frac_patch_layers, frac_patch_tokens, alpha: Hyperparameters
        max_new_tokens: Max tokens to generate (keep small for yes/no tasks)

    Returns:
        List of generated responses
    """
    batch_size = len(source_texts)
    if batch_size == 0:
        return []

    n_layers = model.config.num_hidden_layers
    start_layer, end_layer, _ = get_layer_range(n_layers, frac_patch_layers)
    patch_layers = list(range(start_layer, end_layer + 1))

    # Get dummy/special token IDs
    if tokenizer.pad_token_id is not None:
        dummy_id = tokenizer.pad_token_id
        pad_id = tokenizer.pad_token_id
    elif tokenizer.unk_token_id is not None:
        dummy_id = tokenizer.unk_token_id
        pad_id = tokenizer.eos_token_id or 0
    else:
        dummy_id = 0
        pad_id = 0

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id

    # Tokenize question once
    question_tokens = tokenizer(question, return_tensors="pt", add_special_tokens=False)
    question_ids = question_tokens.input_ids[0]
    question_len = len(question_ids)

    # Collect activations and build per-example data
    batch_activations = []  # List of {layer: tensor} per example
    n_patches = []  # Number of tokens to patch per example

    # Determine the device from the model
    model_device = str(next(model.parameters()).device)

    for text in source_texts:
        all_acts = cached.get(text, device=model_device)
        if all_acts is None:
            raise ValueError(f"Text not found in cache: {text[:50]}...")

        # Filter to needed layers
        acts = {layer: all_acts[layer] for layer in patch_layers if layer in all_acts}

        # Select tokens based on frac_patch_tokens
        sample_acts = acts[patch_layers[0]]
        n_tokens = sample_acts.shape[0]
        token_indices = get_token_indices(n_tokens, frac_patch_tokens)
        n_patch = len(token_indices)

        # Filter to selected tokens
        acts_filtered = {layer: a[token_indices] for layer, a in acts.items()}

        batch_activations.append(acts_filtered)
        n_patches.append(n_patch)

    # Pad to max n_patch
    max_n_patch = max(n_patches)

    # Build padded input_ids: [BOS] + [dummy]*max_n_patch + [question]
    # Shape: (batch_size, 1 + max_n_patch + question_len)
    seq_len = (1 if bos_id is not None else 0) + max_n_patch + question_len
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=model.device)

    # Patch positions per example (relative to sequence start)
    patch_positions_list = []

    for i in range(batch_size):
        n_patch = n_patches[i]
        # Pad activations to max_n_patch
        for layer in patch_layers:
            acts = batch_activations[i][layer]
            if acts.shape[0] < max_n_patch:
                padding = torch.zeros(
                    (max_n_patch - acts.shape[0], acts.shape[1]),
                    dtype=acts.dtype, device=acts.device
                )
                batch_activations[i][layer] = torch.cat([acts, padding], dim=0)

        # Build input for this example
        pos = 0
        if bos_id is not None:
            input_ids[i, pos] = bos_id
            attention_mask[i, pos] = 1
            pos += 1

        # Dummy tokens (only attend to real ones, not padding)
        for j in range(max_n_patch):
            input_ids[i, pos + j] = dummy_id
            if j < n_patch:
                attention_mask[i, pos + j] = 1

        # Patch positions: where to inject activations (only real ones, not padding)
        if bos_id is not None:
            positions = list(range(1, 1 + n_patch))
        else:
            positions = list(range(n_patch))
        patch_positions_list.append(positions)

        pos += max_n_patch

        # Question tokens
        for j, token_id in enumerate(question_ids):
            input_ids[i, pos + j] = token_id
            attention_mask[i, pos + j] = 1

    input_len = seq_len

    # Create batched hook
    def make_batched_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0]  # (batch, seq, hidden)
            current_seq_len = hidden.shape[1]

            # Only inject during prefill
            if current_seq_len != input_len:
                return output

            for batch_idx in range(batch_size):
                layer_acts = batch_activations[batch_idx][layer_idx]
                positions = patch_positions_list[batch_idx]

                for i, pos in enumerate(positions):
                    if i < layer_acts.shape[0] and pos < current_seq_len:
                        h = hidden[batch_idx, pos]
                        v = layer_acts[i].to(h.device, h.dtype)
                        hidden[batch_idx, pos] = (1 - alpha) * h + alpha * v

            return (hidden,) + output[1:]
        return hook

    # Register hooks
    layers = get_layers(model)
    handles = []
    for layer_idx in patch_layers:
        handle = layers[layer_idx].register_forward_hook(make_batched_hook(layer_idx))
        handles.append(handle)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
    finally:
        for h in handles:
            h.remove()

    # Decode each output
    responses = []
    for i in range(batch_size):
        response = tokenizer.decode(outputs[i], skip_special_tokens=True)
        responses.append(response)

    return responses


def evaluate_batched_fast(
    model,
    tokenizer,
    cached_pos: CachedActivations,
    cached_neg: CachedActivations,
    question: str,
    parse_fn: Callable,
    frac_layers: float,
    frac_tokens: float,
    alpha: float,
    n_examples: Optional[int] = None,
    batch_size: int = 8,
) -> float:
    """
    Evaluate using batched generation - much faster on single GPU.

    Args:
        batch_size: Number of examples to process at once (tune based on GPU memory)
    """
    pos_texts = cached_pos.texts[:n_examples] if n_examples else cached_pos.texts
    neg_texts = cached_neg.texts[:n_examples] if n_examples else cached_neg.texts

    correct = 0

    # Process positive examples in batches
    for i in range(0, len(pos_texts), batch_size):
        batch = pos_texts[i:i + batch_size]
        responses = patch_and_query_batched(
            model, tokenizer, cached_pos, batch, question,
            frac_patch_layers=frac_layers,
            frac_patch_tokens=frac_tokens,
            alpha=alpha,
        )
        for resp in responses:
            if parse_fn(resp) == True:
                correct += 1

    # Process negative examples in batches
    for i in range(0, len(neg_texts), batch_size):
        batch = neg_texts[i:i + batch_size]
        responses = patch_and_query_batched(
            model, tokenizer, cached_neg, batch, question,
            frac_patch_layers=frac_layers,
            frac_patch_tokens=frac_tokens,
            alpha=alpha,
        )
        for resp in responses:
            if parse_fn(resp) == False:
                correct += 1

    total = len(pos_texts) + len(neg_texts)
    return correct / total


# --- Multi-GPU Parallelism ---

class MultiGPUWorkerPool:
    """
    Pool of models, one per GPU, for parallel inference.

    Usage:
        pool = MultiGPUWorkerPool(n_gpus=4)
        results = pool.map(texts, lambda model, tok, text: patch_and_query(...))
    """

    def __init__(self, n_gpus: Optional[int] = None, model_loader: Optional[Callable] = None):
        """
        Args:
            n_gpus: Number of GPUs to use (default: all available)
            model_loader: Function to load model, signature: (device) -> (model, tokenizer)
                         Default: load_gemma (with compile disabled for thread safety)
        """
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()

        if n_gpus == 0:
            raise RuntimeError("No GPUs available")

        self.n_gpus = n_gpus
        # Disable torch.compile by default - CUDA graphs don't work well with
        # multi-threaded access to shared tensors (cached activations)
        if model_loader is None:
            self.model_loader = lambda device: load_gemma(device=device, compile_model=False)
        else:
            self.model_loader = model_loader

        # Each GPU gets its own model, tokenizer, and lock
        self.models = []
        self.tokenizers = []
        self.locks = []

        print(f"[POOL] Loading models on {n_gpus} GPUs (compile disabled for thread safety)...")
        for i in range(n_gpus):
            device = f"cuda:{i}"
            model, tokenizer = self.model_loader(device=device)
            self.models.append(model)
            self.tokenizers.append(tokenizer)
            self.locks.append(threading.Lock())
            print(f"[POOL] GPU {i} ready")
        print(f"[POOL] All {n_gpus} models loaded")

    def _worker(self, gpu_id: int, item: any, fn: Callable) -> any:
        """Run fn on a specific GPU with its model."""
        with self.locks[gpu_id]:
            return fn(self.models[gpu_id], self.tokenizers[gpu_id], item)

    def map(self, items: list, fn: Callable, max_workers: Optional[int] = None) -> list:
        """
        Apply fn to each item in parallel across GPUs.

        Args:
            items: List of items to process
            fn: Function with signature (model, tokenizer, item) -> result
            max_workers: Max concurrent workers (default: n_gpus)

        Returns:
            List of results in same order as items
        """
        if max_workers is None:
            max_workers = self.n_gpus

        results = [None] * len(items)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, item in enumerate(items):
                gpu_id = i % self.n_gpus
                future = executor.submit(self._worker, gpu_id, item, fn)
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results


def evaluate_parallel(
    pool: MultiGPUWorkerPool,
    cached_pos: CachedActivations,
    cached_neg: CachedActivations,
    question: str,
    parse_fn: Callable,
    frac_layers: float,
    frac_tokens: float,
    alpha: float,
    n_examples: Optional[int] = None,
) -> float:
    """
    Evaluate accuracy using multiple GPUs in parallel.

    Args:
        pool: MultiGPUWorkerPool with loaded models
        cached_pos/neg: Cached activations (will be used per-GPU)
        question: The question to ask
        parse_fn: Function to parse yes/no from response
        frac_layers, frac_tokens, alpha: Hyperparameters
        n_examples: Max examples per class (None = all)

    Returns:
        Accuracy as a float
    """
    pos_texts = cached_pos.texts[:n_examples] if n_examples else cached_pos.texts
    neg_texts = cached_neg.texts[:n_examples] if n_examples else cached_neg.texts

    # Build work items: (text, expected_label, cache)
    work_items = [(t, True, cached_pos) for t in pos_texts] + \
                 [(t, False, cached_neg) for t in neg_texts]

    def process_item(model, tokenizer, item):
        text, expected, cache = item
        resp = patch_and_query_cached(
            model, tokenizer, cache, text, question,
            frac_patch_layers=frac_layers,
            frac_patch_tokens=frac_tokens,
            alpha=alpha, debug=False
        )
        parsed = parse_fn(resp)
        return parsed == expected

    results = pool.map(work_items, process_item)
    return sum(results) / len(results)


def grid_sweep_parallel(
    pool: MultiGPUWorkerPool,
    cached_pos: CachedActivations,
    cached_neg: CachedActivations,
    question: str,
    parse_fn: Callable,
    frac_layers_values: list,
    frac_tokens_values: list,
    alpha_values: list,
    n_examples: Optional[int] = None,
) -> dict:
    """
    Run a full grid sweep using parallel evaluation.

    Returns:
        Dict mapping (alpha, frac_layers, frac_tokens) -> accuracy
    """
    from tqdm.auto import tqdm

    results = {}
    total = len(alpha_values) * len(frac_layers_values) * len(frac_tokens_values)

    with tqdm(total=total, desc="Grid sweep") as pbar:
        for alpha in alpha_values:
            for fl in frac_layers_values:
                for ft in frac_tokens_values:
                    acc = evaluate_parallel(
                        pool, cached_pos, cached_neg, question, parse_fn,
                        frac_layers=fl, frac_tokens=ft, alpha=alpha,
                        n_examples=n_examples
                    )
                    results[(alpha, fl, ft)] = acc
                    pbar.update(1)
                    pbar.set_postfix({"acc": f"{acc:.1%}"})

    return results


def grid_sweep_hyperparam_parallel(
    pool: MultiGPUWorkerPool,
    cached_pos: CachedActivations,
    cached_neg: CachedActivations,
    question: str,
    parse_fn: Callable,
    frac_layers_values: list,
    frac_tokens_values: list,
    alpha_values: list,
    n_examples: Optional[int] = None,
) -> dict:
    """
    Run a full grid sweep with hyperparameter-level parallelism.

    Unlike grid_sweep_parallel (which parallelizes examples within each eval),
    this distributes different hyperparameter combinations across GPUs.
    Better when you have many GPUs and few examples per evaluation.

    Args:
        pool: MultiGPUWorkerPool with loaded models
        cached_pos/neg: Cached activations
        question: The question to ask
        parse_fn: Function to parse yes/no from response
        frac_layers_values: List of frac_layers values to sweep
        frac_tokens_values: List of frac_tokens values to sweep
        alpha_values: List of alpha values to sweep
        n_examples: Max examples per class (None = all)

    Returns:
        Dict mapping (alpha, frac_layers, frac_tokens) -> accuracy
    """
    from tqdm.auto import tqdm
    import itertools

    # Build all hyperparameter combinations
    hyperparam_combos = list(itertools.product(alpha_values, frac_layers_values, frac_tokens_values))

    pos_texts = cached_pos.texts[:n_examples] if n_examples else cached_pos.texts
    neg_texts = cached_neg.texts[:n_examples] if n_examples else cached_neg.texts

    def evaluate_single_hyperparam(model, tokenizer, combo):
        """Evaluate all examples for one hyperparameter combination using batched inference."""
        alpha, fl, ft = combo
        correct = 0
        total = len(pos_texts) + len(neg_texts)

        # Process positive examples in batch
        if pos_texts:
            pos_responses = patch_and_query_batched(
                model, tokenizer, cached_pos, list(pos_texts), question,
                frac_patch_layers=fl,
                frac_patch_tokens=ft,
                alpha=alpha,
            )
            for resp in pos_responses:
                if parse_fn(resp) == True:
                    correct += 1

        # Process negative examples in batch
        if neg_texts:
            neg_responses = patch_and_query_batched(
                model, tokenizer, cached_neg, list(neg_texts), question,
                frac_patch_layers=fl,
                frac_patch_tokens=ft,
                alpha=alpha,
            )
            for resp in neg_responses:
                if parse_fn(resp) == False:
                    correct += 1

        return correct / total if total > 0 else 0.0

    print(f"[SWEEP] Evaluating {len(hyperparam_combos)} hyperparameter combinations across {pool.n_gpus} GPUs...")

    # Pre-copy cached activations to all GPUs to avoid cross-GPU transfers
    print(f"[SWEEP] Pre-copying activations to {pool.n_gpus} GPUs...")
    for i in range(pool.n_gpus):
        device = f"cuda:{i}"
        cached_pos.to_device(device)
        cached_neg.to_device(device)
    print(f"[SWEEP] Activations copied, starting sweep...")

    # Use tqdm-wrapped results gathering
    results_list = []
    with ThreadPoolExecutor(max_workers=pool.n_gpus) as executor:
        futures = {}
        for i, combo in enumerate(hyperparam_combos):
            gpu_id = i % pool.n_gpus
            future = executor.submit(pool._worker, gpu_id, combo, evaluate_single_hyperparam)
            futures[future] = combo

        with tqdm(total=len(hyperparam_combos), desc="Grid sweep") as pbar:
            for future in as_completed(futures):
                combo = futures[future]
                acc = future.result()
                results_list.append((combo, acc))
                pbar.update(1)
                pbar.set_postfix({"last_acc": f"{acc:.1%}"})

    # Convert to dict
    results = {combo: acc for combo, acc in results_list}
    return results
