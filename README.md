# Activation Oracle (AO)

A library for extracting and injecting activations in language models for interpretability research.

## Features

- **Activation Extraction**: Extract hidden states from any layer of a transformer model
- **Activation Injection**: Inject activations at specified layers and token positions using convex combination blending
- **Batched Operations**: Efficient batched extraction and generation
- **Multi-GPU Support**: Parallel inference across multiple GPUs with `MultiGPUWorkerPool`
- **Cached Activations**: Pre-compute and cache activations for fast hyperparameter sweeps

## Installation

```bash
uv sync
```

For development dependencies (matplotlib, jupyter, etc.):

```bash
uv sync --extra dev
```

For Flash Attention 2 support:

```bash
uv sync --extra flash
```

## Quick Start

```python
from lib_ao import load_gemma, patch_and_query

# Load model
model, tokenizer = load_gemma()

# Extract activations from source text and inject into a question
response = patch_and_query(
    model, tokenizer,
    source_text="I love this movie! It's fantastic.",
    question="Is this text expressing positive sentiment? Answer yes or no.",
    frac_patch_layers=0.5,  # Patch top 50% of layers
    frac_patch_tokens=1.0,  # Use all tokens
    alpha=0.5,              # 50% blend of original and injected activations
)
```

## Core Functions

### Activation Extraction

- `get_activations(model, tokenizer, text, layer_indices)` - Extract activations for specific layers
- `get_activations_batched(...)` - Batched extraction for multiple texts
- `precompute_activations(...)` - Pre-compute and cache activations for a set of texts

### Activation Injection

- `patch_and_query(...)` - End-to-end extraction and injection
- `patch_and_query_cached(...)` - Use pre-computed activations
- `patch_and_query_batched(...)` - Batched generation with injection

### Evaluation

- `evaluate_batched_fast(...)` - Fast batched evaluation
- `evaluate_parallel(...)` - Multi-GPU parallel evaluation
- `grid_sweep_hyperparam_parallel(...)` - Parallel hyperparameter grid sweep

### Multi-GPU

- `MultiGPUWorkerPool` - Pool of models for parallel inference

## Layer/Token Selection

Layers are selected from the **top** (highest layers) downward:

```python
from lib_ao import get_layer_range

# For a 26-layer model, frac_patch_layers=0.5 gives layers 13-25
start, end, center = get_layer_range(n_layers=26, frac_patch_layers=0.5)
```

Tokens are selected from the **end** of the sequence:

```python
from lib_ao import get_token_indices

# For 100 tokens, frac_patch_tokens=0.5 gives indices 50-99
indices = get_token_indices(n_tokens=100, frac_patch_tokens=0.5)
```

## License

MIT
