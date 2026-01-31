# multipatch

This repo investigates the influence of patching a supervisee's activations onto a supervisor. In contrast to work on activation oracles, where patching was mostly done at the last timestep only and in some layer only, we here explore the effect of
1. patching multiple layers
2. patching multiple timesteps
3. patching with at strengths alpha, $\text{supervisor} = (1-\alpha)\cdot \text{base} + \alpha\cdot \text{supervisee}$

## Installation

```bash
uv sync
```