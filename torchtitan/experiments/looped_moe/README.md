# Looped MoE

A Qwen3-MoE decoder whose whole layer stack is applied `num_loops` times with
shared weights:

```
tokens -> tok_embeddings -> [L0 -> L1 -> ... -> L7] x num_loops -> norm -> lm_head
                             \__ same weights every loop __/
```

Embeddings, attention, MoE (including its router), norms, lm_head and the
tokenizer are reused from `torchtitan/models/qwen3` unchanged. The parameter
count and checkpoint layout match the underlying Qwen3 flavor; only compute
grows with `num_loops`.

## Flavors

| Flavor | Qwen3 layer stack | `num_loops` | Effective depth |
| --- | --- | --- | --- |
| `debugmodel` | `debugmodel_moe` (8 layers, 64 experts, top-8) | 4 | 32 |

## Run

```bash
MODULE=torchtitan_recipes.models.looped_moe CONFIG=looped_moe_debug ./run_train.sh
```

## Status

- Not yet validated: no convergence or throughput runs.
- Pipeline parallelism is not supported (stages would split the loop).
- FSDP re-gathers each layer's weights on every loop.
- Residual-output init is scaled by the number of unique layers, not the
  effective depth.
