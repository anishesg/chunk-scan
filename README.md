# chunk-scan

Fused chunk-wise State Space Dual (SSD) kernel: tensor-core intra-chunk matmul with warp-shuffle inter-chunk parallel scan in a single CUDA kernel launch.

## Background: State Space Duality

Mamba-2 introduced SSD, which reformulates the SSM recurrence as a structured causal attention-like computation. Given input x, projections B, C, diagonal decay A, and timestep delta, the SSM state evolves as:

```
h[t] = diag(exp(A * delta[t])) * h[t-1] + B[t] * x[t]
y[t] = C[t]^T * h[t] + D * x[t]
```

This is equivalent to a weighted causal matmul where the weight between positions i and j is:

```
M[i,j] = C[i]^T * (prod_{k=j+1}^{i} diag(decay[k])) * B[j]  for j <= i
```

## Performance Bottleneck in Triton Implementations

Current Triton-based SSD implementations (as in mamba-ssm) decompose the computation into:

1. Discretization kernel: compute decay = exp(A * delta), output to global memory
2. Intra-chunk kernel: load decay from global memory, compute L x L causal score matrix per chunk
3. Inter-chunk kernel: load chunk summaries from global memory, run sequential or parallel scan

Each kernel launch introduces global memory round-trips for intermediate results. For a sequence of length T with chunk size L and C = T/L chunks:

- Intra-chunk intermediates: C x L x d_state floats written and re-read (typically ~64MB for T=4096)
- Inter-chunk summaries: C x d_state x d_head floats (typically ~8MB)
- Total extra traffic: proportional to T * d_state * (1 + d_head/L)

On an A100 (2 TB/s HBM bandwidth), this extra traffic costs 30-60 microseconds per forward pass, which is significant at the 200-400 microsecond total latency of a Mamba-2 layer.

## Fused Approach

This implementation processes all C chunks in a single kernel launch:

1. Each thread block handles one (batch, head) pair
2. Shared memory holds the current chunk's B, C, x, and decay arrays
3. Within each chunk, intra-chunk causal matmul runs using register-tiled accumulation
4. After each chunk, inter-chunk state is updated and scanned forward using warp-level `__shfl_sync` primitives
5. The inter-chunk state contribution is added to the local chunk output before storing

For d_state <= 64, all state vectors fit in registers and the inter-chunk scan uses pure warp shuffles with zero shared memory traffic. For larger states, staging uses shared memory but avoids global memory entirely.

The associative operator for the parallel scan over chunk summaries is:
```
(decay1, state1) * (decay2, state2) = (decay1 * decay2, decay2 * state1 + state2)
```

This is used with Blelloch's work-efficient parallel scan to process up to 1024 chunks in O(log C) steps.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./test_correctness
./bench_latency
```

Requires CUDA 11.8+ and a GPU with compute capability 8.0+ (Ampere or newer).

## Python Extension

```bash
pip install -e .
```

```python
from chunk_scan import SSDLayer
layer = SSDLayer.from_config(d_model=2048, d_state=64, n_heads=16, chunk_size=128)
y = layer(x)  # dispatches to fused CUDA kernel
```

## Repository Layout

```
src/           CUDA kernel headers and implementation files
tests/         Correctness tests comparing fused vs reference
benchmarks/    Latency benchmarks with bandwidth analysis
csrc/          PyTorch C++ extension bindings
chunk_scan/    Python module (SSDLayer nn.Module)
```
