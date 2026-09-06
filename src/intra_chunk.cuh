#pragma once

#include "ssd_config.cuh"
#include "discretize.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Intra-chunk structured causal matmul.
//
// For a chunk of L tokens with precomputed cumulative decay products cum_decay[l]
// (where cum_decay[l] = prod_{k=0}^{l} decay[k]), computes:
//
//   output[i, d] += sum_{j=0}^{i} M[i,j] * x[j, d]
//
// where M[i,j] = (C[i]^T * B[j]) * (cum_decay[i] / cum_decay[j])  for j <= i
//              = (C[i]^T * B[j]) * prod_{k=j+1}^{i} decay[k]
//
// This is the quadratic (attention-like) form of SSM within one chunk.
// The output is accumulated into `out_acc` (fp32) which is then added to
// the inter-chunk contribution before storing to global memory.
//
// Parameters:
//   B_smem     : shared mem tile [L, d_state], fp32
//   C_smem     : shared mem tile [L, d_state], fp32
//   x_smem     : shared mem tile [L, d_head],  fp32
//   cum_decay  : shared mem [L], fp32, inclusive prefix product of per-step decays
//   out_acc    : register accumulator [d_head per thread], fp32
//   L          : chunk size
//   N          : d_state
//   Dh         : d_head
//   thread_d   : the d_head column this thread is responsible for
//
// Thread layout: one warp per output row i, each lane handles a subset of d_head columns.
// For d_head <= 128 and 32 lanes, each lane processes ceil(d_head/32) columns.

__device__ __forceinline__ void intra_chunk_causal_matmul(
    const float* __restrict__ B_smem,      // [L, N]
    const float* __restrict__ C_smem,      // [L, N]
    const float* __restrict__ x_smem,      // [L, Dh]
    const float* __restrict__ cum_decay,   // [L]
    float*                    out_acc,     // [Dh_per_thread] accumulator, modified
    int                       L,
    int                       N,
    int                       Dh,
    int                       row_i,       // which output row this call computes
    int                       lane)        // thread lane within warp (0..31)
{
    float cd_i = (row_i < L) ? cum_decay[row_i] : 0.f;

    // For each source position j <= i, accumulate M[i,j] * x[j, :]
    for (int j = 0; j <= row_i && j < L; ++j) {
        // Score M[i,j] = dot(C[i], B[j]) * (cum_decay[i] / cum_decay[j])
        // cum_decay[j] > 0 always since decays are exp(negative)
        float cd_j = cum_decay[j];
        float ratio = cd_i / cd_j;  // prod_{k=j+1}^{i} decay[k]

        // Compute dot product C[i] . B[j] over d_state
        float dot = 0.f;
        const float* Ci = C_smem + row_i * N;
        const float* Bj = B_smem + j     * N;
        // Unrolled accumulation; each thread handles the full N since it's a scalar score
        for (int n = 0; n < N; ++n) {
            dot += Ci[n] * Bj[n];
        }
        float score = dot * ratio;

        // Accumulate score * x[j, :] into output
        const float* xj = x_smem + j * Dh;
        for (int d = lane; d < Dh; d += kWarpSize) {
            out_acc[d / kWarpSize] += score * xj[d];
        }
    }
}

// Block-level intra-chunk kernel wrapper.
// Each thread block processes one (batch, head) pair for a single chunk.
// Fills shared memory from global, computes cum_decay, then runs the causal matmul
// for all L output positions and stores partial results to `out_smem` [L, Dh].
//
// This is factored as a device function so the fused kernel can call it inline
// without additional kernel launch overhead.
__device__ void intra_chunk_forward(
    const __half* __restrict__ B_global,   // [L, N] for this chunk, current batch/head
    const __half* __restrict__ C_global,   // [L, N]
    const __half* __restrict__ x_global,   // [L, Dh]
    const float*  __restrict__ A_head,     // [N] log-decay rates for this head
    const float*  __restrict__ delta_global, // [L] timestep scalars for this head
    float*                     out_smem,   // [L, Dh] output accumulator (zeroed by caller)
    float*                     B_smem,     // [L, N] scratch
    float*                     C_smem,     // [L, N] scratch
    float*                     x_smem,     // [L, Dh] scratch
    float*                     decay_smem, // [L] scratch for per-step then cum decay
    float                      D_val,
    int                        L,
    int                        N,
    int                        Dh)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int wid  = tid / kWarpSize;

    // Load B, C into shared memory (fp16 -> fp32 conversion)
    for (int li = tid; li < L * N; li += blockDim.x) {
        int l = li / N, n = li % N;
        B_smem[li] = __half2float(B_global[li]);
        C_smem[li] = __half2float(C_global[li]);
        (void)(l); (void)(n);
    }

    // Load x into shared memory
    for (int li = tid; li < L * Dh; li += blockDim.x) {
        x_smem[li] = __half2float(x_global[li]);
    }

    // Compute per-step decay scalars: decay[l] = exp(sum_n A[n] * delta[l] / N)
    // We use a mean over d_state as the representative scalar for cum_decay weighting.
    // The actual per-state decays are recomputed per-element in the state update.
    for (int l = tid; l < L; l += blockDim.x) {
        float dt = delta_global[l];
        float mean_decay = 0.f;
        for (int n = 0; n < N; ++n) {
            mean_decay += expf(A_head[n] * dt);
        }
        decay_smem[l] = mean_decay / N;
    }
    __syncthreads();

    // Compute inclusive prefix product of decay_smem (cum_decay)
    // Use serial pass by thread 0 for simplicity (L <= 256)
    if (tid == 0) {
        prefix_product_serial(decay_smem, L);
    }
    __syncthreads();

    // Each warp handles one output row i
    // warp `wid` handles rows: wid, wid + n_warps, ...
    int n_warps = blockDim.x / kWarpSize;
    for (int i = wid; i < L; i += n_warps) {
        // Each lane accumulates into its d_head elements
        // out_acc[k] corresponds to column lane + k*kWarpSize
        float out_acc[4] = {0.f, 0.f, 0.f, 0.f};  // supports up to 4*32=128 d_head

        intra_chunk_causal_matmul(
            B_smem, C_smem, x_smem, decay_smem,
            out_acc, L, N, Dh, i, lane);

        // Add D * x[i] (skip connection)
        const float* xi = x_smem + i * Dh;
        for (int d = lane; d < Dh; d += kWarpSize) {
            out_acc[d / kWarpSize] += D_val * xi[d];
        }

        // Store to out_smem[i, :]
        float* out_row = out_smem + i * Dh;
        for (int d = lane; d < Dh; d += kWarpSize) {
            out_row[d] += out_acc[d / kWarpSize];
        }
    }
    __syncthreads();
}
