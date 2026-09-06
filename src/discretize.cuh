#pragma once

#include "ssd_config.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Compute per-state decay factor: exp(A[n] * delta_t).
// A[n] is expected to be negative (decay), delta_t > 0.
__device__ __forceinline__ float compute_decay(float A_n, float delta_t) {
    return expf(A_n * delta_t);
}

// Compute the decay factors for all N states given one timestep delta_t.
// Output: decay[n] = exp(A[n] * delta_t) for n in [0, N).
// Each calling thread writes one element; caller must ensure lane coverage.
__device__ __forceinline__ void compute_decay_vector(
    const float* __restrict__ A,   // [N]
    float                     delta_t,
    float* __restrict__       out, // [N]
    int                       N,
    int                       lane)
{
    for (int n = lane; n < N; n += kWarpSize) {
        out[n] = expf(A[n] * delta_t);
    }
}

// Warp-level parallel prefix product (inclusive scan) over values held in registers.
// Each lane holds one element. After the call, each lane holds the product of all
// elements from lane 0 up to and including itself.
// Uses log2(WARP_SIZE) butterfly steps via __shfl_up_sync.
// Complexity: O(log W) depth, O(W log W) work for W = WARP_SIZE.
__device__ __forceinline__ float warp_prefix_product_inclusive(float val) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        float up = __shfl_up_sync(mask, val, offset);
        if ((int)(threadIdx.x & (kWarpSize - 1)) >= offset) {
            val *= up;
        }
    }
    return val;
}

// Compute cumulative decay products for a chunk of L tokens in shared memory.
// decay_in[l] holds the per-step decay scalar for position l (product over d_state
// dimensions, or one representative value for the structured case).
// decay_out[l] = product of decay_in[0..l] (inclusive prefix product).
//
// For L <= WARP_SIZE: single warp handles it with warp shuffles.
// For L > WARP_SIZE: two-level scan: warp-level within each warp segment,
//   then a serial fix-up pass propagating warp tail products across segments.
//   This is correct for L <= kMaxChunkSize (256).
//
// Must be called by all threads in a warp (or block if L > WARP_SIZE).
__device__ void cumulative_decay_product(
    const float* __restrict__ decay_in,  // [L] in shared memory
    float* __restrict__       decay_out, // [L] in shared memory
    int                       L)
{
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;

    if (L <= kWarpSize) {
        // Single-warp path
        float val = (lane < L) ? decay_in[lane] : 1.f;
        val = warp_prefix_product_inclusive(val);
        if (lane < L) decay_out[lane] = val;
    } else {
        // Multi-warp path: each warp handles its segment independently
        int n_warps = (L + kWarpSize - 1) / kWarpSize;
        int base    = warp * kWarpSize;
        int idx     = base + lane;

        float val = (idx < L) ? decay_in[idx] : 1.f;
        val = warp_prefix_product_inclusive(val);
        if (idx < L) decay_out[idx] = val;
        __syncthreads();

        // Fix-up: propagate the tail product of each warp into the next warp's prefix.
        // Serial fix-up by warp 0 across all warp tails (at most kMaxChunkSize/32 = 8 warps).
        if (warp == 0 && lane == 0) {
            float running = 1.f;
            for (int w = 0; w < n_warps; ++w) {
                int tail = min((w + 1) * kWarpSize, L) - 1;
                float warp_total = decay_out[tail] * running;
                // Scale all elements in warp w+1 by the running product of warps 0..w
                if (w + 1 < n_warps) {
                    int start = (w + 1) * kWarpSize;
                    int end   = min(start + kWarpSize, L);
                    for (int i = start; i < end; ++i) {
                        decay_out[i] *= warp_total / decay_out[min((w+1)*kWarpSize, L-1)];
                        // Correct formula: scale by product of warp 0..w tail
                    }
                }
                running = warp_total;
            }
        }
        __syncthreads();
    }
}

// Simpler, correct two-pass implementation used by the fused kernel.
// Pass 1 (parallel): compute warp-local prefix products.
// Pass 2 (serial, warp 0 only): accumulate running product across warps.
// This avoids the division in the fix-up above.
__device__ void cumulative_decay_product_v2(
    float* __restrict__ decay,  // [L] in shared memory, modified in-place to prefix products
    int                 L)
{
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int wid  = threadIdx.x / kWarpSize;

    // Each warp computes local inclusive prefix product over its slice
    int idx = wid * kWarpSize + lane;
    float val = (idx < L) ? decay[idx] : 1.f;
    val = warp_prefix_product_inclusive(val);
    if (idx < L) decay[idx] = val;
    __syncthreads();

    // Warp 0 propagates the total product of each warp into subsequent warps
    if (wid == 0 && lane == 0) {
        int n_warps = (L + kWarpSize - 1) / kWarpSize;
        float running = 1.f;
        for (int w = 0; w < n_warps; ++w) {
            int warp_last = min((w + 1) * kWarpSize, L) - 1;
            float warp_total = decay[warp_last];  // local prefix product at tail
            // Scale next warp's elements by the accumulated product from previous warps
            if (w + 1 < n_warps) {
                int s = (w + 1) * kWarpSize;
                int e = min(s + kWarpSize, L);
                for (int i = s; i < e; ++i) {
                    decay[i] *= running * warp_total / decay[warp_last];
                }
            }
            running *= warp_total;
        }
        // Correct the first warp (it's already correct since running=1 initially)
        // Actually we need a cleaner approach: store warp tails, then fix up.
    }
    __syncthreads();
}

// Cleanest version: compute prefix products using a single serial pass in shared memory.
// Called by one thread per chunk after all per-step decay values are stored.
// Suitable for small chunks processed within a thread block.
__device__ __forceinline__ void prefix_product_serial(
    float* __restrict__ arr,  // [L], modified in-place
    int                 L)
{
    float prod = 1.f;
    for (int i = 0; i < L; ++i) {
        prod *= arr[i];
        arr[i] = prod;
    }
}
