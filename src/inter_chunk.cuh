#pragma once

#include "ssd_config.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Associative operator for the inter-chunk parallel scan.
// Each chunk c produces a summary (scalar decay_c, state_c) where:
//   decay_c = product of all per-step decay scalars within chunk c
//   state_c = hidden state *generated* by chunk c starting from zero initial state
//
// The operator * combines two summaries as:
//   (d1, s1) * (d2, s2) = (d1*d2, d2*s1 + s2)
//
// This is the associative semigroup for linear recurrences:
//   h_after = d2 * h_before + s2,  where h_before incorporates d1*s0 + s1

// Compact representation of one chunk's contribution.
// For d_state * d_head > register budget (>= 64*64 = 4096 floats), use shared memory staging.
// We support d_state up to 128 via shared memory.

// Warp-level inclusive prefix scan over 32 (decay, state) pairs stored in registers.
// Specialized for d_state <= 64 so state fits in registers (64 floats per thread).
// For larger d_state, the state lives in shared memory and this function is not used.
//
// After return: each lane l holds the combined (decay, state) for chunks 0..l inclusive.
template <int N>
__device__ __forceinline__ void warp_scan_register(
    float&        decay,      // in/out: scalar decay for this lane's chunk
    float         state[N])   // in/out: state vector for this lane's chunk
{
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        float d_prev = __shfl_up_sync(mask, decay, offset);
        float s_prev[N];
        #pragma unroll
        for (int n = 0; n < N; ++n) {
            s_prev[n] = __shfl_up_sync(mask, state[n], offset);
        }
        int lane = threadIdx.x & (kWarpSize - 1);
        if (lane >= offset) {
            // Combine: (d_prev, s_prev) * (decay, state) = (d_prev*decay, decay*s_prev + state)
            #pragma unroll
            for (int n = 0; n < N; ++n) {
                state[n] = decay * s_prev[n] + state[n];
            }
            decay = d_prev * decay;
        }
    }
}

// Two-level parallel scan over up to 1024 chunks with state in shared memory.
// Level 1: intra-warp scan (32 chunks per warp, up to 32 warps for 1024 chunks).
// Level 2: inter-warp scan over warp tail values (serial, run by warp 0).
//
// Parameters:
//   chunk_decays : [n_chunks] scalar decay per chunk (global product of exp(A*dt))
//   chunk_states : [n_chunks, N, Dh] state generated per chunk from zero init
//   scan_decays  : [n_chunks] output: prefix-product decay from chunk 0..c inclusive
//   scan_states  : [n_chunks, N, Dh] output: prefix-sum state incorporating history
//   n_chunks     : number of chunks (must be <= kMaxChunks = 1024)
//   N            : d_state
//   Dh           : d_head
//   smem_scratch : shared memory scratch [2 * kWarpSize * (1 + N * Dh)] floats
//                  for inter-warp communication (n_warps <= 32)
//
// Each thread handles one chunk; blockDim.x must be >= n_chunks (up to 1024 threads).
// For n_chunks <= 32, only one warp is needed.
__device__ void inter_chunk_parallel_scan(
    const float* __restrict__ chunk_decays,  // [n_chunks]
    const float* __restrict__ chunk_states,  // [n_chunks * N * Dh]
    float*                    scan_decays,   // [n_chunks] output
    float*                    scan_states,   // [n_chunks * N * Dh] output
    int                       n_chunks,
    int                       N,
    int                       Dh,
    float*                    smem_scratch)  // [kWarpSize * (1 + N*Dh)] floats
{
    const int tid  = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int wid  = tid / kWarpSize;

    // Partition of smem_scratch: [kWarpSize] for warp tail decays,
    //                            [kWarpSize * N * Dh] for warp tail states.
    float* warp_tail_decays = smem_scratch;
    float* warp_tail_states = smem_scratch + kWarpSize;  // [kWarpSize, N, Dh]

    // Load this thread's chunk summary
    float my_decay = (tid < n_chunks) ? chunk_decays[tid] : 1.f;
    // my_state: for large N*Dh we keep it in shared memory;
    // here we use a fixed-size register array for N*Dh <= 16*128 = 2048 elements.
    // For the scan itself we load/store from chunk_states directly.

    // Level 1: warp-level inclusive prefix scan
    // Each warp scans its slice [wid*32 .. (wid+1)*32) of chunks.
    // The scan combines (decay, state) pairs; state lives in global memory so we
    // stream it through registers one (N*Dh)-sized block at a time, but for the
    // decay prefix product we can do a pure register scan first.

    // Step A: compute warp-level decay prefix product
    {
        unsigned mask = 0xffffffff;
        float d = my_decay;
        #pragma unroll
        for (int off = 1; off < kWarpSize; off <<= 1) {
            float d_up = __shfl_up_sync(mask, d, off);
            if (lane >= off) d *= d_up;
        }
        if (tid < n_chunks) scan_decays[tid] = d;
        // Store warp tail decay for level-2
        if (lane == kWarpSize - 1 || tid == n_chunks - 1) {
            warp_tail_decays[wid] = d;
        }
    }
    __syncthreads();

    // Step B: warp 0 computes exclusive prefix of warp tails (serial for <= 32 warps)
    if (tid == 0) {
        int n_warps = (n_chunks + kWarpSize - 1) / kWarpSize;
        float running = 1.f;
        for (int w = 0; w < n_warps; ++w) {
            float wt = warp_tail_decays[w];
            warp_tail_decays[w] = running;  // exclusive prefix
            running *= wt;
        }
    }
    __syncthreads();

    // Step C: multiply each thread's prefix decay by its warp's exclusive prefix
    if (tid < n_chunks) {
        scan_decays[tid] *= warp_tail_decays[wid];
    }
    __syncthreads();

    // Step D: compute scan over states using the corrected decay prefixes.
    // scan_states[c] = sum_{k=0}^{c-1} (scan_decays[c] / scan_decays[k+1]) * chunk_states[k]
    //               + chunk_states[c]
    // Equivalently: scan_states[c] = chunk_states[c] + decay_{k+1..c} * scan_states[c-1]
    // This is a forward sequential pass since we need h_c = decay_c * h_{c-1} + s_c.
    // For parallel prefix over states, use the same butterfly pattern applied to
    // (decay, state) pairs where state is the N*Dh vector.

    // We implement an in-place parallel prefix scan over the state vectors.
    // Due to the large state size (N*Dh), we do this in a warp-serial fashion
    // (warp 0 does the inter-warp fixup); intra-warp uses shared memory staging.

    // Intra-warp prefix scan over states: each warp handles 32 consecutive chunks
    // For each butterfly step, we accumulate into shared memory buffers.
    // This is O(N*Dh * log(n_chunks)) work, acceptable for N*Dh <= 128*128 = 16384.

    // Initialize scan_states from chunk_states
    for (int idx = tid; idx < n_chunks * N * Dh; idx += blockDim.x) {
        scan_states[idx] = chunk_states[idx];
    }
    __syncthreads();

    // Forward sequential scan (simple, correct implementation for any n_chunks)
    // Run by thread 0 to avoid race conditions in the state accumulation.
    if (tid == 0) {
        for (int c = 1; c < n_chunks; ++c) {
            float dc = scan_decays[c] / scan_decays[c - 1];  // decay from c-1 to c
            float* sc   = scan_states + c       * N * Dh;
            float* sc_1 = scan_states + (c - 1) * N * Dh;
            for (int i = 0; i < N * Dh; ++i) {
                sc[i] = dc * sc_1[i] + sc[i];
            }
        }
    }
    __syncthreads();
}

// Compute the chunk-level decay scalar (product of all exp(A*dt) within chunk c)
// and the chunk's zero-init state contribution.
// decay_prod = prod_{l=0}^{L-1} mean_n(exp(A[n] * delta[l]))
// This is the representative scalar used in the inter-chunk scan.
__device__ __forceinline__ float chunk_decay_product(
    const float* __restrict__ A_head,      // [N]
    const float* __restrict__ delta_chunk, // [L] timestep for this chunk and head
    int L, int N)
{
    float prod = 1.f;
    for (int l = 0; l < L; ++l) {
        float dt = delta_chunk[l];
        float mean = 0.f;
        for (int n = 0; n < N; ++n) mean += expf(A_head[n] * dt);
        prod *= mean / N;
    }
    return prod;
}
