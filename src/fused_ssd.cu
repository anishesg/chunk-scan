#include "fused_ssd.cuh"
#include "ssd_config.cuh"
#include "discretize.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

// Shared memory layout within the fused kernel (all fp32):
//   B_smem    [chunk_size * d_state]
//   C_smem    [chunk_size * d_state]
//   x_smem    [chunk_size * d_head]
//   decay_smem[chunk_size]              per-step mean decay, then cum_decay
//   out_smem  [chunk_size * d_head]     intra-chunk output accumulator
//   state_smem[d_state * d_head]        running hidden state h_{c-1} at chunk start
//   scratch   [kWarpSize * (1+d_state*d_head)]  inter-chunk scan scratch (reserved)
//
// The kernel processes all chunks sequentially within the thread block.
// For the inter-chunk contribution, it maintains state_smem which it updates
// after each chunk using the full per-state decay and B contributions.

static __global__ void __launch_bounds__(256) fused_ssd_kernel(
    SSDConfig cfg,
    SSDParams p)
{
    const int batch = blockIdx.x;
    const int head  = blockIdx.y;
    const int L     = cfg.chunk_size;
    const int N     = cfg.d_state;
    const int Dh    = cfg.d_head;
    const int C     = cfg.n_chunks();
    const int T     = cfg.seq_len;
    const int tid   = threadIdx.x;
    const int lane  = tid & (kWarpSize - 1);
    const int wid   = tid / kWarpSize;

    extern __shared__ float smem[];

    // Partition shared memory
    float* B_smem     = smem;
    float* C_smem     = B_smem   + L * N;
    float* x_smem     = C_smem   + L * N;
    float* decay_smem = x_smem   + L * Dh;
    float* out_smem   = decay_smem + L;
    float* state_smem = out_smem + L * Dh;  // [N, Dh] running state h_{c-1}
    // state_smem is initialized to zero once before the chunk loop

    const float* A_head = p.A + head * N;
    const float  D_val  = p.D[head];

    // Batch offsets into global memory
    const __half* B_base     = p.B     + (long long)batch * p.B_batch_stride;
    const __half* C_base     = p.C     + (long long)batch * p.C_batch_stride;
    const float*  delta_base = p.delta + (long long)batch * p.delta_batch_stride;
    const __half* x_base     = p.x     + (long long)batch * p.x_batch_stride;
    __half*       y_base     = p.y     + (long long)batch * p.y_batch_stride;

    // Initialize running state to zero
    for (int i = tid; i < N * Dh; i += blockDim.x) state_smem[i] = 0.f;
    __syncthreads();

    // Process each chunk sequentially
    for (int c = 0; c < C; ++c) {
        int t0 = c * L;  // global start token index for this chunk

        // --- Load chunk data into shared memory ---

        // B: [L, N] for this chunk
        const __half* B_chunk = B_base + (long long)t0 * N;
        for (int i = tid; i < L * N; i += blockDim.x) {
            B_smem[i] = __half2float(B_chunk[i]);
        }

        // C: [L, N] for this chunk
        const __half* C_chunk = C_base + (long long)t0 * N;
        for (int i = tid; i < L * N; i += blockDim.x) {
            C_smem[i] = __half2float(C_chunk[i]);
        }

        // x: [L, Dh] for this chunk (strided: global x is [T, n_heads, Dh])
        for (int l = tid; l < L; l += blockDim.x) {
            const __half* xlt = x_base + ((long long)(t0 + l) * cfg.n_heads + head) * Dh;
            float* xs = x_smem + l * Dh;
            for (int d = 0; d < Dh; ++d) xs[d] = __half2float(xlt[d]);
        }

        // delta for this chunk (per-head scalar at each position)
        for (int l = tid; l < L; l += blockDim.x) {
            decay_smem[l] = delta_base[(long long)(t0 + l) * cfg.n_heads + head];
        }
        __syncthreads();

        // --- Compute per-step mean decay and then cumulative product ---
        // decay_smem[l] = mean_n exp(A[n] * delta[l])
        for (int l = tid; l < L; l += blockDim.x) {
            float dt = decay_smem[l];
            float mean = 0.f;
            for (int n = 0; n < N; ++n) mean += expf(A_head[n] * dt);
            decay_smem[l] = mean / N;
        }
        __syncthreads();

        // Thread 0 computes inclusive prefix product (L <= 256, serial is fine)
        if (tid == 0) prefix_product_serial(decay_smem, L);
        __syncthreads();

        // --- Initialize output accumulator ---
        for (int i = tid; i < L * Dh; i += blockDim.x) out_smem[i] = 0.f;
        __syncthreads();

        // --- Intra-chunk causal matmul ---
        // Each warp handles a subset of output rows.
        // Warp w processes rows: w, w + n_warps, w + 2*n_warps, ...
        int n_warps = blockDim.x / kWarpSize;
        for (int row = wid; row < L; row += n_warps) {
            float cd_i = decay_smem[row];
            const float* Ci = C_smem + row * N;

            for (int j = 0; j <= row; ++j) {
                float cd_j = decay_smem[j];
                float ratio = cd_i / cd_j;

                // dot(C[row], B[j]) -- each lane computes part and reduces via warp
                float dot = 0.f;
                const float* Bj = B_smem + j * N;
                for (int n = lane; n < N; n += kWarpSize) {
                    dot += Ci[n] * Bj[n];
                }
                // Warp reduction for dot product
                unsigned mask = 0xffffffff;
                for (int off = kWarpSize >> 1; off > 0; off >>= 1) {
                    dot += __shfl_down_sync(mask, dot, off);
                }
                dot = __shfl_sync(mask, dot, 0);  // broadcast to all lanes

                float score = dot * ratio;

                float* out_row = out_smem + row * Dh;
                const float* xj = x_smem + j * Dh;
                for (int d = lane; d < Dh; d += kWarpSize) {
                    atomicAdd(&out_row[d], score * xj[d]);
                }
            }

            // D * x[row] skip connection
            const float* xi = x_smem + row * Dh;
            float* out_row = out_smem + row * Dh;
            for (int d = lane; d < Dh; d += kWarpSize) {
                atomicAdd(&out_row[d], D_val * xi[d]);
            }
        }
        __syncthreads();

        // --- Inter-chunk contribution: add state_smem contribution to each output row ---
        // For row l in this chunk: y[l] += C[l]^T * diag(cum_decay[l] / cum_decay_chunk_start) * h_{c-1}
        // Since state_smem = h_{c-1} and cum_decay_chunk_start = 1 (relative to chunk start),
        // the chunk-relative cum_decay is just decay_smem[l] / (initial product before chunk start).
        // We use cum_decay_chunk_relative[l] = decay_smem[l] (already chunk-relative since
        // prefix product was computed from l=0 of this chunk).
        // The state correction is: y[l] += sum_n C[l,n] * cum_decay[l] * state_smem[n, :]
        for (int row = wid; row < L; row += n_warps) {
            float cd_rel = decay_smem[row];  // cum decay from start of this chunk to row
            const float* Ci = C_smem + row * N;

            // dot(C[row], cum_decay_scaled * h) for each output dimension d
            float* out_row = out_smem + row * Dh;
            for (int d = lane; d < Dh; d += kWarpSize) {
                float acc = 0.f;
                for (int n = 0; n < N; ++n) {
                    acc += Ci[n] * cd_rel * state_smem[n * Dh + d];
                }
                atomicAdd(&out_row[d], acc);
            }
        }
        __syncthreads();

        // --- Store output for this chunk to global memory ---
        for (int l = tid / Dh; l < L && Dh > 0; l += blockDim.x / Dh) {
            int d = tid % Dh;
            if (d < Dh) {
                __half* yt = y_base + ((long long)(t0 + l) * cfg.n_heads + head) * Dh + d;
                *yt = __float2half(out_smem[l * Dh + d]);
            }
        }
        // Simpler store loop
        for (int l = 0; l < L; ++l) {
            float* out_row = out_smem + l * Dh;
            __half* y_row  = y_base   + ((long long)(t0 + l) * cfg.n_heads + head) * Dh;
            for (int d = tid; d < Dh; d += blockDim.x) {
                y_row[d] = __float2half(out_row[d]);
            }
        }
        __syncthreads();

        // --- Update running hidden state h_c from h_{c-1} ---
        // h_c = (prod_{l=0}^{L-1} decay_l_per_state) * h_{c-1}  +  sum_{l=0}^{L-1} decay_{l+1..L} * B[l] * x[l]
        // We recompute per-state (not mean) decays here for correct state update.
        // Thread assignment: thread n handles state row n, iterates over Dh columns.
        // Reload delta for this chunk (decay_smem holds cum_mean_decay, overwrite with raw delta)
        for (int l = tid; l < L; l += blockDim.x) {
            decay_smem[l] = delta_base[(long long)(t0 + l) * cfg.n_heads + head];
        }
        __syncthreads();

        // Update state: for each state dimension n and head dimension d:
        // First apply full-chunk decay to h_{c-1}, then accumulate B[l]*x[l] contributions
        for (int n = tid; n < N; n += blockDim.x) {
            // Compute full-chunk per-state decay: prod_{l=0}^{L-1} exp(A[n]*delta[l])
            float total_decay = 1.f;
            for (int l = 0; l < L; ++l) {
                total_decay *= expf(A_head[n] * decay_smem[l]);
            }
            // Decay the previous state
            float* h_row = state_smem + n * Dh;
            for (int d = 0; d < Dh; ++d) h_row[d] *= total_decay;
        }
        __syncthreads();

        // Accumulate B[l]*x[l] contributions with correct decay weighting
        // Contribution of position l: decay_{l+1..L-1} * B[l,n] * x[l,d]
        // = (prod_{k=l+1}^{L-1} exp(A[n]*delta[k])) * B[l,n] * x[l,d]
        for (int n = tid; n < N; n += blockDim.x) {
            float* h_row = state_smem + n * Dh;
            for (int l = 0; l < L; ++l) {
                // Compute decay from l+1 to end of chunk
                float tail_decay = 1.f;
                for (int k = l + 1; k < L; ++k) {
                    tail_decay *= expf(A_head[n] * decay_smem[k]);
                }
                float b_val = B_smem[l * N + n];
                for (int d = 0; d < Dh; ++d) {
                    h_row[d] += tail_decay * b_val * x_smem[l * Dh + d];
                }
            }
        }
        __syncthreads();
    }  // end chunk loop
}

size_t fused_ssd_smem_bytes(const SSDConfig& cfg) {
    int L = cfg.chunk_size;
    int N = cfg.d_state;
    int Dh = cfg.d_head;
    // B_smem + C_smem + x_smem + decay_smem + out_smem + state_smem
    size_t bytes = (size_t)(L*N + L*N + L*Dh + L + L*Dh + N*Dh) * sizeof(float);
    return bytes;
}

void fused_ssd_forward(
    const SSDConfig& cfg,
    const SSDParams& params,
    cudaStream_t     stream)
{
    dim3 grid(cfg.batch_size, cfg.n_heads);
    int  threads = 256;
    size_t smem  = fused_ssd_smem_bytes(cfg);

    // Increase max dynamic shared memory if needed (up to 164KB on Ampere)
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(
            fused_ssd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
    }

    fused_ssd_kernel<<<grid, threads, smem, stream>>>(cfg, params);
}
