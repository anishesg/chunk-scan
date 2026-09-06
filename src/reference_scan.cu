#include "reference_scan.cuh"
#include <cmath>
#include <cstring>

// One thread block per (batch, head) pair. Processes all T timesteps sequentially.
// State buffer h[d_state x d_head] lives in shared memory when it fits; otherwise global.
__global__ void reference_ssm_kernel(
    const SSDConfig cfg,
    const SSDParams p)
{
    const int batch = blockIdx.x;
    const int head  = blockIdx.y;
    const int T     = cfg.seq_len;
    const int N     = cfg.d_state;
    const int Dh    = cfg.d_head;

    // Shared memory: h[N * Dh] float32 state matrix
    extern __shared__ float smem[];
    float* h = smem;  // [N, Dh]

    // Initialize state to zero
    for (int i = threadIdx.x; i < N * Dh; i += blockDim.x) {
        h[i] = 0.f;
    }
    __syncthreads();

    const float* A_head     = p.A + head * N;           // [N]
    const float* D_head     = p.D + head;                // scalar
    const float  D_val      = *D_head;

    // Batch offsets
    const __half* B_base     = p.B     + (long long)batch * p.B_batch_stride;
    const __half* C_base     = p.C     + (long long)batch * p.C_batch_stride;
    const float*  delta_base = p.delta + (long long)batch * p.delta_batch_stride;
    const __half* x_base     = p.x     + (long long)batch * p.x_batch_stride;
    __half*       y_base     = p.y     + (long long)batch * p.y_batch_stride;

    // Sequential loop over timesteps
    for (int t = 0; t < T; ++t) {
        // Pointers to current timestep
        const __half* Bt    = B_base     + (long long)t * N;
        const __half* Ct    = C_base     + (long long)t * N;
        float         dt    = delta_base[(long long)t * cfg.n_heads + head];
        const __half* xt    = x_base     + ((long long)t * cfg.n_heads + head) * Dh;
        __half*       yt    = y_base     + ((long long)t * cfg.n_heads + head) * Dh;

        // Step 1: decay and update each state row
        // h[n, :] = exp(A[n] * dt) * h[n, :] + B[t, n] * x[t, :]
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            float decay   = expf(A_head[n] * dt);
            float b_val   = __half2float(Bt[n]);
            float* h_row  = h + n * Dh;
            for (int d = 0; d < Dh; ++d) {
                float x_val = __half2float(xt[d]);
                h_row[d] = decay * h_row[d] + b_val * x_val;
            }
        }
        __syncthreads();

        // Step 2: output projection y[t] = C[t]^T * h + D * x[t]
        for (int d = threadIdx.x; d < Dh; d += blockDim.x) {
            float acc = D_val * __half2float(xt[d]);
            for (int n = 0; n < N; ++n) {
                acc += __half2float(Ct[n]) * h[n * Dh + d];
            }
            yt[d] = __float2half(acc);
        }
        __syncthreads();
    }
}

void reference_ssd_forward(
    const SSDConfig& cfg,
    const SSDParams& params,
    cudaStream_t     stream)
{
    dim3 grid(cfg.batch_size, cfg.n_heads);
    // Use as many threads as needed to cover d_head and d_state in parallel
    int threads = 128;
    // Shared memory: state matrix h[d_state * d_head] fp32
    size_t smem_bytes = (size_t)cfg.d_state * cfg.d_head * sizeof(float);

    reference_ssm_kernel<<<grid, threads, smem_bytes, stream>>>(cfg, params);
}
