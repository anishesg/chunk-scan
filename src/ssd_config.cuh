#pragma once

#include <cstdint>

// Alignment requirement for coalesced 128-byte cache line access on Ampere.
// All device buffers should be allocated with at least this alignment.
static constexpr int kAlignBytes = 128;
static constexpr int kAlignElems16 = kAlignBytes / sizeof(__half);  // 64 fp16 elements
static constexpr int kAlignElems32 = kAlignBytes / sizeof(float);   // 32 fp32 elements

// Maximum chunk size supported by the intra-chunk kernel (fits in shared mem).
static constexpr int kMaxChunkSize = 256;

// Maximum number of chunks supportable by the two-level inter-chunk scan.
static constexpr int kMaxChunks = 1024;

// Warp size (architectural constant for all NVIDIA GPUs).
static constexpr int kWarpSize = 32;

// Model and sequence dimensions for one SSD layer.
struct SSDConfig {
    int d_model;     // Total model dimension
    int d_state;     // SSM state dimension (N in Mamba-2 notation)
    int n_heads;     // Number of SSM heads
    int d_head;      // Dimension per head: d_model / n_heads
    int chunk_size;  // Number of tokens per chunk (L in SSD notation)
    int seq_len;     // Total sequence length (must be divisible by chunk_size)
    int batch_size;  // Batch size

    int n_chunks() const { return seq_len / chunk_size; }

    // Stride for a (seq_len, n_heads) tensor in row-major layout.
    int head_stride() const { return n_heads; }

    // Stride for a (seq_len, d_state) tensor in row-major layout.
    int state_stride() const { return d_state; }
};

// Pointers to all SSD parameters and inputs on device.
// All arrays are in row-major (C-contiguous) layout.
struct SSDParams {
    // A: diagonal decay log-rates, shape (n_heads, d_state), fp32
    // Stored negative; effective decay = exp(A_i * delta_t) with A_i < 0.
    const float* A;

    // B: input-dependent state projection, shape (batch, seq_len, d_state), fp16
    const __half* B;

    // C: output projection, shape (batch, seq_len, d_state), fp16
    const __half* C;

    // D: skip connection scalar, shape (n_heads,), fp32
    const float* D;

    // delta: input-dependent timestep, shape (batch, seq_len, n_heads), fp32
    const float* delta;

    // x: input activations, shape (batch, seq_len, n_heads, d_head), fp16
    const __half* x;

    // y: output activations, shape (batch, seq_len, n_heads, d_head), fp16 (write)
    __half* y;

    // Strides for batch dimension (number of elements to skip per batch item).
    int B_batch_stride;      // seq_len * d_state
    int C_batch_stride;      // seq_len * d_state
    int delta_batch_stride;  // seq_len * n_heads
    int x_batch_stride;      // seq_len * n_heads * d_head
    int y_batch_stride;      // seq_len * n_heads * d_head

    static SSDParams from_config(const SSDConfig& cfg,
                                 const float*  A,
                                 const __half* B,
                                 const __half* C,
                                 const float*  D,
                                 const float*  delta,
                                 const __half* x,
                                 __half*       y) {
        SSDParams p;
        p.A = A; p.B = B; p.C = C; p.D = D; p.delta = delta; p.x = x; p.y = y;
        p.B_batch_stride     = cfg.seq_len * cfg.d_state;
        p.C_batch_stride     = cfg.seq_len * cfg.d_state;
        p.delta_batch_stride = cfg.seq_len * cfg.n_heads;
        p.x_batch_stride     = cfg.seq_len * cfg.n_heads * cfg.d_head;
        p.y_batch_stride     = cfg.seq_len * cfg.n_heads * cfg.d_head;
        return p;
    }
};
