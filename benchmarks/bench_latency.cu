#include "fused_ssd.cuh"
#include "reference_scan.cuh"
#include "ssd_config.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// CUDA event timer helper
struct EventTimer {
    cudaEvent_t start, stop;
    EventTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~EventTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin(cudaStream_t s = nullptr) { cudaEventRecord(start, s); }
    float end(cudaStream_t s = nullptr) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Measure median latency over `reps` warm+measure iterations
static float bench_ms(auto kernel_fn, int warmup = 5, int reps = 20) {
    for (int i = 0; i < warmup; ++i) kernel_fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    EventTimer timer;
    for (int i = 0; i < reps; ++i) {
        timer.begin();
        kernel_fn();
        times.push_back(timer.end());
    }
    // Return median
    std::sort(times.begin(), times.end());
    return times[reps / 2];
}

// Allocate all device buffers for one configuration
struct DeviceBuffers {
    float  *A, *D, *delta;
    __half *B, *C, *x, *y;

    DeviceBuffers(const SSDConfig& cfg) {
        int bs = cfg.batch_size;
        CUDA_CHECK(cudaMalloc(&A,     (size_t)cfg.n_heads * cfg.d_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&D,     (size_t)cfg.n_heads * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&delta, (size_t)bs * cfg.seq_len * cfg.n_heads * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B,     (size_t)bs * cfg.seq_len * cfg.d_state * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&C,     (size_t)bs * cfg.seq_len * cfg.d_state * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&x,     (size_t)bs * cfg.seq_len * cfg.n_heads * cfg.d_head * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&y,     (size_t)bs * cfg.seq_len * cfg.n_heads * cfg.d_head * sizeof(__half)));

        // Fill with small random-ish values via memset patterns
        cudaMemset(A,     0x3E, (size_t)cfg.n_heads * cfg.d_state * sizeof(float));  // ~-0.2f
        cudaMemset(D,     0,    (size_t)cfg.n_heads * sizeof(float));
        cudaMemset(delta, 0x3C, (size_t)bs * cfg.seq_len * cfg.n_heads * sizeof(float)); // small positive
        cudaMemset(B,     0x3A, (size_t)bs * cfg.seq_len * cfg.d_state * sizeof(__half));
        cudaMemset(C,     0x3A, (size_t)bs * cfg.seq_len * cfg.d_state * sizeof(__half));
        cudaMemset(x,     0x3A, (size_t)bs * cfg.seq_len * cfg.n_heads * cfg.d_head * sizeof(__half));
    }

    ~DeviceBuffers() {
        cudaFree(A); cudaFree(D); cudaFree(delta);
        cudaFree(B); cudaFree(C); cudaFree(x); cudaFree(y);
    }
};

// Estimate bytes transferred for fused kernel (no intermediates)
static double fused_traffic_gb(const SSDConfig& cfg) {
    long long bs = cfg.batch_size;
    long long T  = cfg.seq_len;
    long long N  = cfg.d_state;
    long long Dh = cfg.d_head;
    long long H  = cfg.n_heads;
    // Reads: B, C, x, delta; A and D are small constants (cached)
    long long read  = bs * (T*N + T*N + T*H*Dh + T*H) * 2LL;  // fp16 for B,C,x; fp32 for delta
    // Writes: y
    long long write = bs * T * H * Dh * 2LL;
    return (double)(read + write) / 1e9;
}

// Estimate extra bytes for multi-launch baseline (adds intermediates)
static double multilaunched_extra_traffic_gb(const SSDConfig& cfg) {
    long long bs = cfg.batch_size;
    long long C  = cfg.n_chunks();
    long long N  = cfg.d_state;
    long long Dh = cfg.d_head;
    long long L  = cfg.chunk_size;
    long long H  = cfg.n_heads;
    long long T  = cfg.seq_len;
    // Intra-chunk intermediates: per-chunk cum_decay [T*H] + score matrix [C*H*L*L]
    long long intra_intermediate = bs * (T * H + C * H * L * L) * sizeof(float);
    // Inter-chunk summaries: C * H * N * Dh fp32
    long long inter_summary = bs * C * H * N * Dh * sizeof(float);
    // Discretized decay output: T * H * N fp32
    long long disc_output   = bs * T * H * N * sizeof(float);
    return (double)(intra_intermediate + inter_summary + disc_output) / 1e9;
}

int main() {
    // Mamba-2 370M configuration
    const int d_model    = 2048;
    const int d_state    = 64;
    const int n_heads    = 16;
    const int d_head     = d_model / n_heads;  // 128
    const int chunk_size = 128;
    const int batch_size = 1;

    int seq_lens[] = {256, 512, 1024, 2048, 4096, 8192, 16384};

    printf("%-8s  %-12s  %-12s  %-10s  %-10s  %-14s  %-14s\n",
           "seq_len", "fused_ms", "ref_ms", "ref/fused", "fused_BW",
           "fused_GB", "extra_saved_GB");
    printf("%s\n", std::string(100, '-').c_str());

    for (int sl : seq_lens) {
        if (sl % chunk_size != 0) continue;

        SSDConfig cfg;
        cfg.d_model    = d_model;
        cfg.d_state    = d_state;
        cfg.n_heads    = n_heads;
        cfg.d_head     = d_head;
        cfg.chunk_size = chunk_size;
        cfg.seq_len    = sl;
        cfg.batch_size = batch_size;

        // Check shared memory budget
        size_t smem = fused_ssd_smem_bytes(cfg);
        if (smem > 96 * 1024) {
            printf("seq=%5d: SKIP (smem=%zuKB > 96KB)\n", sl, smem/1024);
            continue;
        }

        DeviceBuffers bufs(cfg);
        SSDParams ref_p   = SSDParams::from_config(cfg, bufs.A, bufs.B, bufs.C, bufs.D, bufs.delta, bufs.x, bufs.y);
        SSDParams fuse_p  = SSDParams::from_config(cfg, bufs.A, bufs.B, bufs.C, bufs.D, bufs.delta, bufs.x, bufs.y);

        float fused_ms = bench_ms([&]{ fused_ssd_forward(cfg, fuse_p); });
        float ref_ms   = bench_ms([&]{ reference_ssd_forward(cfg, ref_p); });

        double traffic_gb = fused_traffic_gb(cfg);
        double extra_gb   = multilaunched_extra_traffic_gb(cfg);
        double bw_tbps    = traffic_gb / (fused_ms * 1e-3) / 1e12;

        printf("%-8d  %-12.3f  %-12.3f  %-10.2fx  %-10.4f  %-14.4f  %-14.4f\n",
               sl, fused_ms, ref_ms, ref_ms / fused_ms, bw_tbps,
               traffic_gb, extra_gb);
    }

    printf("\nNote: fused_BW = effective HBM bandwidth (TB/s); extra_saved_GB = global memory\n");
    printf("      traffic eliminated relative to a multi-launch implementation.\n");

    return 0;
}
