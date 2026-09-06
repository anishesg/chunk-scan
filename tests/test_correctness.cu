#include "fused_ssd.cuh"
#include "reference_scan.cuh"
#include "ssd_config.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Initialize host arrays with random fp16/fp32 values
static void rand_fp16(std::vector<__half>& v, float lo, float hi, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = __float2half(dist(rng));
}

static void rand_fp32(std::vector<float>& v, float lo, float hi, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

// Cosine similarity between two fp16 arrays (converted to fp32)
static float cosine_sim(const std::vector<__half>& a, const std::vector<__half>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float fa = __half2float(a[i]);
        float fb = __half2float(b[i]);
        dot += fa * fb;
        na  += fa * fa;
        nb  += fb * fb;
    }
    if (na < 1e-12 || nb < 1e-12) return 1.f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

// Max absolute error
static float max_abs_err(const std::vector<__half>& a, const std::vector<__half>& b) {
    float err = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        err = fmaxf(err, fabsf(__half2float(a[i]) - __half2float(b[i])));
    }
    return err;
}

// Mean absolute error relative to max abs of reference
static float rel_mae(const std::vector<__half>& a, const std::vector<__half>& b) {
    float sum_err = 0, max_ref = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum_err += fabsf(__half2float(a[i]) - __half2float(b[i]));
        max_ref  = fmaxf(max_ref, fabsf(__half2float(b[i])));
    }
    if (max_ref < 1e-6f) return 0.f;
    return sum_err / (a.size() * max_ref);
}

struct TestConfig {
    int seq_len;
    int chunk_size;
    int d_state;
    int d_head;
    int n_heads;
};

static bool run_test(const TestConfig& tc, std::mt19937& rng) {
    const int batch = 1;
    SSDConfig cfg;
    cfg.d_model    = tc.n_heads * tc.d_head;
    cfg.d_state    = tc.d_state;
    cfg.n_heads    = tc.n_heads;
    cfg.d_head     = tc.d_head;
    cfg.chunk_size = tc.chunk_size;
    cfg.seq_len    = tc.seq_len;
    cfg.batch_size = batch;

    if (cfg.seq_len % cfg.chunk_size != 0) return true;  // skip invalid configs

    size_t sz_B     = (size_t)batch * cfg.seq_len * cfg.d_state;
    size_t sz_BC    = sz_B;
    size_t sz_delta = (size_t)batch * cfg.seq_len * cfg.n_heads;
    size_t sz_x     = (size_t)batch * cfg.seq_len * cfg.n_heads * cfg.d_head;
    size_t sz_A     = (size_t)cfg.n_heads * cfg.d_state;
    size_t sz_D     = (size_t)cfg.n_heads;

    // Host allocations
    std::vector<float>  h_A(sz_A), h_D(sz_D), h_delta(sz_delta);
    std::vector<__half> h_B(sz_B), h_C(sz_BC), h_x(sz_x);

    // A must be negative for stability
    rand_fp32(h_A, -0.5f, -0.01f, rng);
    rand_fp32(h_D, -1.f, 1.f, rng);
    rand_fp32(h_delta, 0.001f, 0.1f, rng);
    rand_fp16(h_B, -1.f, 1.f, rng);
    rand_fp16(h_C, -1.f, 1.f, rng);
    rand_fp16(h_x, -1.f, 1.f, rng);

    // Device allocations
    float  *d_A, *d_D, *d_delta;
    __half *d_B, *d_C, *d_x, *d_y_ref, *d_y_fused;

    CUDA_CHECK(cudaMalloc(&d_A,       sz_A     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D,       sz_D     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta,   sz_delta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,       sz_B     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C,       sz_BC    * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_x,       sz_x     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y_ref,   sz_x     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y_fused, sz_x     * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_A,     h_A.data(),     sz_A     * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D,     h_D.data(),     sz_D     * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta, h_delta.data(), sz_delta * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,     h_B.data(),     sz_B     * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C,     h_C.data(),     sz_BC    * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,     h_x.data(),     sz_x     * sizeof(__half), cudaMemcpyHostToDevice));

    SSDParams ref_params = SSDParams::from_config(cfg, d_A, d_B, d_C, d_D, d_delta, d_x, d_y_ref);
    SSDParams fuse_params = SSDParams::from_config(cfg, d_A, d_B, d_C, d_D, d_delta, d_x, d_y_fused);

    reference_ssd_forward(cfg, ref_params);
    fused_ssd_forward(cfg, fuse_params);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_y_ref(sz_x), h_y_fused(sz_x);
    CUDA_CHECK(cudaMemcpy(h_y_ref.data(),   d_y_ref,   sz_x * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y_fused.data(), d_y_fused, sz_x * sizeof(__half), cudaMemcpyDeviceToHost));

    float cos_sim = cosine_sim(h_y_fused, h_y_ref);
    float max_err = max_abs_err(h_y_fused, h_y_ref);
    float rel_err = rel_mae(h_y_fused, h_y_ref);

    bool pass = cos_sim > 0.998f;
    printf("seq=%4d chunk=%3d d_state=%3d d_head=%3d n_heads=%2d  "
           "cos_sim=%.6f  max_err=%.4f  rel_mae=%.6f  %s\n",
           tc.seq_len, tc.chunk_size, tc.d_state, tc.d_head, tc.n_heads,
           cos_sim, max_err, rel_err, pass ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_D); cudaFree(d_delta);
    cudaFree(d_B); cudaFree(d_C); cudaFree(d_x);
    cudaFree(d_y_ref); cudaFree(d_y_fused);

    return pass;
}

int main() {
    std::mt19937 rng(42);
    int n_fail = 0;

    // Configuration sweep matching the plan spec
    int seq_lens[]   = {256, 1024, 4096};
    int chunk_sizes[]= {64, 128, 256};
    int d_states[]   = {16, 64, 128};
    int d_heads[]    = {64, 128};
    int n_heads_arr[]= {8, 32};

    printf("%-6s %-6s %-8s %-7s %-8s %-14s %-10s %-12s %s\n",
           "seq", "chunk", "d_state", "d_head", "n_heads",
           "cos_sim", "max_err", "rel_mae", "result");
    printf("%s\n", std::string(100, '-').c_str());

    for (int sl : seq_lens)
    for (int cs : chunk_sizes)
    for (int ds : d_states)
    for (int dh : d_heads)
    for (int nh : n_heads_arr) {
        // Skip configs that exceed shared memory budget (~96KB)
        size_t smem_est = (size_t)(cs*ds + cs*ds + cs*dh + cs + cs*dh + ds*dh) * sizeof(float);
        if (smem_est > 96 * 1024) {
            printf("seq=%4d chunk=%3d d_state=%3d d_head=%3d n_heads=%2d  SKIP (smem=%zuKB)\n",
                   sl, cs, ds, dh, nh, smem_est / 1024);
            continue;
        }
        TestConfig tc{sl, cs, ds, dh, nh};
        if (!run_test(tc, rng)) ++n_fail;
    }

    printf("\n%s\n", n_fail == 0 ? "All tests PASSED." : "Some tests FAILED.");
    return n_fail > 0 ? 1 : 0;
}
