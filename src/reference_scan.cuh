#pragma once

#include "ssd_config.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Sequential reference SSM forward pass. Correctness oracle with no optimization.
// Computes: h[t] = diag(exp(A*dt)) * h[t-1] + B[t]*x[t], y[t] = C[t]^T * h[t] + D*x[t]
void reference_ssd_forward(
    const SSDConfig& cfg,
    const SSDParams& params,
    cudaStream_t     stream = nullptr);
