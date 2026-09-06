#pragma once

#include "ssd_config.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Host-side entry point for the fused SSD forward pass.
// Launches one thread block per (batch, head) pair.
// Dynamic shared memory is computed based on cfg dimensions.
void fused_ssd_forward(
    const SSDConfig& cfg,
    const SSDParams& params,
    cudaStream_t     stream = nullptr);

// Returns the required dynamic shared memory in bytes for the given config.
size_t fused_ssd_smem_bytes(const SSDConfig& cfg);
