#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "fused_ssd.cuh"
#include "reference_scan.cuh"
#include "ssd_config.cuh"

// Validate common tensor properties
static void check_tensor(const torch::Tensor& t, const char* name, bool fp16 = true) {
    TORCH_CHECK(t.is_cuda(),      name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    if (fp16)
        TORCH_CHECK(t.scalar_type() == torch::kHalf, name, " must be float16");
    else
        TORCH_CHECK(t.scalar_type() == torch::kFloat, name, " must be float32");
}

// Build SSDConfig from tensor shapes. Expected layouts:
//   x     : (batch, seq_len, n_heads, d_head)  fp16
//   B, C  : (batch, seq_len, d_state)           fp16
//   delta : (batch, seq_len, n_heads)            fp32
//   A     : (n_heads, d_state)                  fp32
//   D     : (n_heads,)                           fp32
static SSDConfig make_config(
    const torch::Tensor& x,
    const torch::Tensor& B,
    int chunk_size)
{
    SSDConfig cfg;
    cfg.batch_size = x.size(0);
    cfg.seq_len    = x.size(1);
    cfg.n_heads    = x.size(2);
    cfg.d_head     = x.size(3);
    cfg.d_state    = B.size(2);
    cfg.d_model    = cfg.n_heads * cfg.d_head;
    cfg.chunk_size = chunk_size;

    TORCH_CHECK(cfg.seq_len % cfg.chunk_size == 0,
                "seq_len must be divisible by chunk_size");
    return cfg;
}

torch::Tensor fused_ssd_forward_py(
    const torch::Tensor& x,       // (B, T, H, Dh) fp16
    const torch::Tensor& B_proj,  // (B, T, N)     fp16
    const torch::Tensor& C_proj,  // (B, T, N)     fp16
    const torch::Tensor& A,       // (H, N)        fp32
    const torch::Tensor& D,       // (H,)          fp32
    const torch::Tensor& delta,   // (B, T, H)     fp32
    int                  chunk_size)
{
    check_tensor(x,      "x",     true);
    check_tensor(B_proj, "B",     true);
    check_tensor(C_proj, "C",     true);
    check_tensor(A,      "A",     false);
    check_tensor(D,      "D",     false);
    check_tensor(delta,  "delta", false);

    TORCH_CHECK(x.dim()      == 4, "x must be 4D (batch, seq_len, n_heads, d_head)");
    TORCH_CHECK(B_proj.dim() == 3, "B must be 3D (batch, seq_len, d_state)");
    TORCH_CHECK(C_proj.dim() == 3, "C must be 3D (batch, seq_len, d_state)");
    TORCH_CHECK(A.dim()      == 2, "A must be 2D (n_heads, d_state)");
    TORCH_CHECK(D.dim()      == 1, "D must be 1D (n_heads,)");
    TORCH_CHECK(delta.dim()  == 3, "delta must be 3D (batch, seq_len, n_heads)");

    int batch   = x.size(0);
    int seq_len = x.size(1);
    int n_heads = x.size(2);
    int d_state = B_proj.size(2);

    TORCH_CHECK(B_proj.size(0) == batch   && B_proj.size(1) == seq_len,
                "B batch/seq dims must match x");
    TORCH_CHECK(C_proj.size(0) == batch   && C_proj.size(1) == seq_len &&
                C_proj.size(2) == d_state, "C dims must match B");
    TORCH_CHECK(A.size(0)      == n_heads && A.size(1)      == d_state,
                "A dims must be (n_heads, d_state)");
    TORCH_CHECK(D.size(0)      == n_heads, "D must have n_heads elements");
    TORCH_CHECK(delta.size(0)  == batch   && delta.size(1)  == seq_len &&
                delta.size(2)  == n_heads, "delta dims must match (batch, seq_len, n_heads)");

    SSDConfig cfg = make_config(x, B_proj, chunk_size);

    // Check shared memory budget
    size_t smem = fused_ssd_smem_bytes(cfg);
    TORCH_CHECK(smem <= 164 * 1024,
                "Configuration requires ", smem / 1024, " KB shared memory, max 164 KB");

    auto y = torch::empty_like(x);

    SSDParams params = SSDParams::from_config(
        cfg,
        A.data_ptr<float>(),
        reinterpret_cast<const __half*>(B_proj.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(C_proj.data_ptr<at::Half>()),
        D.data_ptr<float>(),
        delta.data_ptr<float>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>())
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_ssd_forward(cfg, params, stream);

    return y;
}

torch::Tensor reference_ssd_forward_py(
    const torch::Tensor& x,
    const torch::Tensor& B_proj,
    const torch::Tensor& C_proj,
    const torch::Tensor& A,
    const torch::Tensor& D,
    const torch::Tensor& delta,
    int                  chunk_size)
{
    check_tensor(x,      "x",     true);
    check_tensor(B_proj, "B",     true);
    check_tensor(C_proj, "C",     true);
    check_tensor(A,      "A",     false);
    check_tensor(D,      "D",     false);
    check_tensor(delta,  "delta", false);

    SSDConfig cfg = make_config(x, B_proj, chunk_size);
    auto y = torch::empty_like(x);

    SSDParams params = SSDParams::from_config(
        cfg,
        A.data_ptr<float>(),
        reinterpret_cast<const __half*>(B_proj.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(C_proj.data_ptr<at::Half>()),
        D.data_ptr<float>(),
        delta.data_ptr<float>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>())
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    reference_ssd_forward(cfg, params, stream);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ssd_forward",
          &fused_ssd_forward_py,
          "Fused SSD forward pass (single-launch chunk processing)",
          py::arg("x"), py::arg("B"), py::arg("C"), py::arg("A"),
          py::arg("D"), py::arg("delta"), py::arg("chunk_size") = 128);

    m.def("reference_ssd_forward",
          &reference_ssd_forward_py,
          "Sequential reference SSM forward (correctness oracle)",
          py::arg("x"), py::arg("B"), py::arg("C"), py::arg("A"),
          py::arg("D"), py::arg("delta"), py::arg("chunk_size") = 128);
}
