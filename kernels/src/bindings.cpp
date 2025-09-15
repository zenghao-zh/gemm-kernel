#include <torch/extension.h>

// Include all files
#include <gemm.h>

torch::Tensor matmul_w4a4(const torch::Tensor &A, const torch::Tensor &B)
{
  torch::checkAllContiguous("matmul", {{A, "A", 0},
                                       {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0},
                                    {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1) * kElementsPerVector; // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_w4a4_host(A.data_ptr<Int4Storage>(), B.data_ptr<Int4Storage>(), M, N, K, C.data_ptr<int32_t>());

  return C;
}

torch::Tensor matmul_w8a8(const torch::Tensor &A, const torch::Tensor &B)
{
  torch::checkAllContiguous("matmul", {{A, "A", 0},
                                       {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0},
                                    {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1);
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_w8a8_host(A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), M, N, K, C.data_ptr<int32_t>());

  return C;
}

torch::Tensor matmul_w4a8(const torch::Tensor &A, const torch::Tensor &B)
{
  torch::checkAllContiguous("matmul", {{A, "A", 0},
                                       {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0},
                                    {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1);
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_w4a8_host(A.data_ptr<int8_t>(), B.data_ptr<Int4Storage>(), M, N, K, C.data_ptr<int32_t>());

  return C;
}


//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("matmul_w4a4", &matmul_w4a4,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));
  m.def("matmul_w4a8", &matmul_w4a8,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));
  m.def("matmul_w8a8", &matmul_w8a8,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"));
}