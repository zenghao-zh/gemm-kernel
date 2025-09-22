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

torch::Tensor matmul_w4a8x4(const torch::Tensor &A1, const torch::Tensor &B1, 
                            const torch::Tensor &A2, const torch::Tensor &B2)
{
  // Check all tensors are contiguous
  torch::checkAllContiguous("matmul_w4a8x4", {{A1, "A1", 0}, {B1, "B1", 1}, 
                                               {A2, "A2", 2}, {B2, "B2", 3}});
  
  // Check all tensors are on CUDA
  torch::checkDeviceType("matmul_w4a8x4", {A1, B1, A2, B2}, at::DeviceType::CUDA);

  // Check all tensors are on the same GPU
  torch::checkAllSameGPU("matmul_w4a8x4", {{A1, "A1", 0}, {B1, "B1", 1}, 
                                            {A2, "A2", 2}, {B2, "B2", 3}});

  // Extract dimensions for first GEMM (8bit x 8bit)
  uint32_t M1 = A1.size(0);
  uint32_t N1 = B1.size(0);  
  uint32_t K1 = A1.size(1);

  // Extract dimensions for second GEMM (4bit x 4bit)
  uint32_t M2 = A2.size(0);
  uint32_t N2 = B2.size(0);
  uint32_t K2 = A2.size(1) * kElementsPerVector; // 4bit packing is on the columns

  // Validate that output dimensions match
  TORCH_CHECK(M1 == M2 && N1 == N2, 
              "Output dimensions must match: M1=", M1, " M2=", M2, " N1=", N1, " N2=", N2);

  // Create output tensor
  auto C = torch::empty({M1, N1}, torch::dtype(torch::kInt32).device(A1.device()));

  // Call the host function
  matmul_w4a8x4_host(
      A1.data_ptr<int8_t>(),      // A1: 8bit activations
      B1.data_ptr<int8_t>(),      // B1: 8bit weights  
      A2.data_ptr<Int4Storage>(), // A2: 4bit activations
      B2.data_ptr<Int4Storage>(), // B2: 4bit weights
      M1, N1, K1,                 // First GEMM dimensions
      M2, N2, K2,                 // Second GEMM dimensions
      C.data_ptr<int32_t>()       // Output
  );

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
  m.def("matmul_w4a8x4", &matmul_w4a8x4,
        "input: (A1: torch.Tensor(M x K1, INT8, CUDA), B1: torch.Tensor(N x K1, INT8, CUDA),\n"
        "        A2: torch.Tensor(M x K2, UINT8, CUDA), B2: torch.Tensor(N x K2, UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = (A1 @ B1^T) + (int4Unpacking(A2) @ int4Unpacking(B2)^T)",
        py::arg("A1"), py::arg("B1"), py::arg("A2"), py::arg("B2"));
}

//================ TorchScript 注册 =================//
// TORCH_LIBRARY(my_ops, m) {
//   m.def("matmul_w4a4(Tensor A, Tensor B) -> Tensor");
//   m.def("matmul_w8a8(Tensor A, Tensor B) -> Tensor");
//   m.def("matmul_w4a8(Tensor A, Tensor B) -> Tensor");
//   m.def("matmul_w4a8x4(Tensor A1, Tensor B1, Tensor A2, Tensor B2) -> Tensor");
// }

// TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
//   m.impl("matmul_w4a4", matmul_w4a4);
//   m.impl("matmul_w8a8", matmul_w8a8);
//   m.impl("matmul_w4a8", matmul_w4a8);
//   m.impl("matmul_w4a8x4", matmul_w4a8x4);
// }