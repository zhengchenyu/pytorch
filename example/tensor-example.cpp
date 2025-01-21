#include <iostream>
# include <ATen/ops/empty_native.h>
# include <ATen/ArrayRef.h>
# include <ATen/ScalarType.h>
# include <ATen/ops/mul_native.h>

#include "example.h"

const int TENSOR_EXAMPLE1_DIM0 = 2;
const int TENSOR_EXAMPLE1_DIM1 = 3;

// 1 tensor example
void tensor_example1(bool show) {
  // 构造Tensor
  at::Tensor x {
    at::native::empty_cpu(
        at::IntArrayRef{TENSOR_EXAMPLE1_DIM0, TENSOR_EXAMPLE1_DIM1},
        std::optional(c10::ScalarType::Float),
        std::optional(c10::Layout::Strided),
        std::optional(c10::Device(c10::kCPU)),
        std::optional(false),
        std::optional(c10::MemoryFormat::Contiguous))
  };

  // 初始化Tensor
  for (int i = 0; i < TENSOR_EXAMPLE1_DIM0; i++) {
    for (int j = 0; j < TENSOR_EXAMPLE1_DIM1; j++) {
      at::Tensor t1 {at::native::select_symint(x, 0, i)};
      at::Tensor t2 {at::native::select_symint(t1, 0, j)};
      at::native::fill_(t2, i * TENSOR_EXAMPLE1_DIM1 + j);
    }
  }

  if (show) {
    // 第一种打印方式
    std::cout << "x = " << x << std::endl;
    // 第二种打印方式
    float* fp = reinterpret_cast<float*>(x.data_ptr());
    for (int i = 0; i < TENSOR_EXAMPLE1_DIM0; i++) {
      for (int j = 0; j < TENSOR_EXAMPLE1_DIM1; j++) {
        std::cout << *(fp + (i * TENSOR_EXAMPLE1_DIM1 + j)) << " ";
      }
      std::cout << std::endl;
    }
  }

  // 运算
  auto xt = at::native::transpose(x, 0, 1);
  if (show) {
    std::cout << "xt = " << xt << std::endl;
  }
  // auto tmp1 = at::native::mul(x, xt);
  auto tmp1 = at::native::multiply(x, xt);

  if (show) {
    std::cout << "tmp1 = " << tmp1 << std::endl;
  }
}