#include <iostream>
# include <ATen/ops/empty_native.h>
# include <ATen/ArrayRef.h>
# include <ATen/ScalarType.h>
# include <ATen/ops/mul_native.h>

#include "example.h"

const int TENSOR_EXAMPLE1_DIM0 = 2;
const int TENSOR_EXAMPLE1_DIM1 = 3;

// 执行命令之前设置环境变量TORCH_SHOW_DISPATCH_TRACE=true, 可以打印op tree

void show_detail_tensor(const torch::Tensor& t) {
  std::cout << "Tensor name: " << t.name() << std::endl;
  std::cout << "Tensor data: " << t << std::endl;
  if (t.requires_grad()) {
    if (t.is_leaf()) {
      std::cout << "Tensor grad: " << t.grad() << std::endl;
    } else {
      std::cout << "Tensor is not leaf." << std::endl;
    }
  } else {
    std::cout << "Tensor does not require grad." << std::endl;
  }
}

// 例子1: 直接使用aten库的简单例子, 但是无法使用自动微分的功能
void aten_tensor_test() {
  // 直接使用tensor库的简单例子, 但是无法使用自动微分的功能
  at::Tensor x = at::ones({1}, at::requires_grad());
  at::Tensor y = (x.multiply(x)).add(x);
  std::cout << "tensor x: "<< x << std::endl;
  std::cout << "tensor y: "<< y << std::endl;
}

// 例子2: 使用c++ api操作tensor, 可以使用自动微分功能
void torch_tensor_test(bool show) {
  torch::Tensor x1 = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, at::requires_grad());
  torch::autograd::impl::set_name(x1, "x1");
  torch::Tensor x2 = torch::tensor({{5.0, 6.0}, {7.0, 8.0}}, at::requires_grad());
  torch::autograd::impl::set_name(x2, "x2");
  auto y = x1.mul(2);
  torch::autograd::impl::set_name(y, "y");
  auto z = y.sum();
  torch::autograd::impl::set_name(z, "z");
  z.backward();
  if (show) {
    show_detail_tensor(x1);
    show_detail_tensor(x2);
    show_detail_tensor(y);
    show_detail_tensor(z);
  }
}

// 例子3: 使用linear C++ api
void linear_test() {
  torch::Tensor x = torch::rand({2, 2});
  torch::Tensor w = torch::rand({3, 2}, at::requires_grad());
  torch::Tensor b = torch::rand({3}, at::requires_grad());
  auto y = torch::linear(x, w, b);
  y.sum().backward();
  // std::cout << x << std::endl;
  // std::cout << y << std::endl;
  // std::cout << w.grad() << std::endl;
  // std::cout << b.grad() << std::endl;
}

// 例子4: 另外一个tensor操作例子
void tensor_test(bool show) {
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

  std::cout << "Tensor example2:" << std::endl;

  // 运算
  auto xt = at::native::transpose(x, 0, 1);
  if (show) {
    std::cout << "xt = " << xt << std::endl;
  }
  auto tmp1 = at::native::matmul(x, xt);
  if (show) {
    std::cout << "tmp1 = " << tmp1 << std::endl;
  }
  auto tmp2 = at::native::multiply(x, x);
  if (show) {
    std::cout << "tmp2 = " << tmp2 << std::endl;
  }
}


// https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/
void torch_tensor_test2(bool show) {
  torch::Tensor x1 = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, at::requires_grad());
  torch::autograd::impl::set_name(x1, "x1");
  torch::Tensor x2 = torch::tensor({{2.0, 3.0}, {4.0, 5.0}}, at::requires_grad());
  torch::autograd::impl::set_name(x2, "x2");
  torch::Tensor a = x1.mul(x2);
  torch::autograd::impl::set_name(a, "a");
  torch::Tensor y1 = a.log();
  torch::autograd::impl::set_name(y1, "y1");
  torch::Tensor y2 = x2.sin();
  torch::autograd::impl::set_name(y2, "y2");
  torch::Tensor w = y1.mul(y2);
  torch::autograd::impl::set_name(w, "w");
  auto z = w.sum();
  torch::autograd::impl::set_name(z, "z");
  z.backward();

  if (show) {
    show_detail_tensor(x1);
    show_detail_tensor(x2);
    show_detail_tensor(a);
    show_detail_tensor(y1);
    show_detail_tensor(y2);
    show_detail_tensor(w);
    show_detail_tensor(z);
  }
}