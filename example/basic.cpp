#include "example.h"
#include <iostream>

// 执行命令之前设置环境变量TORCH_SHOW_DISPATCH_TRACE=true, 可以打印op tree

// 直接使用aten库的简单例子, 但是无法使用自动微分的功能
void basic_aten() {
  // 直接使用tensor库的简单例子, 但是无法使用自动微分的功能
  at::Tensor x = at::ones({1}, at::requires_grad());
  at::Tensor y = (x.multiply(x)).add(x);
  std::cout << "tensor x: "<< x << std::endl;
  std::cout << "tensor y: "<< y << std::endl;
}

// 使用c++ api
void basic_cpp_api() {
  torch::Tensor x = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, at::requires_grad());
  torch::autograd::impl::set_name(x, "x");
  auto y = x.mul(x);
  torch::autograd::impl::set_name(y, "y");
  auto z = y.sum();
    torch::autograd::impl::set_name(z, "z");
  z.backward();
  // std::cout << "x = " << x << std::endl;
  // std::cout << "y = " << y << std::endl;
  // std::cout << "z = " << z << std::endl;
  // std::cout << "x.name = " << x.name() << std::endl;
  // std::cout << "y.name = " << y.name() << std::endl;
  // std::cout << "z.name = " << z.name() << std::endl;
  // std::cout << "x.grad = " << x.grad() << std::endl;
  // std::cout << "y.grad = " << y.grad() << std::endl;
  // std::cout << "z.grad = " << z.grad() << std::endl;
}

void basic_linear() {
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