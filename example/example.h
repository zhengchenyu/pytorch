//
// Created by zhengchenyu on 2024/11/19.
//

#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <Aten/ATen.h>
#include <torch/torch.h>

void basic_aten();
void basic_cpp_api();
void basic_linear();
void mnist();
void tensor_example1(bool show = false);
void test1(bool is_server);

#endif //EXAMPLE_H
