//
// Created by zhengchenyu on 2024/11/19.
//

#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <Aten/ATen.h>
#include <torch/torch.h>

// tensor example
void aten_tensor_test();
void torch_tensor_test(bool show);
void linear_test();
void tensor_test(bool show);
void torch_tensor_test2(bool show);

// store example
void test_tcpstore(bool is_server);

// process group example
void process_group_example(bool isMaster);

// gloo example
void gloo_example(bool rank);

#endif //EXAMPLE_H
