#include "example.h"

int main(int argc, char* argv[]) {
  // 1 Tensor example
  // aten_tensor_test();
  torch_tensor_test(true);
  // linear_test();
  // tensor_test(true);

  // 2 store example
  // if (argc != 2) {
  //   std::cout << "you should input one parameter!" << std::endl;
  //   return 1;
  // }
  // test_tcpstore(std::strcmp("server", argv[1]) == 0);

  // 3 process group example
  // process_group_example(argc > 1 && strcmp(argv[1], "server") == 0);

  // 4 gloo example
  // if (argc != 2) {
  //   std::cerr << "Usage: ./gloo_example <rank>" << std::endl;
  // }
  // int rank = std::atoi(argv[1]);
  // gloo_example(rank);
}
