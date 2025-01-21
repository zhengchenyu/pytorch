#include "example.h"

int main(int argc, char* argv[]) {
  // basic_cpp_api();
  // tensor_example1(true);
  if (argc != 2) {
    std::cout << "you should input one parameter!" << std::endl;
    return 1;
  }
  test1(std::strcmp("server", argv[1]) == 0);
}
