#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <example.h>

void test_process_group_example(int rank) {
  // 1 setup
  setenv("MASTER_ADDR", "127.0.0.1", 1);
  setenv("MASTER_PORT", "29500", 1);
  int world_size = 2;

  // 2 创建TCPStore和ProcessGroup
  c10::intrusive_ptr<c10d::Store> store =
    c10::make_intrusive<c10d::TCPStore>("127.0.0.1",
      c10d::TCPStoreOptions{.port = 29500, .numWorkers = world_size, .isServer=(rank == 0)});
  auto options = c10::make_intrusive<c10d::ProcessGroupGloo::Options>();
  options->timeout = std::chrono::seconds(10);
  // 指定Gloo
  options->devices.push_back(
      c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1")
  );
  auto pg = c10::make_intrusive<c10d::ProcessGroupGloo>(
    store, rank, world_size, options);

  // 3 计算
  std::vector<torch::Tensor> tensors = {torch::tensor({rank, rank + 1})};
  std::cout << "before all reduce, tensor = " << tensors[0] << std::endl;
  c10d::AllreduceOptions opts;
  opts.reduceOp = c10d::ReduceOp::SUM;
  auto work = pg->allreduce(tensors, opts);
  work->wait(std::chrono::milliseconds(10000000));
  std::cout << "after all reduce, tensor = " << tensors[0] << std::endl;
}

void process_group_example(int argc, char* argv[]) {
  bool isMaster = argc > 1 && strcmp(argv[1], "server") == 0;
  if (isMaster) {
    test_process_group_example(0);
  } else {
    test_process_group_example(1);
  }
}
