#include <example.h>
#include <filesystem>

#include <gloo/algorithm.h>
#include <gloo/allreduce_ring.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/allreduce.h>
#include <gloo/transport/uv/device.h>
#include <c10/core/ScalarType.h>

namespace fs = std::filesystem;

void clear_directory(const fs::path& dir_path) {
  try {
    // 遍历目录内所有条目
    for (const auto& entry : fs::directory_iterator(dir_path)) {
      if (fs::is_directory(entry)) {
        // 递归删除子目录及其内容
        fs::remove_all(entry);
      } else {
        // 直接删除文件
        fs::remove(entry);
      }
    }
    std::cout << "目录内容已清空，保留根目录: " << dir_path << std::endl;
  } catch (const fs::filesystem_error& e) {
    std::cerr << "错误: " << e.what() << std::endl;
  }
}

void test_gloo_example_allreduce_ring(int rank) {
  // 1. 创建共享存储（用于节点发现）
  // 执行前需要清空/tmp/gloo_store: rm -rf /tmp/gloo_store/*
  gloo::rendezvous::FileStore store("/tmp/gloo_store");

  // 2. 创建传输设备（这里使用 TCP 设备）
  //    参数为设备的主机名或 IP（空字符串表示自动选择）
  gloo::transport::uv::attr attr;
  attr.hostname = "127.0.0.1";
  auto dev = gloo::transport::uv::CreateDevice(attr);

  // 3. 初始化通信上下文
  auto context = std::make_shared<gloo::rendezvous::Context>(rank, 2);
  context->connectFullMesh(store, dev);

  // 4. 准备数据缓冲区（输入和输出共用同一内存）
  int data[4] = {1 + rank, 2 + rank, 3 + rank, 4 + rank  };
  std::vector<int*> ptrs = {data};  // 指针列表（此处只有一个指针）
  std::cout << "setup data done" << std::endl;

  // 5. 执行 All-Reduce 操作（原地更新，结果直接写入 data）
  auto allreduce =
      std::make_shared<gloo::AllreduceRing<int>>(context, ptrs, 4);
  std::cout << "setup allreduce done" << std::endl;
  allreduce->run();
  std::cout << "allreduce done" << std::endl;

  // 输出结果（假设 4 个节点，每个节点的 data 变为 1+2+3+4=10.0）
  std::cout << "Result: ";
  for (int i = 0; i < 4; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

typedef void (*ReduceFunc)(void*, const void*, const void*, size_t);

void test_gloo_example_allreduce(int rank) {
  // 1. 创建共享存储（用于节点发现）
  // 执行前需要清空/tmp/gloo_store: rm -rf /tmp/gloo_store/*
  gloo::rendezvous::FileStore store("/tmp/gloo_store");

  // 2. 创建传输设备（这里使用 TCP 设备）
  //    参数为设备的主机名或 IP（空字符串表示自动选择）
  gloo::transport::uv::attr attr;
  attr.hostname = "127.0.0.1";
  auto dev = gloo::transport::uv::CreateDevice(attr);

  // 3. 初始化通信上下文
  auto context = std::make_shared<gloo::rendezvous::Context>(rank, 2);
  context->connectFullMesh(store, dev);

  // 4. 准备数据缓冲区（输入和输出共用同一内存）
  int data[4] = {1 + rank, 2 + rank, 3 + rank, 4 + rank};

  std::vector<int*> ptrs = {data};  // 指针列表（此处只有一个指针）
  int count = 4;  // 数据元素数量

  // 5. 执行 All-Reduce 操作（原地更新，结果直接写入data）
  gloo::AllreduceOptions opts(context);
  opts.setReduceFunction(ReduceFunc(&::gloo::sum<float>));
  opts.setTag(0);
  opts.setOutputs(ptrs, 4);
  gloo::allreduce(opts);

  // 输出结果（假设 4 个节点，每个节点的 data 变为 1+2+3+4=10.0）
  std::cout << "Result: ";
  for (int i = 0; i < 4; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

void gloo_example(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./gloo_example <rank>" << std::endl;
  }
  int rank = std::atoi(argv[1]);
  if (rank == 0) {
    // 先执行rank=0的进程
    clear_directory(fs::path("/tmp/gloo_store"));
  }
  test_gloo_example_allreduce(rank);
}
