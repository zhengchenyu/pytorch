#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/torch.h>

const std::uint16_t PORT = 12345;

void test1_server() {
  // initial store
  c10d::TCPStoreOptions opt{.port = PORT, .numWorkers = 2, .isServer=true};
  c10d::TCPStore server_tcp_store {"127.0.0.1", opt};
  c10d::PrefixStore server_prefix_store {"/prefix", c10::intrusive_ptr<c10d::TCPStore>{&server_tcp_store, {}}};

  // set key
  server_prefix_store.set("key1", "value1");
  std::cout << "[Server] key1 = " << server_prefix_store.get("key1") << std::endl;
  std::cout << "[Server] key2 = " << server_prefix_store.get("key2") << std::endl;

  // initial process group
  // c10d::ProcessGroupGloo backend {};
  // c10d::ProcessGroup pg {};
}

void test1_client() {
  // initial store
  c10d::TCPStoreOptions opt{.port = PORT, .numWorkers = 2, .isServer=false};
  c10d::TCPStore client_tcp_store {"127.0.0.1", opt};
  c10d::PrefixStore client_prefix_store {"/prefix", c10::intrusive_ptr<c10d::TCPStore>{&client_tcp_store, {}}};

  // set key
  client_prefix_store.set("key2", "value2");
  std::cout << "[Client] key1 = " << client_prefix_store.get("key1") << std::endl;
  std::cout << "[Client] key2 = " << client_prefix_store.get("key2") << std::endl;

  // initial process group


}

void test1(bool is_server) {
  if (is_server) {
    test1_server();
  } else {
    test1_client();
  }
}

