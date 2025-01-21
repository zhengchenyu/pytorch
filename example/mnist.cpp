#include <torch/torch.h>

// 由于C++ API的dataset是不会自动下载的，所以需要手动下载
const char* kDataRoot = "/Users/zhengchenyu/cache/MNIST/raw";
const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 1000;
const int64_t kNumberOfEpochs = 10;
const int64_t kLogInterval = 10;

struct Net: torch::nn::Module {
  Net() : layer1(torch::nn::Linear(28 * 28, 512)),
          layer2(torch::nn::Linear(512, 256)),
          layer3(torch::nn::Linear(256, 10)),
          flatten(torch::nn::Flatten()){
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
  }
  torch::Tensor forward(torch::Tensor x) {
    x = flatten(x);
    // std::cout << "sizes: " << x.sizes() << std::endl;
    x = torch::relu(layer1->forward(x));
    x = torch::relu(layer2->forward(x));
    return torch::softmax(layer3->forward(x), 1);
  }
  torch::nn::Linear layer1;
  torch::nn::Linear layer2;
  torch::nn::Linear layer3;
  torch::nn::Flatten flatten;
};

template <typename DataLoader>
void train(size_t epoch, Net &model, torch::Device device, DataLoader &data_loader, torch::optim::Optimizer &optimizer,
           size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::cross_entropy_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          (long)(batch_idx * batch.data.size(0)),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(Net &model, torch::Device device, DataLoader &data_loader, size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(output, targets,/*weight=*/{}, torch::Reduction::Sum).template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }
  test_loss /= dataset_size;
  std::printf("\nTest set: Average loss: %.4f | Accuracy: %.3f\n", test_loss,static_cast<double>(correct) / dataset_size);
}

// 单机多卡mnist
void mnist() {
  // 1 init device
  torch::manual_seed(1);
  torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  torch::Device device(device_type);

  // 2 int model
  Net model;
  model.to(device);

  // 3 get train and test set
  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_dataset), kTrainBatchSize);
  auto test_dataset = torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  // 4 define optimizer
  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  // 5 train and test
  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }
}