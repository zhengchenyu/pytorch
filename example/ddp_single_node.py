import os

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import Optimizer
from torch.nn.modules import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.distributed.optim import ZeroRedundancyOptimizer

batch_size = 64
epochs = 5
snapshot_path = "../../tmp/ckp_distributed_example_ddp1.pt"

def load_data() -> (DataLoader, DataLoader):
  # 1 Working with data
  training_data = datasets.FashionMNIST(
    root="~/.cache",
    train=True,
    download=True,
    transform=ToTensor(),
  )
  test_data = datasets.FashionMNIST(
    root="~/.cache",
    train=False,
    download=True,
    transform=ToTensor(),
  )
  ## For DDP, shuffle should be false
  train_dataloader: DataLoader = DataLoader(
    training_data,
    batch_size=batch_size,
    shuffle=False,
    sampler=DistributedSampler(training_data)
  )
  test_dataloader: DataLoader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
  )
  return train_dataloader, test_dataloader

class NerualNetword(nn.Module):
  def __init__(self):
    super(NerualNetword, self).__init__()
    self.flatten = nn.Flatten()
    self.layers = nn.Sequential(
      nn.Linear(28 * 28, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 10),
    )
  def forward(self, x):
    x = self.flatten(x)   # 将x维度修改为(batch_size, image_dim), image_dim是图像的长乘以宽
    logits = self.layers(x)
    return logits

def train(data: DataLoader, model: nn.Module, loss_fn: CrossEntropyLoss, optimizer: Optimizer, rank, epoch):
  size = len(data.dataset)
  data.sampler.set_epoch(epoch)
  for batch, (X, y) in enumerate(data):
    X: Tensor = X.to(rank)
    y: Tensor = y.to(rank)
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if batch % 200 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      print(f"loss={loss}, [{current}/{size}]")
  print(f"epoch final loss={loss.item()}")

def predict(data: DataLoader, ddp_model: nn.Module, loss_fn: CrossEntropyLoss, rank: int):
  size = len(data.dataset)
  num_batch = len(data)
  total_loss = 0
  correct = 0
  for X, y in data:
    X: Tensor = X.to(rank)
    y: Tensor = y.to(rank)
    with torch.no_grad():
      pred: Tensor = ddp_model(X)
      loss = loss_fn(pred, y)
      total_loss += loss.item()
      correct += (pred.argmax(dim = 1) == y).sum().item()
  print(f'predict mean loss={total_loss/num_batch}, correct = {correct/size}')

def setup(rank, world_size):
  os.environ["MASTER_ADDR"] = 'localhost'
  os.environ["MASTER_PORT"] = '12355'
  if torch.cuda.is_available():
    torch.cuda.set_device(rank)
  dist.init_process_group("nccl", rank=rank, world_size=world_size)

def save_model(ddp_model, epochs_run):
  snapshot = {}
  snapshot["MODEL_STATE"] = ddp_model.module.state_dict()
  snapshot["EPOCHS_RUN"] = epochs_run
  torch.save(snapshot, snapshot_path)
  print(f"Epoch {epochs_run} | Training snapshot saved at snapshot.pt")

def main(rank, world_size, checkpoint_enable, use_zero):
  # 1 setup
  setup(rank, world_size)

  # 2 load data
  train_data, test_data = load_data()

  # 3 define model
  epochs_run = 0
  model: NerualNetword = NerualNetword()
  # 2 load the model if necessary
  if checkpoint_enable and os.path.exists(snapshot_path):
    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot["MODEL_STATE"])
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {epochs_run}")
  model.to(rank)
  ddp_model = DDP(model, device_ids=[rank])

  # 4 define loss and optimer
  loss_fn: CrossEntropyLoss = nn.CrossEntropyLoss()
  if use_zero:
    optimizer: Optimizer = ZeroRedundancyOptimizer(ddp_model.parameters(), optimizer_class=torch.optim.Adam, lr=0.01)
  else:
    optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

  # 5 train and predict
  for e in range(epochs):
    train(train_data, ddp_model, loss_fn, optimizer, rank, epochs_run)
    epochs_run += 1
    if (rank == 0):
      predict(test_data, ddp_model, loss_fn, rank)
    if epochs_run % 2 == 0 and checkpoint_enable and rank == 0:
      save_model(ddp_model, epochs_run)

  # 6 save model
  if checkpoint_enable and rank == 0:
    save_model(ddp_model, epochs_run)

if __name__ == '__main__':
  world_size = 1
  checkpoint_enable = True
  use_zero = True
  mp.spawn(main, args=(world_size, checkpoint_enable, use_zero), nprocs=world_size)

# References:
# 1 https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html

