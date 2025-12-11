import os

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import Optimizer
from torch.nn.modules import CrossEntropyLoss

batch_size = 64
epochs = 1

def load_data() -> (DataLoader, DataLoader):
    # 1 Working with data
    training_data = datasets.FashionMNIST(
        root="~/.cache",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="./cache",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    train_dataloader: DataLoader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader: DataLoader = DataLoader(test_data, batch_size=batch_size)
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

def train(data: DataLoader, model: nn.Module, loss_fn: CrossEntropyLoss , optimizer: Optimizer, device: str):
    size = len(data.dataset)
    for batch, (X, y) in enumerate(data):
        X: Tensor = X.to(device)
        y: Tensor = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 200 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss={loss}, [{current}/{size}]")
    print(f"epoch final loss={loss.item()}")

def predict(data: DataLoader, model: nn.Module, loss_fn: CrossEntropyLoss, device: str):
    size = len(data.dataset)
    num_batch = len(data)
    total_loss = 0
    correct = 0
    for X, y in data:
        X: Tensor = X.to(device)
        y: Tensor = y.to(device)
        with torch.no_grad():
            pred: Tensor = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct += (pred.argmax(dim = 1) == y).sum().item()
    print(f'predict mean loss={total_loss/num_batch}, correct = {correct/size}')

def main():
    # 1 load device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # 2 load data
    train_data, test_data = load_data()

    # 3 define model
    model: NerualNetword = NerualNetword().to(device)

    # 4 define loss and optimer
    loss_fn: CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 5 train and predict
    for epoch in range(epochs):
        train(train_data, model, loss_fn, optimizer, device)
        predict(test_data, model, loss_fn, device)

    # 6 save model
    # torch.save(model.state_dict(), "../../tmp/distributed_example_single.pth")

if __name__ == '__main__':
    os.environ['TORCH_SHOW_DISPATCH_TRACE'] = "true"
    main()