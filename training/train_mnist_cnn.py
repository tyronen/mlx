import logging
import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

from common import utils, arguments
from models import CNN

args = arguments.get_args("Mnist CNN")


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logging.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():
    utils.setup_logging()
    logging.info("Downloading MNIST dataset...")

    utils.randomize()

    # Load original MNIST data
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # add mouse-like behaviour
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
    )

    root = os.environ.get("HF_DATASETS_CACHE")
    training_data = tqdm(
        datasets.MNIST(root=root, train=True, download=True, transform=transform)
    )

    test_data = tqdm(
        datasets.MNIST(root=root, train=False, download=True, transform=transform)
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = utils.get_device()
    logging.info(f"Using {device} device")
    model = CNN.CNN()
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = args.epochs
    for t in range(epochs):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

    torch.save(model.state_dict(), args.model_path)
    logging.info(f"Saved PyTorch model state to {args.model_path}")


if __name__ == "__main__":
    main()
