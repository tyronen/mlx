import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from models import MnistCNN, MODEL_PATH


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():
    print("Downloading MNIST dataset...")
    raw_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    stats_dataloader = DataLoader(
        raw_data, batch_size=len(raw_data.data), shuffle=False
    )
    images, _ = next(iter(stats_dataloader))
    mean = images.mean()
    std = images.std()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )

    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = MnistCNN()
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

    model_dict = {
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
    }
    torch.save(model_dict, MODEL_PATH)
    print(f"Saved PyTorch Model State to {MODEL_PATH}")


if __name__ == "__main__":
    main()
