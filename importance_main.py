# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor


from tools.importance_training import train, train_batch_importance, test
from tools.dataset_wrapper import DatasetWithIndices


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(28*28, 512)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        logits = self.lin3(x)
        return logits


def train_batch(X, y, model, loss_fn, optimizer):
    model.train()
    X, y = X.to(device), y.to(device)
    pred = model(X)

    # Compute prediction error
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    torch.manual_seed(0)

    # Download training data from open datasets
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Uniform sampling
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    # Importance sampling
    from tools.conditions import RewrittenCondition

    batch_size_B = 5*batch_size
    train_dataloader_B = DataLoader(DatasetWithIndices(training_data), batch_size=batch_size_B, shuffle=True)

    scores = None
    b = batch_size
    B = batch_size_B
    tau_th = float(B + 3 * b) / (3 * b)
    momentum = 0.9  # TODO momentum should correspond with SGD
    condition = RewrittenCondition(tau_th, momentum)
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        for step in range(steps_in_epoch):
            # 1) sample batch
            # TODO: create some sampling object, which will check condition and sample either uniform or informed

            if condition.satisfied():
                if not condition.previously_satisfied():
                    print("Switching to importance sampling")
                # 1a.1) get weights from forward pass of B samples
                weights = train_batch_importance(train_dataloader_B, model, loss_fn, optimizer)  # TODO: with replacement or without?
                # 1a.2) create WeightedsRandomSampler()
                weighted_sampler = WeightedRandomSampler(weights, batch_size)
                # 1a.3) sample small batch of b samples to train on
                train_dataloader = DataLoader(training_data, sampler=weighted_sampler)
                X, y = weighted_sampler.sample()
            else:
                # 1b) sample batch uniformly
                X, y =
                pass

            X, y = next(iter(train_dataloader))

            # 2) train batch
            loss, scores = train_batch(X, y, model, loss_fn, optimizer)

            # 3) update sampler and sample updates condition
            condition.update(scores)  # TODO: I don't need the indices! DataLoader handles that automatically

            # print statistics
            if step % 100 == 0:
                loss, current = loss.item(), step * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{steps_in_epoch:>5d}]")

        test(test_dataloader, model, loss_fn)
    print("Done!")
