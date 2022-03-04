# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor


from tools.importance_training import train, train_batch, approximate_weights, test
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


def train_uniform(train_dataloader, test_dataloader):
    model = NeuralNetwork().to(device)
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")


def train_importance(train_dataloader, test_dataloader):
    from tools.conditions import RewrittenCondition
    model = NeuralNetwork().to(device)
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    batch_size_B = 5 * batch_size  # TODO: how to select B according to the article?
    train_dataloader_B = DataLoader(DatasetWithIndices(training_data), batch_size=batch_size_B, shuffle=True)

    scores = None
    b = batch_size
    B = batch_size_B
    tau_th = float(B + 3 * b) / (3 * b)
    condition = RewrittenCondition(tau_th, momentum)
    trn_examples = len(training_data)
    steps_in_epoch = trn_examples // batch_size
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_dataloader_iter = iter(train_dataloader)
        for step in range(steps_in_epoch):
            # 1) sample batch
            if condition.satisfied:
                # 1a) sample batch based on importance
                if not condition.previously_satisfied():
                    pass
                print("Switching to importance sampling\n")
                # 1a.1) get weights from forward pass of B samples
                weights = approximate_weights(train_dataloader_B, model, loss_fn, optimizer, device)  # TODO: with replacement or without?
                # 1a.2) create WeightedsRandomSampler()
                weighted_sampler = WeightedRandomSampler(weights, batch_size)
                # 1a.3) sample small batch of b samples to train on
                train_dataloader_weighted = DataLoader(training_data, sampler=weighted_sampler)
                train_dataloader_iter = iter(train_dataloader_weighted)
            else:
                # 1b) sample batch uniformly
                # print("nope")
                pass

            X, y = next(train_dataloader_iter)

            # 2) train batch
            loss, scores = train_batch(X, y, model, loss_fn, optimizer, device)  # TODO: what are scores?

            # 3) update sampler and sample updates condition
            condition.update(scores)

            # print statistics
            if step % 100 == 0:
                loss, current = loss.item(), step * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{steps_in_epoch:>5d}]")

        test(test_dataloader, model, loss_fn, device)
    print("Done!")


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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device} device")

    # train_uniform(train_dataloader, test_dataloader)

    train_importance(train_dataloader, test_dataloader)
