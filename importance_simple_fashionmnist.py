# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from tools.importance_training import ImportanceSamplingModule, train_importance, train_uniform


class NeuralNetwork(ImportanceSamplingModule):
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
        x = self.lin3(x)
        return x

    def freeze_all_but_last_trainable_layer(self):
        self.lin1.weight.requires_grad = False
        self.lin1.bias.requires_grad = False
        self.lin2.weight.requires_grad = False
        self.lin2.bias.requires_grad = False

    def unfreeze_all_trainable_layers(self):
        self.lin1.weight.requires_grad = True
        self.lin1.bias.requires_grad = True
        self.lin2.weight.requires_grad = True
        self.lin2.bias.requires_grad = True

    def get_last_trainable_layer(self):
        # should be a conv or a linear layer
        return self.lin3

    def get_per_sample_grad(self):
        # assumes compute_grad1() was called beforehand
        per_sample_grad_weights = torch.abs(self.lin3.weight.grad1).sum((2, 1))
        per_sample_grad_bias = torch.abs(self.lin3.bias.grad1).sum(1)
        per_sample_grad = per_sample_grad_weights + per_sample_grad_bias
        return per_sample_grad


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

    batch_size = 128

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

    lr = 0.01
    epochs = 30

    model = NeuralNetwork().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[5, 15], gamma=0.2)
    # train_uniform(model, train_dataloader, test_dataloader, epochs, optim, sched, device)

    tau_th = 1.5
    while tau_th <= 2:
        model = NeuralNetwork().to(device)
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 20], gamma=0.2)
        train_importance(model, training_data, test_data, batch_size, epochs, optim, sched, tau_th, device=device)
        tau_th += 0.1
