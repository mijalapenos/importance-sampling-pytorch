# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from tools.importance_training import ImportanceSamplingModule, train, train_batch, approximate_weights, test, write_stats
from tools.dataset_wrapper import DatasetWithIndices


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
        logits = self.lin3(x)
        return logits

    def freeze_all_but_last_trainable_layer(self):
        self.lin1.weight.requires_grad = False
        self.lin1.bias.requires_grad = False
        self.lin2.weight.requires_grad = False
        self.lin2.bias.requires_grad = False

    def defreeze_all_trainable_layers(self):
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


def train_uniform(model, train_dataloader, test_dataloader, epochs, optim, sched=None):
    print("Training with uniform sampling")
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter(comment=f"_Uniform_Sched_lr={lr}")

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        trn_loss = train(train_dataloader, model, loss_fn, optim, device)
        acc, val_loss = test(test_dataloader, model, loss_fn, device)
        if sched is not None:
            sched.step()
        print(f"Validation accuracy: {acc:.1f}%, avg loss: {val_loss:>8f}\n")
        write_stats(writer, t, trn_loss, acc, val_loss)
    print("Done!\n")


def train_importance(model, train_dataloader, test_dataloader, epochs, optim, sched=None, tau_th=1.5):
    print("Training with importance sampling")
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter(comment=f"_Importance_Sched_lr={lr}_tauth={tau_th:.1f}")

    from tools.conditions import RewrittenCondition
    batch_size_B = 8 * batch_size  # TODO: how to select B according to the article?
    train_dataloader_B = DataLoader(DatasetWithIndices(training_data), batch_size=batch_size_B, shuffle=True)

    scores = None
    b = batch_size
    B = batch_size_B
    # tau_th = float(B + 3 * b) / (3 * b)
    # tau_th = 1.5  # used in the paper, they say it should work as well
    condition = RewrittenCondition(tau_th, 0.9)
    trn_examples = len(training_data)
    steps_in_epoch = trn_examples // batch_size
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_dataloader_iter = iter(train_dataloader)
        importance_sampling = 0
        total_trn_loss = 0
        for _ in tqdm(range(steps_in_epoch)):
            # 1) sample batch
            if condition.satisfied:
                importance_sampling += 1
                # 1a) sample batch based on importance
                # 1a.1) get weights from forward pass of B samples
                weights = approximate_weights(train_dataloader_B, model, loss_fn, optim, device)  # TODO: with replacement or without?
                # 1a.2) create WeightedsRandomSampler()
                weighted_sampler = WeightedRandomSampler(weights, batch_size_B)
                # weighted_batch_sampler = BatchSampler(weighted_sampler, batch_size, False)
                # 1a.3) sample small batch of b samples to train on
                train_dataloader_weighted = DataLoader(training_data, batch_size=batch_size, sampler=weighted_sampler)
                X, y = next(iter(train_dataloader_weighted))
            else:
                # 1b) sample batch uniformly
                # print("nope")
                X, y = next(train_dataloader_iter)

            # 2) train batch
            loss, scores = train_batch(X, y, model, loss_fn, optim, device)
            total_trn_loss += loss.item()

            # 3) update sampler and sample updates condition
            condition.update(scores)

            # print statistics
            # if step % 100 == 0:
            #     loss, current = loss.item(), step * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{steps_in_epoch:>5d}]")

        if sched is not None:
            sched.step()

        trn_loss = total_trn_loss / steps_in_epoch
        acc, loss = test(test_dataloader, model, loss_fn, device)
        print(f"Validation accuracy: {acc:.1f}%, avg loss: {loss:>8f}")
        if importance_sampling:
            print(f" > Ran importance sampling {importance_sampling} times")
        print()
        write_stats(writer, t, trn_loss, acc, loss, importance_sampling)
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
    # train_uniform(model, train_dataloader, test_dataloader, epochs, optim, sched)

    tau_th = 1.5
    while tau_th <= 2:
        model = NeuralNetwork().to(device)
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 20], gamma=0.2)
        train_importance(model, train_dataloader, test_dataloader, epochs, optim, sched, tau_th)
        tau_th += 0.1
