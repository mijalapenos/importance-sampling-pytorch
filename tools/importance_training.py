import torch
from tools.autograd_hacks import add_hooks, remove_hooks, compute_grad1, clear_backprops
from torch import nn
from tools.dataset_wrapper import DatasetWithIndices
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler

from tools.conditions import RewrittenCondition


class ImportanceSamplingModule(nn.Module):
    def __init__(self):
        super(ImportanceSamplingModule, self).__init__()

    def freeze_all_but_last_trainable_layer(self):
        """Sets all trainable layers except the last one to layer_params.requires_grad = False"""
        raise NotImplementedError

    def unfreeze_all_trainable_layers(self):
        """Sets all trainable layers to layer_params.requires_grad = True"""
        raise NotImplementedError

    def get_last_trainable_layer(self):
        """Returns last trainable layer, should be conv or linear"""
        raise NotImplementedError

    def get_per_sample_grad(self):
        """Computes per sample gradient from grad1 parameter of the last trainable layer.
        Assumes compute_grad1() was called beforehand"""
        raise NotImplementedError


def write_stats(writer, epoch_idx, train_loss, val_acc, val_loss, importance_sampling=0):
    writer.add_scalar("Train/loss", train_loss, epoch_idx)
    writer.add_scalar("Train/importance", importance_sampling, epoch_idx)
    writer.add_scalar("Validation/accuracy", val_acc, epoch_idx)
    writer.add_scalar("Validation/loss", val_loss, epoch_idx)


def approximate_weights(model, loader_with_indices, loss_fn, optimizer, device):
    model.train()

    indices, X, y = next(iter(loader_with_indices))
    X, y = X.to(device), y.to(device)

    # 1) freeze all layers with parameters except the last one
    # we are interested only in the backprop w.r.t. the last parametrized layer,
    # this will stop gradient from being computed further back and therefore speed up things
    model.freeze_all_but_last_trainable_layer()

    # 2) register hooks to get per sample gradient for the last layer
    add_hooks(model, model.get_last_trainable_layer())
    pred = model(X)

    # Compute prediction error
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Compute per sample gradient
    compute_grad1(model, model.get_last_trainable_layer())
    remove_hooks(model)

    per_sample_grad = model.get_per_sample_grad()

    clear_backprops(model)

    # unfreeze trainable layers
    model.unfreeze_all_trainable_layers()

    # align weights with indices, set other to zero
    num_samples = len(loader_with_indices.dataset)
    scores = torch.zeros(num_samples)
    scores[indices] = per_sample_grad
    return scores


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return total_loss / len(dataloader)


def train_batch(X, y, model, loss_fn, optimizer, device):
    model.train()
    X, y = X.to(device), y.to(device)

    # register hooks to get per sample gradient for the last layer
    add_hooks(model, model.get_last_trainable_layer())

    pred = model(X)  # compute predictions
    loss = loss_fn(pred, y)  # compute prediction error

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    compute_grad1(model, model.get_last_trainable_layer())  # compute per sample gradient
    remove_hooks(model)

    scores = model.get_per_sample_grad()

    clear_backprops(model)

    return loss, scores


def test(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    acc = 100 * correct / size
    return acc, test_loss


def train_uniform(model, train_dataloader, test_dataloader, epochs, optim, sched=None, device='cpu'):
    print("Training with uniform sampling")
    loss_fn = nn.CrossEntropyLoss()

    initial_lr = optim.param_groups[0]['lr']
    writer = SummaryWriter(comment=f"_Uniform_Sched_lr={initial_lr}")

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        trn_loss = train(train_dataloader, model, loss_fn, optim, device)
        acc, val_loss = test(test_dataloader, model, loss_fn, device)
        if sched is not None:
            sched.step()
        print(f"Validation accuracy: {acc:.1f}%, avg loss: {val_loss:>8f}\n")
        write_stats(writer, t, trn_loss, acc, val_loss)
    print("Done!\n")


def train_importance(model, train_data, test_data, batch_size, epochs, optim, sched=None,
                     tau_th=None, large_bs=None, device='cpu'):
    """
    Hyperparameters are tau_th (threshold) and large batch size
    """
    print("Training with importance sampling")
    loss_fn = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    initial_lr = optim.param_groups[0]['lr']
    writer = SummaryWriter(comment=f"_Importance_Sched_lr={initial_lr}_tauth={tau_th:.1f}")

    large_bs = 8 * batch_size if large_bs is None else large_bs
    train_dataloader_B = DataLoader(DatasetWithIndices(train_data), batch_size=large_bs, shuffle=True)

    scores = None
    b = batch_size
    B = large_bs
    tau_th = float(B + 3 * b) / (3 * b) if tau_th is None else tau_th
    condition = RewrittenCondition(tau_th, 0.9)
    trn_examples = len(train_data)
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
                weights = approximate_weights(model, train_dataloader_B, loss_fn, optim, device)
                # 1a.2) create WeightedRandomSampler()
                weighted_sampler = WeightedRandomSampler(weights, large_bs)
                # weighted_batch_sampler = BatchSampler(weighted_sampler, batch_size, False)
                # 1a.3) sample small batch of b samples to train on
                train_dataloader_weighted = DataLoader(train_data, batch_size=batch_size, sampler=weighted_sampler)
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
