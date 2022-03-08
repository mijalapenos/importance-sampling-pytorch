import torch
from tools.autograd_hacks import add_hooks, remove_hooks, compute_grad1, clear_backprops
from tqdm import tqdm


def approximate_weights(loader_with_indices, model, loss_fn, optimizer, device):
    model.train()

    indices, X, y = next(iter(loader_with_indices))
    X, y = X.to(device), y.to(device)

    # 1) freeze all layers with parameters except the last one
    # we are interested only in the backprop w.r.t. the last parametrized layer,
    # this will stop gradient from being computed further back and therefore speed up things
    model.lin1.weight.requires_grad = False
    model.lin1.bias.requires_grad = False
    model.lin2.weight.requires_grad = False
    model.lin2.bias.requires_grad = False
    # TODO: would be nice to automatize the freezing of the layers

    # 2) register hooks to get per sample gradient for the last layer
    add_hooks(model, model.lin3)
    pred = model(X)

    # Compute prediction error
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Compute per sample gradient
    compute_grad1(model, model.lin3)
    remove_hooks(model)

    per_sample_grad_weights = torch.abs(model.lin3.weight.grad1).sum(axis=2).sum(axis=1)
    per_sample_grad_bias = torch.abs(model.lin3.bias.grad1).sum(axis=1)
    per_sample_grad = per_sample_grad_weights + per_sample_grad_bias

    clear_backprops(model)

    # align weights with indices, set other to zero
    num_samples = len(loader_with_indices.dataset)
    scores = torch.zeros(num_samples)
    scores[indices] = per_sample_grad
    return scores


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train_batch(X, y, model, loss_fn, optimizer, device):
    model.train()
    X, y = X.to(device), y.to(device)

    # register hooks to get per sample gradient for the last layer
    add_hooks(model, model.lin3)

    pred = model(X)  # compute predictions
    loss = loss_fn(pred, y)  # compute prediction error
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    compute_grad1(model, model.lin3)  # compute per sample gradient
    remove_hooks(model)

    per_sample_grad_weights = torch.abs(model.lin3.weight.grad1).sum(axis=2).sum(axis=1)
    per_sample_grad_bias = torch.abs(model.lin3.bias.grad1).sum(axis=1)
    scores = per_sample_grad_weights + per_sample_grad_bias

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
    print(f"Test Error: \n Accuracy: {acc:.1f}%, Avg loss: {test_loss:>8f} \n")
    return acc, test_loss
