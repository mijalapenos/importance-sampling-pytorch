
def train_batch_importance(loader_with_indices, model, loss_fn, optimizer):
    model.train()
    X, y = X.to(device), y.to(device)

    # 1) freeze all layers with parameters except the last one
    # we are interested only in the backprop w.r.t. the last parametrized layer, this will stop gradient from being computed further back
    model.lin1.weight.requires_grad = False
    model.lin1.bias.requires_grad = False
    model.lin2.weight.requires_grad = False
    model.lin2.bias.requires_grad = False

    # 2) register hooks to get per sample gradient for the last layer
    from tools.autograd_hacks import add_hooks, compute_grad1
    add_hooks(model, model.lin3)
    pred = model(X)

    # Compute prediction error
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Compute per sample gradient
    compute_grad1(model, model.lin3)

    optimizer.step()  # TODO: this should be turned on for the bigger batch

    return None, None  # TODO: return scores


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
