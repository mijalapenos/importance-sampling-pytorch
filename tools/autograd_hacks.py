# https://github.com/cybertronai/autograd-hacks/blob/master/autograd_hacks.py
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

_supported_layers = ['Linear', 'Conv2d']  # Supported layer class types
_hooks_disabled: bool = False  # work-around for https://github.com/pytorch/pytorch/issues/25723
_enforce_fresh_backprop: bool = False  # global switch to catch double backprop errors on Hessian computation


def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return

    setattr(layer, "activations", input[0].detach())


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _enforce_fresh_backprop

    if _hooks_disabled:
        return

    if _enforce_fresh_backprop:
        assert not hasattr(layer, 'backprops_list'), "Seeing result of previous backprop, use clear_backprops(model) to clear"
        _enforce_fresh_backprop = False

    if not hasattr(layer, 'backprops_list'):
        setattr(layer, 'backprops_list', [])
    layer.backprops_list.append(output[0].detach())


def add_hooks(model: nn.Module, layer):
    """
    Adds hooks to model to save activations and backprop values.
    The hooks will
    1. save activations into param.activations during forward pass
    2. append backprops to params.backprops_list during backward pass.
    Call "remove_hooks(model)" to disable this.
    Args:
        model:
    """
    global _hooks_disabled
    _hooks_disabled = False

    assert layer.__class__.__name__ in _supported_layers, "Passed layer is not a supported layer"

    handles = []
    handles.append(layer.register_forward_hook(_capture_activations))
    handles.append(layer.register_backward_hook(_capture_backprops))

    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)


def remove_hooks(model: nn.Module, layer):
    """
    Remove hooks added by add_hooks(model)
    """

    assert model == 0, "not working, remove this after fix to https://github.com/pytorch/pytorch/issues/25723"

    if not hasattr(model, 'autograd_hacks_hooks'):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_hacks_hooks:
            handle.remove()
        del model.autograd_hacks_hooks


def compute_grad1(model: nn.Module, layer, loss_type: str = 'mean'):
    """
    Compute per-example gradients and save them under 'param.grad1'. Must be called after loss.backprop()
    Args:
        model:
        loss_type: either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
    """

    assert loss_type in ('sum', 'mean')
    layer_type = layer.__class__.__name__
    assert layer.__class__.__name__ in _supported_layers, "Passed layer is not a supported layer"

    assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
    assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
    assert len(
        layer.backprops_list) == 1, "Multiple backprops detected, make sure to call clear_backprops(model)"

    A = layer.activations
    n = A.shape[0]
    if loss_type == 'mean':
        B = layer.backprops_list[0] * n
    else:  # loss_type == 'sum':
        B = layer.backprops_list[0]

    if layer_type == 'Linear':
        setattr(layer.weight, 'grad1', torch.einsum('ni,nj->nij', B, A))
        if layer.bias is not None:
            setattr(layer.bias, 'grad1', B)

    elif layer_type == 'Conv2d':
        A = torch.nn.functional.unfold(A, layer.kernel_size)
        B = B.reshape(n, -1, A.shape[-1])
        grad1 = torch.einsum('ijk,ilk->ijl', B, A)
        shape = [n] + list(layer.weight.shape)
        setattr(layer.weight, 'grad1', grad1.reshape(shape))
        if layer.bias is not None:
            setattr(layer.bias, 'grad1', torch.sum(B, dim=2))