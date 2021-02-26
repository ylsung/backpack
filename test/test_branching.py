"""Test branching with modules."""

# TODO Integrate into extensions/secondorder/diag_ggn after making branching work

from test.automated_test import check_sizes_and_values
from test.core.derivatives.utils import classification_targets

import pytest
import torch
from torch.nn import CrossEntropyLoss, Identity, Linear, ReLU, Sequential, Sigmoid
from torch.nn.utils.convert_parameters import parameters_to_vector

from backpack import backpack, extend, extensions
from backpack.branching import ActiveIdentity, Branch, Merge
from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list


def set_up(apply_extend=False, active_identity=True):
    """Set seed. Generate and return inputs, labels, model and loss function.

    Args:
        active_identity (bool): Whether the identity function should create a new node
            in the computation graph.
        apply_extend (bool): Whether model and loss function should be extended.
    """
    torch.manual_seed(0)

    N = 7

    in_features = 20
    hidden_features = 10
    out_features = 3

    X = torch.rand((N, in_features))
    y = classification_targets((N,), out_features)

    identity = ActiveIdentity() if active_identity else Identity()

    model = Sequential(
        Linear(in_features, hidden_features),
        ReLU(),
        Branch(
            identity,
            Linear(hidden_features, hidden_features),
        ),
        Merge(),
        Sigmoid(),
        Linear(hidden_features, out_features),
    )
    loss_function = CrossEntropyLoss(reduction="mean")

    if apply_extend:
        model = extend(model)
        loss_function = extend(loss_function)

    return X, y, model, loss_function


def autograd_diag_ggn_exact(X, y, model, loss_function):
    """Compute the generalized Gauss-Newton diagonal via autodiff."""
    D = sum(p.numel() for p in model.parameters())

    outputs = model(X)
    loss = loss_function(outputs, y)

    ggn_diag = torch.zeros(D)

    # compute GGN columns by GGNVPs with one-hot vectors
    for d in range(D):
        e_d = torch.zeros(D)
        e_d[d] = 1.0
        e_d_list = vector_to_parameter_list(e_d, model.parameters())

        ggn_d_list = ggn_vector_product(loss, outputs, model, e_d_list)

        ggn_diag[d] = parameters_to_vector(ggn_d_list)[d]

    return ggn_diag


def backpack_diag_ggn_exact(X, y, model, loss_function):
    """Compute the generalized Gauss-Newton diagonal via BackPACK."""
    outputs = model(X)
    loss = loss_function(outputs, y)

    with backpack(extensions.DiagGGNExact()):
        loss.backward()

    return torch.cat([p.diag_ggn_exact.flatten() for p in model.parameters()])


def test_diag_ggn_exact_active_identity():
    """Compare diagonal GGN of a ResNet."""
    X, y, model, loss_function = set_up()

    autograd_result = autograd_diag_ggn_exact(X, y, model, loss_function)

    X, y, model, loss_function = set_up(apply_extend=True)

    backpack_result = backpack_diag_ggn_exact(X, y, model, loss_function)

    check_sizes_and_values(autograd_result, backpack_result)


def test_diag_ggn_exact_nn_Identity_fails():
    """``torch.nn.Identity`` does not create a node and messes up backward hooks."""
    X, y, model, loss_function = set_up(active_identity=False)

    autograd_result = autograd_diag_ggn_exact(X, y, model, loss_function)

    X, y, model, loss_function = set_up(apply_extend=True, active_identity=False)

    with pytest.raises(AttributeError):
        backpack_result = backpack_diag_ggn_exact(X, y, model, loss_function)

        check_sizes_and_values(autograd_result, backpack_result)
