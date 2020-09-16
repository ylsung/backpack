"""Tests for `backpack.__init__.py`."""

import pytest
import torch

from backpack import backpack_deactivate_io, extend, is_io_active


def test_backpack_deactivate_io():
    """Check IO is not tracked."""
    torch.manual_seed(0)

    input = torch.rand(3, 5)
    module = torch.nn.Linear(5, 2)
    extend(module)

    with backpack_deactivate_io():
        _ = module(input)
        assert not hasattr(module, "input0")
        assert not hasattr(module, "output")

    _ = module(input)
    assert hasattr(module, "input0")
    assert hasattr(module, "output")


def test_backpack_deactivate_io_singleton():
    """Check that the context can only be used as a singleton."""
    with pytest.raises(RuntimeError):
        with backpack_deactivate_io():
            with backpack_deactivate_io():
                pass


def test_is_io_active():
    """Check whether IO tracking is disabled by ``backpack_deactivate_io``"""
    assert is_io_active()

    with backpack_deactivate_io():
        assert not is_io_active()

    assert is_io_active()
