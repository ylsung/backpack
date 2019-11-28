"""Understanding how the pooling parameters are computed in AdaptiveAvgPool2d.

Illustration for a simple example and why it not always coincides with AvgPool2d:
https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work

Based on this question:
https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work

Torch implementation:
https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AdaptiveAveragePooling.cpp

Torch discussion mentioning AdaptiveAvgPool2d:
https://discuss.pytorch.org/t/adaptive-avg-pool2d-vs-avg-pool2d/27011/3
"""

import random

import torch
from torch.nn import AdaptiveAvgPool1d, AdaptiveAvgPool2d, AvgPool1d, AvgPool2d


def can_be_implemented_as_non_adaptive(input_size, output_size):
    """Is there an AvgPool module with same forward pass as AdaptiveAvgPool."""

    def convert_pool_params(input_size, output_size):
        """Convert fixed pooling parameters from adaptive pooling parameters."""
        assert len(input_size) == len(output_size)

        def convert(in_size, out_size):
            padding = 0
            stride = in_size // out_size
            kernel = in_size - (out_size - 1) * stride
            return kernel, stride, padding

        kernel_size, stride, padding = [], [], []
        for in_size, out_size in zip(input_size, output_size):
            k, s, p = convert(in_size, out_size)
            kernel_size.append(k)
            stride.append(s)
            padding.append(p)

        return kernel_size, stride, padding

    def compare(B, C, input_size, output_size):
        if len(input_size) == 1:
            ada_cls = AdaptiveAvgPool1d
            avg_cls = AvgPool1d
        elif len(input_size) == 2:
            ada_cls = AdaptiveAvgPool2d
            avg_cls = AvgPool2d
            pass
        else:
            raise ValueError

        ada_pool = ada_cls(output_size)
        kernel_size, stride, padding = convert_pool_params(input_size, output_size)
        avg_pool = avg_cls(kernel_size, stride=stride, padding=padding)

        batch, channels = 1, 1
        shape = (batch, channels, *input_size)
        x = torch.rand((B, C, *input_size)).float()

        y1 = ada_pool(x)
        y2 = avg_pool(x)

        same_shape = y1.shape == y2.shape
        # assert same_shape
        same_vals = torch.allclose(y1, y2, atol=1e-6)
        # assert same_vals
        return same_shape and same_vals

    return compare(1, 1, input_size, output_size)


def random_params():
    input_size = (random.randint(10, 20), random.randint(10, 20))
    output_size = (random.randint(1, 10), random.randint(1, 10))
    return input_size, output_size


RUNS = 100
for _ in range(RUNS):
    input_size, output_size = random_params()
    implementable = can_be_implemented_as_non_adaptive(input_size, output_size)
    print(f"in: {input_size}, out: {output_size} -> {implementable}")
