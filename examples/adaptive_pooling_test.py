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
import torchvision.models
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


IMAGENET_SHAPE = (1, 3, 224, 224)


def torchvision_to_sequential(vgg, verify=(IMAGENET_SHAPE,)):
    """Convert torchvision model to a pure sequential without overwriting forward.

    Tested on:
        - `torchvision.models.vgg11` (and with BN)
        - `torchvision.models.vgg13` (and with BN)
        - `torchvision.models.vgg16` (and with BN)
        - `torchvision.models.vgg19` (and with BN)
        - `torchvision.models.alexnet`

    Note:
    -----
    Works, if the model forward pass is
        x -> features -> avgpool -> flatten -> classifier

    Use `torch.nn.Flatten`.
    Verify identical forward pass for ImageNet-shaped inputs.
    """

    def convert(vgg):
        """ Naming conventions from:
        https://pytorch.org/docs/stable/_modules/torchvision/torchvision.models/vgg.html
        """
        features = list(vgg.features.children())
        avgpool = vgg.avgpool
        classifier = list(vgg.classifier.children())

        sequential = torch.nn.Sequential()
        sequential.features = torch.nn.Sequential(*features)
        sequential.avgpool = avgpool

        sequential.flatten = torch.nn.Flatten()
        sequential.classifier = torch.nn.Sequential(*classifier)
        return sequential

    sequential = convert(vgg)

    def verify_forward(model1, model2, shapes):
        def forward(model, x, seed=None):
            if seed is not None:
                torch.manual_seed(seed)
            return model(x)

        shapes = [] if shapes is None else shapes
        for shape in shapes:
            x = torch.rand(shape)
            seed = 42
            y1 = forward(model1, x, seed=seed)
            y2 = forward(model2, x, seed=seed)
            if not torch.allclose(y1, y2):
                raise ValueError("Error in conversion: Forward pass not identical.")

    verify_forward(vgg, sequential, shapes=verify)

    return sequential


def print_in_out_shapes_during_forward(module):
    old_forward = module.forward

    def new_forward(x):
        print(f"({module.__class__.__name__}) Input: {x.shape}")
        y = old_forward(x)
        print(f"({module.__class__.__name__}) Output: {y.shape}")
        return y

    module.forward = new_forward
    return module


##############################################################################

models = {
    "vgg11": torchvision.models.vgg11,
    "vgg11_bn": torchvision.models.vgg11_bn,
    "vgg13": torchvision.models.vgg13,
    "vgg13_bn": torchvision.models.vgg13_bn,
    "vgg16": torchvision.models.vgg16,
    "vgg16_bn": torchvision.models.vgg16_bn,
    "vgg19": torchvision.models.vgg19,
    "vgg19_bn": torchvision.models.vgg19_bn,
    "alexnet": torchvision.models.alexnet,
}

for name, load_model in models.items():
    print(name)
    model = load_model()
    # model.avgpool = print_in_out_shapes_during_forward(model.avgpool)
    model_seq = torchvision_to_sequential(model)
    print("Successful")

vgg_pool_in = (7, 7)
vgg_pool_out = (7, 7)
print(
    "(VGG) Replace Adaptive Pool by normal pool for Imagenet:",
    can_be_implemented_as_non_adaptive(vgg_pool_in, vgg_pool_out),
)

alex_pool_in = (6, 6)
alex_pool_out = (6, 6)
print(
    "(AlexNet) Replace Adaptive Pool by normal pool for Imagenet:",
    can_be_implemented_as_non_adaptive(alex_pool_in, alex_pool_out),
)
