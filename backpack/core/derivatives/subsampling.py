"""Utility functions to implement mini-batch subsampling of BackPACK quantities."""


def subsample(tensor, subsampling=None):
    """Subsample along the batch dimension.

    Args:
        tensor (torch.Tensor): Arbitrary tensor whose leading axis is a batch dimension.
        subsampling ([int], None): Indices of samples that should be used from the
            current mini-batch. Use the entire mini-batch if ``None``.

    Returns:
        torch.Tensor: Tensor sliced along the batch dimension such that only the
            specified samples are contained.
    """
    if subsampling is None:
        return tensor
    else:
        return tensor[subsampling]


def subsample_input(module, subsampling=None):
    """Return input to the module with selected samples.

    Args:
        module (torch.nn.Module): Extended module. BackPACK's IO must have been stored
            in the module in a forward pass.
        subsampling ([int], None): Indices of samples that should be used from the
            current mini-batch. Use the entire mini-batch if ``None``.

    Returns:
        torch.Tensor: Input of the module, sliced along the batch dimension such that
            only the active samples are contained.
    """
    return subsample(module.input0, subsampling=subsampling)


def subsample_output(module, subsampling=None):
    """Return ouput of the module. Filter inactive samples if using subsampling.

    Args:
        module (torch.nn.Module): Extended module. BackPACK's IO must have been stored
            in the module in a forward pass.
        subsampling ([int], None): Indices of samples that should be used from the
            current mini-batch. Use the entire mini-batch if ``None``.

    Returns:
        torch.Tensor: Input of the module, sliced along the batch dimension such that
            only the active samples are contained.
    """
    return subsample(module.output, subsampling=subsampling)


def subsampled_shape(tensor, subsampling=None):
    """Return shape of sub-sampled tensor.

    Args:
        tensor (torch.Tensor): Arbitrary tensor whose leading axis is a batch dimension.
        subsampling ([int], None): Indices of samples that should be used from the
            current mini-batch. Use the entire mini-batch if ``None``.

    Returns:
        list: Shape of the sub-sampled tensor.
    """
    shape = list(tensor.shape)

    if subsampling is not None:
        N_axis = 0
        shape[N_axis] = len(subsampling)

    return shape


def subsampled_output_shape(module, subsampling=None):
    """Return shape of sub-sampled module output.

    Args:
        module (torch.nn.Module): Extended module. BackPACK's IO must have been stored
            in the module in a forward pass.
        subsampling ([int], None): Indices of samples that should be used from the
            current mini-batch. Use the entire mini-batch if ``None``.

    Returns:
        list: Shape of the sub-sampled module output.
    """
    return subsampled_shape(module.output, subsampling=subsampling)


def subsampled_input_shape(module, subsampling=None):
    """Return shape of sub-sampled module input.

    Args:
        module (torch.nn.Module): Extended module. BackPACK's IO must have been stored
            in the module in a forward pass.
        subsampling ([int], None): Indices of samples that should be used from the
            current mini-batch. Use the entire mini-batch if ``None``.

    Returns:
        list: Shape of the sub-sampled module output.
    """
    return subsampled_shape(module.input0, subsampling=subsampling)
