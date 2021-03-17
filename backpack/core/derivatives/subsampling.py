"""Utility functions to implement mini-batch subsampling of BackPACK quantities."""


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
    input = module.input0

    if subsampling is not None:
        input = input[subsampling]

    return input


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
    output = module.output

    if subsampling is not None:
        output = output[subsampling]

    return output
