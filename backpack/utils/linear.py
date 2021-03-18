from torch import einsum

from backpack.core.derivatives.subsampling import subsample_input


def extract_weight_diagonal(module, backproped, sum_batch=True, subsampling=None):
    input = subsample_input(module, subsampling=subsampling)

    if sum_batch:
        equation = "vno,ni->oi"
    else:
        equation = "vno,ni->noi"

    return einsum(equation, (backproped ** 2, input ** 2))


def extract_bias_diagonal(module, backproped, sum_batch=True, subsampling=None):
    if sum_batch:
        equation = "vno->o"
    else:
        equation = "vno->no"
    return einsum(equation, backproped ** 2)
