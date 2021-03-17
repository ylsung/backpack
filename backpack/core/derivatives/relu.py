from torch import gt

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.core.derivatives.subsampling import subsample_input


class ReLUDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        """`ReLU''(x) = 0`."""
        return True

    def df(self, module, g_inp, g_out, subsampling=None):
        """First ReLU derivative: `ReLU'(x) = 0 if x < 0 else 1`. """
        input = subsample_input(module, subsampling=subsampling)

        return gt(input, 0).float()
