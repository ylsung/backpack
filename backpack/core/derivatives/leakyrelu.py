from torch import gt

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.core.derivatives.subsampling import subsample_input


class LeakyReLUDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        """`LeakyReLU''(x) = 0`."""
        return True

    def df(self, module, g_inp, g_out, subsampling=None):
        """First LeakyReLU derivative:
        `LeakyReLU'(x) = negative_slope if x < 0 else 1`."""
        input = subsample_input(module, subsampling=subsampling)

        df_leakyrelu = gt(input, 0).float()
        df_leakyrelu[df_leakyrelu == 0] = module.negative_slope

        return df_leakyrelu
