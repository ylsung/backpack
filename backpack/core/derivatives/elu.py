from torch import exp, gt

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.core.derivatives.subsampling import subsample_input


class ELUDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        """`ELU''(x) ≠ 0`."""
        return False

    def df(self, module, g_inp, g_out, subsampling=None):
        """First ELU derivative: `ELU'(x) = alpha * e^x if x < 0 else 1`. """
        input = subsample_input(module, subsampling=subsampling)

        df_ELU = gt(input, 0).float()
        idx_zero = df_ELU == 0
        df_ELU[idx_zero] = module.alpha * exp(input[idx_zero])

        return df_ELU

    def d2f(self, module, g_inp, g_out):
        """Second ELU derivative: `ELU''(x) = alpha * e^x if x < 0 else 1`. """
        return self.df(module, g_inp, g_out)
