from torch import exp

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.core.derivatives.subsampling import subsample_input


class LogSigmoidDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        """`logsigmoid''(x) ≠ 0`."""
        return False

    def df(self, module, g_inp, g_out, subsampling=None):
        """First Logsigmoid derivative: `logsigmoid'(x) = 1 / (e^x + 1) `."""
        input = subsample_input(module, subsampling=subsampling)

        return 1 / (exp(input) + 1)

    def d2f(self, module, g_inp, g_out):
        """Second Logsigmoid derivative: `logsigmoid''(x) = - e^x / (e^x + 1)^2`."""
        exp_input = exp(module.input0)
        return -(exp_input / (exp_input + 1) ** 2)
