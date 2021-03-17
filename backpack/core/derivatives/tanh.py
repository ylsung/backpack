from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.core.derivatives.subsampling import subsample_output


class TanhDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        """In general, `tanh''(x) ≠ 0`."""
        return False

    def df(self, module, g_inp, g_out, subsampling=None):
        """First tanh derivative: `tanh'(x) = 1 - tanh²(x)`."""
        output = subsample_output(module, subsampling=subsampling)

        return 1.0 - output ** 2

    def d2f(self, module, g_inp, g_out):
        """Second tanh derivative: `tanh''(x) = -2 / (1 - tanh²(x))`."""
        return -2.0 * module.output * (1.0 - module.output ** 2)
