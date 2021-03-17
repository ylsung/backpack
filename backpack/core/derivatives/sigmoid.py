from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.core.derivatives.subsampling import subsample_output


class SigmoidDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        """`σ''(x) ≠ 0`."""
        return False

    def df(self, module, g_inp, g_out, subsampling=None):
        """First sigmoid derivative: `σ'(x) = σ(x) (1 - σ(x))`."""
        output = subsample_output(module, subsampling=subsampling)
        return output * (1.0 - output)

    def d2f(self, module, g_inp, g_out):
        """Second sigmoid derivative: `σ''(x) = σ(x) (1 - σ(x)) (1 - 2 σ(x))`."""
        return module.output * (1 - module.output) * (1 - 2 * module.output)
