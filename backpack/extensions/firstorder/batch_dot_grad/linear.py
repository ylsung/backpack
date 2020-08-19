from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchDotGradLinear(FirstOrderModuleExtension):
    def __init__(self):
        self.derivatives = LinearDerivatives()
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        # Return value will be stored in savefield of extension
        # TODO: Replace dummy implementation
        print("Executing BatchDotGradLinear for bias (return dummy value of 42)")
        return 42

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        # Return value will be stored in savefield of extension
        # TODO: Replace dummy implementation
        print("Executing BatchDotGradLinear for weight (return dummy value of 42)")
        return 42
