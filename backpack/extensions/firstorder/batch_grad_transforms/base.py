import weakref

from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchGradTransformsBase(FirstOrderModuleExtension):
    def __init__(self, derivatives, params=None):
        self.derivatives = derivatives
        super().__init__(params=params)

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        batch_grad = self.derivatives.bias_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        batch_grad._param_weakref = weakref.ref(module.bias)

        return self.apply_transforms(ext, batch_grad)

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        batch_grad = self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        batch_grad._param_weakref = weakref.ref(module.weight)

        return self.apply_transforms(ext, batch_grad)

    def apply_transforms(self, ext, batch_grad):
        """Apply transformations to individual gradients. Return results as dict."""
        transforms = ext.get_transforms()
        results = {key: func(batch_grad) for key, func in transforms.items()}

        del batch_grad

        return results
