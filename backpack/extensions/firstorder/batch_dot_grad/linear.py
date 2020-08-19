import torch

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchDotGradLinear(FirstOrderModuleExtension):
    def __init__(self):
        self.derivatives = LinearDerivatives()
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        # Return value will be stored in savefield of extension
        grad_batch = self.derivatives.bias_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        return self.pairwise_dot(grad_batch)

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        # Return value will be stored in savefield of extension
        grad_batch = self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        return self.pairwise_dot(grad_batch)

    @staticmethod
    def pairwise_dot(grad_batch):
        # flatten all feature dimensions
        grad_batch_flat = grad_batch.flatten(start_dim=1)
        # pairwise dot product
        return torch.einsum("if,jf->ij", grad_batch_flat, grad_batch_flat)
