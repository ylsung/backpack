"""Base class for extension ``BatchDotGrad``."""

import torch

from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchDotGradBase(FirstOrderModuleExtension):
    def __init__(self, derivatives, params=None):
        self.derivatives = derivatives
        super().__init__(params=params)

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        grad_batch = self.derivatives.bias_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        return self.pairwise_dot(grad_batch)

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        grad_batch = self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        return self.pairwise_dot(grad_batch)

    @staticmethod
    def pairwise_dot(grad_batch):
        """Compute pairwise dot products of individual gradients."""
        # flatten all feature dimensions
        grad_batch_flat = grad_batch.flatten(start_dim=1)
        # pairwise dot product
        return torch.einsum("if,jf->ij", grad_batch_flat, grad_batch_flat)
