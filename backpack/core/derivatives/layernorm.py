from warnings import warn

from torch import einsum
from torch.nn import LayerNorm

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class LayerNormDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return LayerNorm

    def hessian_is_zero(self):
        return False

    def hessian_is_diagonal(self):
        return False

    def get_normalized_input_and_var(self, module):
        input = module.input0

        org_size = input.shape
        reshape_size = input.shape[:-len(module.normalized_shape)]
        reshape_size = [*reshape_size, -1]

        input = input.reshape(reshape_size)
        mean = input.mean(dim=-1)
        var = input.var(dim=-1, unbiased=False)

        out = (input - mean.unsqueeze(-1)) / (var.unsqueeze(-1) + module.eps).sqrt()
        return out, var

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch):
        if not sum_batch:
            warn(
                "BatchNorm batch summation disabled."
                "This may not compute meaningful quantities"
            )
        x_hat, _ = self.get_normalized_input_and_var(module)
        
        if len(x_hat.shape) == 3:
            equation = "vnpi,npi->v{}i".format("" if sum_batch is True else "n")
        else:
            equation = "vni,ni->v{}i".format("" if sum_batch is True else "n")

        operands = [mat, x_hat]
        return einsum(equation, operands)

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        N = module.input0.size(0)
        return mat.unsqueeze(1).repeat(1, N, 1)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):

        if len(mat.shape) == 4:
            if not sum_batch:
                warn(
                    "BatchNorm batch summation disabled."
                    "This may not compute meaningful quantities"
                )
                return mat.sum(-2)
            else:
                N_axis = 1
                return mat.sum(-2).sum(N_axis)

        if not sum_batch:
            warn(
                "BatchNorm batch summation disabled."
                "This may not compute meaningful quantities"
            )
            return mat
        else:
            N_axis = 1
            return mat.sum(N_axis)
