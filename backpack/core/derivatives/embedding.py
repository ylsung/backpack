import torch
from torch import einsum
from torch.nn import Embedding

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class EmbeddingDerivatives(BaseParameterDerivatives):
    """Partial derivatives for the Linear layer.

    Index conventions:
    ------------------
    * v: Free dimension
    * n: Batch dimension
    * o: Output dimension
    * i: Input dimension
    """

    def get_module(self):
        return Embedding

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply transposed Jacobian of the output w.r.t. the weight."""
        d_weight = module.input0

        if len(mat.shape) == 4:
            jac = torch.zeros([d_weight.shape[0]] + list(module.weight.shape), device=d_weight.device)
        
        # print(module, d_weight.shape)
        for i in range(d_weight.shape[0]):
            for j in range(d_weight.shape[1]):
                idx = d_weight[i, j]

                if idx != module.padding_idx:
                    jac[i, idx] += mat[0, i, j]

        # print(jac.shape)

        jac = jac.unsqueeze(0)

        return jac


