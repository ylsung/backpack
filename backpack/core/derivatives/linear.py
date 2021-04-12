from torch import einsum

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.core.derivatives.subsampling import subsample_input


class LinearDerivatives(BaseParameterDerivatives):
    """Partial derivatives for the Linear layer.

    Index conventions:
    ------------------
    * v: Free dimension
    * n: Batch dimension
    * o: Output dimension
    * i: Input dimension
    """

    def hessian_is_zero(self):
        return True

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat, subsampling=None):
        """Batch-apply transposed Jacobian of the output w.r.t. the input.

        Args:
            module (torch.nn.Linear): Linear layer.
            g_inp ((torch.Tensor)): Tuple of gradients w.r.t. layer inputs.
            g_out ((torch.Tensor)): Tuple of gradients w.r.t. layer outputs.
            mat (torch.Tensor): Batch of ``V`` vectors, shaped as the layer outputs
                (``[batch_size, *, out_features]``), onto which the transposed
                output-input Jacobian will be applied. Has shape
                ``[V, batch_size, *, out_features]``. If sub-sampling is used,
                the second axis must have dimension ``len(subsampling)``.
            subsampling ([int] or None): Indices of samples to be considered from
                the mini-batch. Default: ``None`` (uses all samples).

        Returns:
            torch.Tensor: Batched transposed Jacobian vector products. Has shape
                ``[V, batch_size, *, in_features]``. If sub-sampling is used,
                the second axis has dimension ``len(subsampling)``.
        """
        d_input = module.weight.data

        return einsum("oi,vn...o->vn...i", (d_input, mat))

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. the input."""
        d_input = module.weight.data
        return einsum("oi,vni->vno", (d_input, mat))

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        jac = module.weight.data
        return einsum("ik,ij,jl->kl", (jac, mat, jac))

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. the weight."""
        d_weight = module.input0
        return einsum("ni,voi->vno", (d_weight, mat))

    def _weight_jac_t_mat_prod(
        self, module, g_inp, g_out, mat, sum_batch=True, subsampling=None
    ):
        """Apply transposed Jacobian of the output w.r.t. the weight."""
        d_weight = subsample_input(module, subsampling)
        contract = "vno,ni->voi" if sum_batch else "vno,ni->vnoi"
        return einsum(contract, (mat, d_weight))

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """Apply Jacobian of the output w.r.t. the bias."""
        N = module.input0.size(0)
        return mat.unsqueeze(1).expand(-1, N, -1)

    def _bias_jac_t_mat_prod(
        self, module, g_inp, g_out, mat, sum_batch=True, subsampling=None
    ):
        """Apply transposed Jacobian of the output w.r.t. the bias."""
        if sum_batch:
            N_axis = 1
            return mat.sum(N_axis)
        else:
            return mat
