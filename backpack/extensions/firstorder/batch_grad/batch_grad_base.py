from backpack.extensions.firstorder.base import FirstOrderModuleExtension


class BatchGradBase(FirstOrderModuleExtension):
    def __init__(self, derivatives, params=None):
        self.derivatives = derivatives
        super().__init__(params=params)

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        """Apply transposed Jacobian w.r.t. the bias to gradient w.r.t. the output.

        Sub-sample relevant samples from ``g_out`` if sub-sampling is enabled.

        Returns:
            torch.Tensor: Individual gradients w.r.t the module bias.
        """
        subsampling = ext.get_subsampling()

        return self.derivatives.bias_jac_t_mat_prod(
            module,
            g_inp,
            g_out,
            self.subsample(g_out[0], subsampling),
            sum_batch=False,
            subsampling=subsampling,
        )

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        """Apply transposed Jacobian w.r.t. the weight to gradient w.r.t. the output.

        Subsample relevant samples from ``g_out`` if ``subsampling`` is enabled.

        Returns:
            torch.Tensor: Individual gradients w.r.t the module weight.
        """
        subsampling = ext.get_subsampling()

        return self.derivatives.weight_jac_t_mat_prod(
            module,
            g_inp,
            g_out,
            self.subsample(g_out[0], subsampling),
            sum_batch=False,
            subsampling=subsampling,
        )

    @staticmethod
    def subsample(tensor, subsampling):
        """Slice out relevant samples along the batch dimension for sub-sampling.

        Args:
            tensor (torch.Tensor): An arbitrary tensor whose leading dimension is
                the leading axis.
            subsampling ([int] or None): List of indices specifying the sample subset.
                If ``None``, all samples are considered relevant.

        Returns:
            torch.Tensor: Sub-sampled slice of ``tensor`` along the batch dimension.
        """
        if subsampling is None:
            return tensor
        else:
            return tensor[subsampling]
