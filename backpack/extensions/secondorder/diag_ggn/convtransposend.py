from backpack.core.derivatives.subsampling import subsample_input
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.utils import conv_transpose as convUtils


class DiagGGNConvTransposeND(DiagGGNBaseModule):
    def __init__(self, derivatives, N, params=None):
        super().__init__(derivatives=derivatives, params=params)
        self.N = N

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped

        return convUtils.extract_bias_diagonal(module, sqrt_ggn, self.N, sum_batch=True)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        subsampling = ext.get_subsampling()
        input = subsample_input(module, subsampling=subsampling)

        X = convUtils.unfold_by_conv_transpose(input, module)
        weight_diag = convUtils.extract_weight_diagonal(
            module, X, backproped, self.N, sum_batch=True
        )
        return weight_diag


class BatchDiagGGNConvTransposeND(DiagGGNBaseModule):
    def __init__(self, derivatives, N, params=None):
        super().__init__(derivatives=derivatives, params=params)
        self.N = N

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped
        return convUtils.extract_bias_diagonal(
            module, sqrt_ggn, self.N, sum_batch=False
        )

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        subsampling = ext.get_subsampling()
        input = subsample_input(module, subsampling=subsampling)

        X = convUtils.unfold_by_conv_transpose(input, module)
        weight_diag = convUtils.extract_weight_diagonal(
            module, X, backproped, self.N, sum_batch=False
        )
        return weight_diag
