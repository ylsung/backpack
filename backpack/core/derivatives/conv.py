from numpy import prod
from torch import einsum
from torch.nn import Conv1d, Conv2d, Conv3d
from torch.nn.grad import _grad_input_padding
from torch.nn.functional import conv1d, conv2d, conv3d
from torch.nn.functional import conv_transpose1d, conv_transpose2d, conv_transpose3d

from einops import rearrange, repeat, reduce
from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils import conv as convUtils


class ConvNDDerivatives(BaseParameterDerivatives):
    def __init__(self, N, module, conv_func, conv_transpose_func):
        self.conv_dims = N
        self.module = module
        self.conv_func = conv_func
        self.conv_transpose_func = conv_transpose_func

    def hessian_is_zero(self):
        return True

    def get_unfolded_input(self, module):
        return convUtils.unfold_by_conv(module.input0, module)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = rearrange(mat, "v n ... -> (v n) ...")
        jmp_as_conv = self.conv_func(
            mat_as_conv,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        return self.reshape_like_output(jmp_as_conv, module)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = rearrange(mat, "v n ... -> (v n) ...")
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
        input_size = list(module.input0.size())
        input_size[0] = mat.size(0)

        grad_padding = _grad_input_padding(
            grad_output=mat,
            input_size=input_size,
            stride=module.stride,
            padding=module.padding,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
        )

        jac_t_mat = self.conv_transpose_func(
            input=mat,
            weight=module.weight,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            output_padding=grad_padding,
            groups=module.groups,
            dilation=module.dilation,
        )
        return jac_t_mat

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """mat has shape [V, C_out]"""
        # Expand batch dimension
        jac_mat = mat.unsqueeze(1)
        # Expand data dimensions
        for i in range(3, len(module.output.shape) + 1):
            jac_mat = jac_mat.unsqueeze(i)

        expand_shape = [-1, module.output.shape[0], -1, *module.output.shape[2:]]

        return jac_mat.expand(*expand_shape)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        if sum_batch:
            return reduce(mat, "v n c_out ... -> v c_out", reduction="sum")
        else:
            return reduce(mat, "v n c_out ... -> v n c_out", reduction="sum")

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        if module.groups != 1:
            raise NotImplementedError("Groups greater than 1 are not supported yet")

        jac_mat = rearrange(mat, "v n c_out ... -> v n (c_out ...)")
        X = self.get_unfolded_input(module)
        jac_mat = einsum("nij,vki->vnkj", X, jac_mat)
        return self.reshape_like_output(jac_mat, module)

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        if module.groups != 1:
            raise NotImplementedError("Groups greater than 1 are not supported yet")

        V = mat.shape[0]
        N, C_out = module.output.shape[0], module.output.shape[1]
        C_in = module.input0.shape[1]

        mat = repeat(
            mat, "v n c_out ... -> v n (repeat_c_in c_out) ...", repeat_c_in=C_in
        )
        mat = repeat(mat, "v n c_in_c_out ... -> (v n c_in_c_out) dummy ...", dummy=1)

        input = repeat(
            module.input0, "n c ... -> dummy (repeat n c) ...", dummy=1, repeat=V
        )

        grad_weight = self.conv_func(
            input,
            mat,
            bias=None,
            stride=module.dilation,
            padding=module.padding,
            dilation=module.stride,
            groups=C_in * N * V,
        ).squeeze(0)

        for dim in range(self.conv_dims):
            axis = dim + 1
            size = module.weight.shape[2 + dim]
            grad_weight = grad_weight.narrow(axis, 0, size)

        if sum_batch:
            return reduce(
                grad_weight,
                "(v n C_in C_out) ... -> v C_out C_in ...",
                reduction="sum",
                v=V,
                n=N,
                C_in=C_in,
                C_out=C_out,
            )
        else:
            return rearrange(
                grad_weight,
                "(v n C_in C_out) ... -> v n C_out C_in ...",
                v=V,
                n=N,
                C_in=C_in,
                C_out=C_out,
            )

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        in_features = int(prod(module.input0.size()[1:]))
        out_features = int(prod(module.output.size()[1:]))

        mat = mat.reshape(out_features, *module.output.size()[1:])
        jac_t_mat = self.__jac_t(module, mat).reshape(out_features, in_features)

        mat_t_jac = jac_t_mat.t().reshape(in_features, *module.output.size()[1:])
        jac_t_mat_t_jac = self.__jac_t(module, mat_t_jac)
        jac_t_mat_t_jac = jac_t_mat_t_jac.reshape(in_features, in_features)

        return jac_t_mat_t_jac.t()


class Conv1DDerivatives(ConvNDDerivatives):
    def __init__(self):
        super().__init__(
            N=1, module=Conv1d, conv_func=conv1d, conv_transpose_func=conv_transpose1d
        )


class Conv2DDerivatives(ConvNDDerivatives):
    def __init__(self):
        super().__init__(
            N=2, module=Conv2d, conv_func=conv2d, conv_transpose_func=conv_transpose2d
        )


class Conv3DDerivatives(ConvNDDerivatives):
    def __init__(self):
        super().__init__(
            N=3, module=Conv3d, conv_func=conv3d, conv_transpose_func=conv_transpose3d
        )
