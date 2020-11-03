from torch import einsum
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn.functional import conv2d, conv3d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils import conv as convUtils
from backpack.utils.ein import eingroup


class Conv2DDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return Conv2d

    def hessian_is_zero(self):
        return True

    def get_unfolded_input(self, module):
        return convUtils.unfold_func(module)(module.input0)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        _, C_in, H_in, W_in = module.input0.size()
        in_features = C_in * H_in * W_in
        _, C_out, H_out, W_out = module.output.size()
        out_features = C_out * H_out * W_out

        mat = mat.reshape(out_features, C_out, H_out, W_out)
        jac_t_mat = self.__jac_t(module, mat).reshape(out_features, in_features)

        mat_t_jac = jac_t_mat.t().reshape(in_features, C_out, H_out, W_out)
        jac_t_mat_t_jac = self.__jac_t(module, mat_t_jac).reshape(
            in_features, in_features
        )

        return jac_t_mat_t_jac.t()

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,h,w->vn,c,h,w", mat)
        jmp_as_conv = conv2d(
            mat_as_conv,
            module.weight.data,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        return self.reshape_like_output(jmp_as_conv, module)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        mat_as_conv = eingroup("v,n,c,h,w->vn,c,h,w", mat)
        jmp_as_conv = self.__jac_t(module, mat_as_conv)
        return self.reshape_like_input(jmp_as_conv, module)

    def __jac_t(self, module, mat):
        """Apply Conv2d backward operation."""
        _, C_in, H_in, W_in = module.input0.size()
        _, C_out, H_out, W_out = module.output.size()
        H_axis = 2
        W_axis = 3

        conv2d_t = ConvTranspose2d(
            in_channels=C_out,
            out_channels=C_in,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=False,
            dilation=module.dilation,
            groups=module.groups,
        ).to(module.input0.device)

        conv2d_t.weight.data = module.weight

        V_N = mat.size(0)
        output_size = (V_N, C_in, H_in, W_in)

        jac_t_mat = (
            conv2d_t(mat, output_size=output_size)
            .narrow(H_axis, 0, H_in)
            .narrow(W_axis, 0, W_in)
        )
        return jac_t_mat

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        """mat has shape [V, C_out]"""
        # expand for each batch and for each channel
        N_axis, H_axis, W_axis = 1, 3, 4
        jac_mat = mat.unsqueeze(N_axis).unsqueeze(H_axis).unsqueeze(W_axis)

        N, _, H_out, W_out = module.output_shape
        return jac_mat.expand(-1, N, -1, H_out, W_out)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        N_axis, H_axis, W_axis = 1, 3, 4
        axes = [H_axis, W_axis]
        if sum_batch:
            axes = [N_axis] + axes

        return mat.sum(axes)

    # TODO: Improve performance by using conv instead of unfold

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        jac_mat = eingroup("v,o,i,h,w->v,o,ihw", mat)
        X = self.get_unfolded_input(module)

        jac_mat = einsum("nij,vki->vnkj", (X, jac_mat))
        return self.reshape_like_output(jac_mat, module)

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        """Apply output-weight Jacobian."""
        USE_HIGHER = True

        if USE_HIGHER:
            return self.__weight_jac_t_mat_prod_higher_conv(
                module, g_inp, g_out, mat, sum_batch=sum_batch
            )
        else:
            return self.__weight_jac_t_mat_prod_same_conv(
                module, g_inp, g_out, mat, sum_batch=sum_batch
            )

    def __weight_jac_t_mat_prod_higher_conv(
        self, module, g_inp, g_out, mat, sum_batch=True
    ):
        """Requires higher-order convolution.

        The algorithm is proposed in:
            - Rochette, G., Manoel, A., & Tramel, E. W., Efficient per-example
              gradient computations in convolutional neural networks (2019).
        """
        # Get some useful sizes
        V = mat.shape[0]
        C_in = module.in_channels
        C_out = module.out_channels
        G = module.groups
        N, _, H_in, W_in = module.input0.size()

        # Reshape to extract groups from the convolutional layer
        # Channels are seen as an extra spatial dimension with kernel size 1
        input_conv = module.input0.view(1, N * G, C_in // G, H_in, W_in).repeat(
            1, V, 1, 1, 1
        )

        # Compute convolution between input and output; the batchsize is seen
        # as channels, taking advantage of the `groups` argument
        mat_conv = eingroup("v,n,c,h,w->vnc,h,w", mat).unsqueeze(1).unsqueeze(2)

        stride = (1, *module.stride)
        dilation = (1, *module.dilation)
        padding = (0, *module.padding)

        conv = conv3d(
            input_conv,
            mat_conv,
            groups=V * N * G,
            stride=dilation,
            dilation=stride,
            padding=padding,
        ).squeeze(0)

        # Because of rounding shapes when using non-default stride or dilation,
        # convolution result must be truncated to convolution kernel size
        K_H_axis, K_W_axis = 2, 3
        K_H, K_W = module.kernel_size
        conv = conv.narrow(K_H_axis, 0, K_H).narrow(K_W_axis, 0, K_W)

        new_shape = [V, N, C_out, C_in // G, K_H, K_W]
        weight_grad = conv.view(*new_shape)

        if sum_batch:
            weight_grad = weight_grad.sum(1)

        return weight_grad

    def __weight_jac_t_mat_prod_same_conv(
        self, module, g_inp, g_out, mat, sum_batch=True
    ):
        """Requires same-order convolution, but uses more memory."""
        V = mat.shape[0]
        N, C_out, _, _ = module.output_shape
        _, C_in, _, _ = module.input0_shape

        mat = eingroup("v,n,c,w,h->vn,c,w,h", mat).repeat(1, C_in, 1, 1)
        C_in_axis = 1
        # a,b represent the combined/repeated dimensions
        mat = eingroup("a,b,w,h->ab,w,h", mat).unsqueeze(C_in_axis)

        N_axis = 0
        input = eingroup("n,c,h,w->nc,h,w", module.input0).unsqueeze(N_axis)
        input = input.repeat(1, V, 1, 1)

        grad_weight = conv2d(
            input,
            mat,
            bias=None,
            stride=module.dilation,
            padding=module.padding,
            dilation=module.stride,
            groups=C_in * N * V,
        ).squeeze(0)

        K_H_axis, K_W_axis = 1, 2
        _, _, K_H, K_W = module.weight.shape
        grad_weight = grad_weight.narrow(K_H_axis, 0, K_H).narrow(K_W_axis, 0, K_W)

        eingroup_eq = "vnio,x,y->v,{}o,i,x,y".format("" if sum_batch else "n,")
        grad_weight = eingroup(
            eingroup_eq, grad_weight, dim={"v": V, "n": N, "i": C_in, "o": C_out}
        )

        return grad_weight
