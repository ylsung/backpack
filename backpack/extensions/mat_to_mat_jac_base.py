from .module_extension import ModuleExtension


class MatToJacMat(ModuleExtension):
    """ Backpropagate by multiplying with the transpose output-input Jacobian."""

    def __init__(self, derivatives, params=None):
        super().__init__(params)
        self.derivatives = derivatives

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        """Apply the tranpose output-input Jacobian to the backpropagated quantity.

        Args:
            backproped ([torch.Tensor, tensor]): List of backpropagated tensors or a
                single backpropagated tensor, to which the tranpose output-input
                Jacobian should be applied.

        """

        # TODO Remove after all extensions based on MatToJacMat support subsampling
        try:
            subsampling = ext.get_subsampling()
        except AttributeError:
            subsampling = None

        if isinstance(backproped, list):
            return [
                self.derivatives.jac_t_mat_prod(
                    module, grad_inp, grad_out, M, subsampling=subsampling
                )
                for M in backproped
            ]
        else:
            return self.derivatives.jac_t_mat_prod(
                module, grad_inp, grad_out, backproped, subsampling=subsampling
            )
