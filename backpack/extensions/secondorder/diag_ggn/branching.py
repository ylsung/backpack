from backpack.branching import BackpropedByMerge
from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class MergeDerivatives(BaseDerivatives):
    # avoid shape checks by overriding jac_t rather than _jac_t
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        """The (transposed) Jacobian of a sum is the identity matrix."""
        result = BackpropedByMerge((mat,))
        return result


class DiagGGNMerge(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MergeDerivatives())
