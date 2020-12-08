from backpack.core.derivatives.conv import Conv1DDerivatives
from backpack.extensions.secondorder.diag_ggn.convnd import DiagGGNConvND


class DiagGGNConv1d(DiagGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )
