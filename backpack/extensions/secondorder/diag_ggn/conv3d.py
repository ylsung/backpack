from backpack.core.derivatives.conv import Conv3DDerivatives
from backpack.extensions.secondorder.diag_ggn.convnd import DiagGGNConvND


class DiagGGNConv3d(DiagGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )
