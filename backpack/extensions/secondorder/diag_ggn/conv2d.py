from backpack.core.derivatives.conv import Conv2DDerivatives
from backpack.extensions.secondorder.diag_ggn.convnd import DiagGGNConvND


class DiagGGNConv2d(DiagGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )
