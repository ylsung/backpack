from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.extensions.firstorder.batch_grad_transforms.base import (
    BatchGradTransformsBase,
)


class BatchGradTransformsConvTranspose1d(BatchGradTransformsBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(), params=["bias", "weight"]
        )
