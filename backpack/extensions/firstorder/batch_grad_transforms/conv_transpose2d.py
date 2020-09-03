from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.extensions.firstorder.batch_grad_transforms.base import (
    BatchGradTransformsBase,
)


class BatchGradTransformsConvTranspose2d(BatchGradTransformsBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(), params=["bias", "weight"]
        )
