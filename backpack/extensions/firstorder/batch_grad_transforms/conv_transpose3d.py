from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.firstorder.batch_grad_transforms.base import (
    BatchGradTransformsBase,
)


class BatchGradTransformsConvTranspose3d(BatchGradTransformsBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(), params=["bias", "weight"]
        )
