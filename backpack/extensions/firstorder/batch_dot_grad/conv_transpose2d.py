from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.extensions.firstorder.batch_dot_grad.base import BatchDotGradBase


class BatchDotGradConvTranspose2d(BatchDotGradBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(), params=["bias", "weight"]
        )
