from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.extensions.firstorder.batch_dot_grad.base import BatchDotGradBase


class BatchDotGradConvTranspose1d(BatchDotGradBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(), params=["bias", "weight"]
        )
