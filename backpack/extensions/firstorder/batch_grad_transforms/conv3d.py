from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.firstorder.batch_grad_transforms.base import (
    BatchGradTransformsBase,
)


class BatchGradTransformsConv3d(BatchGradTransformsBase):
    def __init__(self):
        super().__init__(derivatives=Conv3DDerivatives(), params=["bias", "weight"])
