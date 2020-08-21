from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.firstorder.batch_dot_grad.base import BatchDotGradBase


class BatchDotGradConv3d(BatchDotGradBase):
    def __init__(self):
        super().__init__(derivatives=Conv3DDerivatives(), params=["bias", "weight"])
