from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.firstorder.batch_grad_transforms.base import (
    BatchGradTransformsBase,
)


class BatchGradTransformsConv1d(BatchGradTransformsBase):
    def __init__(self):
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])
