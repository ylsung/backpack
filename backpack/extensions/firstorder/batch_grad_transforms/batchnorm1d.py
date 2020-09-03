from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.firstorder.batch_grad_transforms.base import (
    BatchGradTransformsBase,
)


class BatchGradTransformsBatchNorm1d(BatchGradTransformsBase):
    def __init__(self):
        super().__init__(
            derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"]
        )
