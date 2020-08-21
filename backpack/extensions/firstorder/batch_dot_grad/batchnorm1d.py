from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.firstorder.batch_dot_grad.base import BatchDotGradBase


class BatchDotGradBatchNorm1d(BatchDotGradBase):
    def __init__(self):
        super().__init__(
            derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"]
        )
