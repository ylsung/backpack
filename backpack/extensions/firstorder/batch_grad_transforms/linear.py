from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.batch_grad_transforms.base import (
    BatchGradTransformsBase,
)


class BatchGradTransformsLinear(BatchGradTransformsBase):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])
