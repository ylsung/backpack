from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.batch_dot_grad.base import BatchDotGradBase


class BatchDotGradLinear(BatchDotGradBase):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])
