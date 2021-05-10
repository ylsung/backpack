from backpack.core.derivatives.layernorm import LayerNormDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradLayerNorm(BatchGradBase):
    def __init__(self):
        super().__init__(
            derivatives=LayerNormDerivatives(), params=["bias", "weight"]
        )
