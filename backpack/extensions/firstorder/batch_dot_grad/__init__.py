from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Linear,
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (
    batchnorm1d,
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    linear,
)


class BatchDotGrad(BackpropExtension):
    """Individual gradients pairwise dot product for each sample in a minibatch.

    Stores the output in ``batch_dot`` as a ``[N x N]`` tensor,
    where ``N`` is the batch size.

    The entry ``[i, j]`` corresponds to the dot product of individual gradients
    for sample ``i`` and sample ``j`` in the minibatch.

    .. note: beware of scaling issue

        The `individual gradients` depend on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``BatchDotGrad`` will return

        - ``gᵢᵀgⱼ`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``(¹/ₙ gᵢ)ᵀ (¹/ₙ gⱼ)`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.

    The concept of individual gradients and their dot product  is only meaningful
    if the objective is a sum of independent functions (no batchnorm).
    """

    def __init__(self):
        super().__init__(
            savefield="batch_dot",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.BatchDotGradLinear(),
                Conv1d: conv1d.BatchDotGradConv1d(),
                Conv2d: conv2d.BatchDotGradConv2d(),
                Conv3d: conv3d.BatchDotGradConv3d(),
                ConvTranspose1d: conv_transpose1d.BatchDotGradConvTranspose1d(),
                ConvTranspose2d: conv_transpose2d.BatchDotGradConvTranspose2d(),
                ConvTranspose3d: conv_transpose3d.BatchDotGradConvTranspose3d(),
                BatchNorm1d: batchnorm1d.BatchDotGradBatchNorm1d(),
            },
        )
