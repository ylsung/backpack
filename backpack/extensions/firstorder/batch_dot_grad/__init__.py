from torch.nn import Linear

from backpack.extensions.backprop_extension import BackpropExtension

from . import linear


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
            },
        )
