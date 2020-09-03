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


class BatchGradTransforms(BackpropExtension):
    """Transformations of individual gradients from a minibatch.

    Individual gradients require memory proportional to the minibatch size and
    the model size. This extension computes individual gradients and applies
    custom transformations on them. Only the result of these transformations
    are stored.

    Args:
        transforms (dict): Values are callables that will be applied on the
            individual gradients during backpropagation. Keys correspond to
            names under which the computation result will be saved in the
            dictionary saved to ``grad_batch_transforms``.

    .. note: beware of scaling issue

        The `individual gradients` depend on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``BatchGradTransforms`` will operate on

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.

    The concept of individual gradients is only meaningful if the
    objective is a sum of independent functions (no batchnorm).

    """

    def __init__(self, transforms):
        self._transforms = transforms
        super().__init__(
            savefield="grad_batch_transforms",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.BatchGradTransformsLinear(),
                Conv1d: conv1d.BatchGradTransformsConv1d(),
                Conv2d: conv2d.BatchGradTransformsConv2d(),
                Conv3d: conv3d.BatchGradTransformsConv3d(),
                ConvTranspose1d: conv_transpose1d.BatchGradTransformsConvTranspose1d(),
                ConvTranspose2d: conv_transpose2d.BatchGradTransformsConvTranspose2d(),
                ConvTranspose3d: conv_transpose3d.BatchGradTransformsConvTranspose3d(),
                BatchNorm1d: batchnorm1d.BatchGradTransformsBatchNorm1d(),
            },
        )

    def get_transforms(self):
        """Return dictionary containing (name, callable) pairs of transformations."""
        return self._transforms
