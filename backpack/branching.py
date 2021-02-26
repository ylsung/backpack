"""Emulating branching with modules."""

import torch

# Enable modifications in BackPACK's backpropagation to handle branching
BRANCHING = True


class ActiveIdentity(torch.nn.Module):
    """Like ``torch.nn.Identity``, but creates a new node in the computation graph."""

    def forward(self, input):
        return 1.0 * input


class Branch(torch.nn.Module):
    """Module used by BackPACK to handle branching in the computation graph."""

    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        """Feed one input through a set of modules.

              → module1 → output1
        input → module2 → output2
              → ...     → ...

        Args:
            modules (torch.nn.Module): Sequence of modules. Input will be fed
                through every of these modules.

        """
        result = InputToMerge(module(input) for module in self.children())
        print(len(result))

        return result


class InputToMerge(tuple):
    """Acts as a container for the inputs handed to ``Merge`` module."""

    pass


class BackpropedByMerge(tuple):
    """Acts as container for the quantities backpropagated from the ``Merge`` module."""

    pass


class Merge(torch.nn.Module):
    """Module used by BackPACK to handle branch merges in the computation graph."""

    def forward(self, input):
        """Sum up all inputs. ``inputs`` must be a tuple containing multiple tensors."""
        if not isinstance(input, InputToMerge):
            raise ValueError(
                f"Input must be of class InputToMerge, but got {input.__class__}"
            )
        assert len(input) > 1, "Merge must act on multiple tensors"

        # print(input)
        result = sum(input)
        # print(result)

        return result


class Parallel:
    """Parallel series of modules. Used by BackPACK to emulate branched computations.

             module 1
    Branch → module 2 → Merge
             ...

    """

    pass
