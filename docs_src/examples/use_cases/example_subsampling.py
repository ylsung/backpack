"""Mini-batch sub-sampling
==========================

If not further specified, BackPACK considers all samples in a mini-batch for
its quantities. In this example we will show how to restrict the computations
to a subset of samples in the current mini-batch.

This may be interesting for applications that seek to treat parts of batch
samples differently, e.g. computing curvature and gradient information on
different subsets. Limiting the computations to fewer samples also requires
less operations and memory.
"""

# %%
# Let's start by loading some dummy data and extending the model

import torch
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend
from backpack.extensions import BatchDiagGGNExact, BatchGrad, DiagGGNExact
from backpack.utils.examples import load_one_batch_mnist

# make deterministic
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
X, y = load_one_batch_mnist(batch_size=256)
X, y = X.to(device), y.to(device)

# model
model = Sequential(Flatten(), Linear(784, 10)).to(device)
lossfunc = CrossEntropyLoss().to(device)

model = extend(model)
lossfunc = extend(lossfunc)

# %%
# Individual gradients for a mini-batch subset
# --------------------------------------------
#
# Let's say we only want to compute individual gradients for samples 0, 1,
# 13, and 42. Naively, we could perform the computation for all samples, then
# slice out the samples we care about.

# selected samples
subsampling = [0, 1, 13, 42]

loss = lossfunc(model(X), y)

with backpack(BatchGrad()):
    loss.backward()

# naive approach: compute for all, slice out relevant
subsampled_naive = [p.grad_batch[subsampling] for p in model.parameters()]

# %%
# This is not efficient, as the individual gradient computations spent on
# all other samples in the mini-batch are wasted. We can do better by
# specifying the active samples directly with the ``subsampling`` argument of
# :py:class:`BatchGrad <backpack.extensions.BatchGrad>`.

loss = lossfunc(model(X), y)

# efficient approach: specify active samples during backward pass
with backpack(BatchGrad(subsampling=subsampling)):
    loss.backward()

subsampled_efficient = [p.grad_batch for p in model.parameters()]

# %%
# Let's verify that both ways yield the same result:

match = all(
    torch.allclose(naive, efficient)
    for naive, efficient in zip(subsampled_naive, subsampled_efficient)
)

print(f"Naive and efficient gradient results match? {match}")

if not match:
    raise ValueError("Naive and efficient gradient results don't match.")

# %%
# Individual diagonal curvature for a mini-batch subset
# -----------------------------------------------------
# Sub-sampling also works with second-order extensions. Let's compare three ways
# to compute the diagonal GGN/Fisher of samples 0, 1, 13, 42:
#
# - (naive) Compute individual GGN/Fisher diagonal for all samples, slice out the
#   relevant samples, sum over samples.
# - (efficient) Directly compute the GGN/Fisher diagonal on the specified samples.
# - (check) Like naive, but uses subsampling for individual GGN/Fisher diagonals.
#   Included as a double check.

# selected samples
subsampling = [0, 1, 13, 42]

# %%
# Here is the naive approach:

loss = lossfunc(model(X), y)

with backpack(BatchDiagGGNExact()):
    loss.backward()

batch_axis = 0
subsampled_naive = [
    p.diag_ggn_exact_batch[subsampling].sum(batch_axis) for p in model.parameters()
]


# %%
# The efficient, recommended approach specifies the ``subsampling`` argument of
# :py:class:`DiagGGNExact<backpack.extensions.DiagGGNExact>`:

loss = lossfunc(model(X), y)

with backpack(DiagGGNExact(subsampling=subsampling)):
    loss.backward()

subsampled_efficient = [p.diag_ggn_exact for p in model.parameters()]

# %%
# To double-check our results, we compute the subsampled individual diagonals
# using :py:class:`BatchDiagGGNExact<backpack.extensions.BatchDiagGGNExact>`,
# then perform the summation over samples manually:

loss = lossfunc(model(X), y)

with backpack(BatchDiagGGNExact(subsampling=subsampling)):
    loss.backward()

batch_axis = 0
subsampled_check = [p.diag_ggn_exact_batch.sum(batch_axis) for p in model.parameters()]


# %%
# Time to see if all three approaches have identical results:

match = all(
    torch.allclose(naive, efficient) and torch.allclose(efficient, check)
    for naive, efficient, check in zip(
        subsampled_naive, subsampled_efficient, subsampled_check
    )
)

print(f"Naive and efficient diagonal curvature results match? {match}")

if not match:
    raise ValueError("Naive and efficient diagonal curvature results don't match.")
