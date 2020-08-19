"""A minimal example how to use the new pairwise gradient extension."""

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend
from backpack.extensions import BatchDotGrad
from backpack.utils.examples import load_one_batch_mnist

X, y = load_one_batch_mnist(batch_size=4)

model = extend(Sequential(Flatten(), Linear(784, 10),))
lossfunc = extend(CrossEntropyLoss())

loss = lossfunc(model(X), y)
with backpack(BatchDotGrad(), debug=True):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".batch_dot.shape:        ", param.batch_dot)
