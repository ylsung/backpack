from test.extensions.implementation.base import ExtensionsImplementation

import backpack.extensions as new_ext
from backpack import backpack


class BackpackExtensions(ExtensionsImplementation):
    """Extension implementations with BackPACK."""

    def __init__(self, problem):
        problem.extend()
        super().__init__(problem)

    def batch_grad(self):
        with backpack(new_ext.BatchGrad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_grads = [p.grad_batch for p in self.problem.model.parameters()]
        return batch_grads

    def batch_dot_grad(self):
        with backpack(new_ext.BatchDotGrad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_dots = [p.batch_dot for p in self.problem.model.parameters()]
        return batch_dots

    def batch_l2_grad(self):
        with backpack(new_ext.BatchL2Grad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_l2_grad = [p.batch_l2 for p in self.problem.model.parameters()]
        return batch_l2_grad

    def sgs(self):
        with backpack(new_ext.SumGradSquared()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            sgs = [p.sum_grad_squared for p in self.problem.model.parameters()]
        return sgs

    def variance(self):
        with backpack(new_ext.Variance()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            variances = [p.variance for p in self.problem.model.parameters()]
        return variances

    def batch_grad_transforms(self, transforms):
        with backpack(new_ext.BatchGradTransforms(transforms)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_grad_transforms = [
                p.grad_batch_transforms for p in self.problem.model.parameters()
            ]
        return batch_grad_transforms
