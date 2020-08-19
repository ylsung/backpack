from test.extensions.implementation.base import ExtensionsImplementation

import torch


class AutogradExtensions(ExtensionsImplementation):
    """Extension implementations with autograd."""

    def batch_grad(self):
        N = self.problem.input.shape[0]
        batch_grads = [
            torch.zeros(N, *p.size()).to(self.problem.device)
            for p in self.problem.model.parameters()
        ]

        loss_list = torch.zeros((N))
        gradients_list = []
        for b in range(N):
            _, _, loss = self.problem.forward_pass(sample_idx=b)
            gradients = torch.autograd.grad(loss, self.problem.model.parameters())
            gradients_list.append(gradients)
            loss_list[b] = loss

        _, _, batch_loss = self.problem.forward_pass()
        factor = self.problem.get_reduction_factor(batch_loss, loss_list)

        for b, gradients in zip(range(N), gradients_list):
            for idx, g in enumerate(gradients):
                batch_grads[idx][b, :] = g.detach() * factor

        return batch_grads

    def batch_dot_grad(self):
        batch_grads = self.batch_grad()
        return self.pairwise_dot(batch_grads)

    @staticmethod
    def pairwise_dot(grad_batch):
        # flatten all feature dimensions
        grad_batch_flat = grad_batch.flatten(start_dim=1)
        # pairwise dot product
        return torch.einsum("if,jf->ij", grad_batch_flat, grad_batch_flat)
