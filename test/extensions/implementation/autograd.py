from test.extensions.implementation.base import ExtensionsImplementation

import torch

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.hessianfree.rop import R_op
from backpack.utils.convert_parameters import vector_to_parameter_list


class AutogradExtensions(ExtensionsImplementation):
    """Extension implementations with autograd."""

    def batch_grad(self, subsampling=None):
        batch_size = self.problem.input.shape[0]

        if subsampling is None:
            subsampling = list(range(batch_size))

        batch_grad = [
            torch.zeros(len(subsampling), *p.size()).to(self.problem.device)
            for p in self.problem.model.parameters()
        ]
        factor = self.problem.compute_reduction_factor()

        for out_idx, n in enumerate(subsampling):
            _, _, loss_n = self.problem.forward_pass(sample_idx=n)
            loss_n *= factor
            grad_n = torch.autograd.grad(loss_n, self.problem.model.parameters())

            for param_idx, g_n in enumerate(grad_n):
                batch_grad[param_idx][out_idx] = g_n.detach()

        return batch_grad

    def batch_l2_grad(self):
        batch_grad = self.batch_grad()
        batch_l2_grads = [(g ** 2).flatten(start_dim=1).sum(1) for g in batch_grad]
        return batch_l2_grads

    def sgs(self):
        factor = self.problem.compute_reduction_factor()
        sgs = [
            torch.zeros(*p.size()).to(self.problem.device)
            for p in self.problem.model.parameters()
        ]

        N = self.problem.input.shape[0]
        for b in range(N):
            _, _, loss_b = self.problem.forward_pass(sample_idx=b)
            loss_b *= factor
            grad_b = torch.autograd.grad(loss_b, self.problem.model.parameters())

            for param_idx, g_b in enumerate(grad_b):
                sgs[param_idx] += g_b.detach() ** 2

        return sgs

    def variance(self):
        batch_grad = self.batch_grad()
        variances = [torch.var(g, dim=0, unbiased=False) for g in batch_grad]
        return variances

    def _get_diag_ggn(self, loss, output, subsampling=None):
        if subsampling is None:

            def extract_ith_element_of_diag_ggn(i, p, loss, output):
                v = torch.zeros(p.numel()).to(self.problem.device)
                v[i] = 1.0
                vs = vector_to_parameter_list(v, [p])
                GGN_vs = ggn_vector_product_from_plist(loss, output, [p], vs)
                GGN_v = torch.cat([g.detach().view(-1) for g in GGN_vs])
                return GGN_v[i]

            diag_ggns = []
            for p in list(self.problem.model.parameters()):
                diag_ggn_p = torch.zeros_like(p).view(-1)

                for parameter_index in range(p.numel()):
                    diag_value = extract_ith_element_of_diag_ggn(
                        parameter_index, p, loss, output
                    )
                    diag_ggn_p[parameter_index] = diag_value

                diag_ggns.append(diag_ggn_p.view(p.size()))
            return diag_ggns

        else:
            diag_ggn_batch = self.diag_ggn_batch(subsampling=subsampling)
            N_axis = 0
            return [d.sum(N_axis) for d in diag_ggn_batch]

    def diag_ggn(self, subsampling=None):
        _, output, loss = self.problem.forward_pass()
        return self._get_diag_ggn(loss, output, subsampling=subsampling)

    def diag_ggn_batch(self, subsampling=None):
        batch_size = self.problem.input.shape[0]
        factor = self.problem.compute_reduction_factor()

        if subsampling is None:
            subsampling = list(range(batch_size))

        # batch_diag_ggn has entries [sample_idx][param_idx]
        batch_diag_ggn = [None for _ in range(len(subsampling))]

        for out_idx, n in enumerate(subsampling):
            _, output_n, loss_n = self.problem.forward_pass(sample_idx=n)
            loss_n *= factor
            diag_ggn_n = self._get_diag_ggn(loss_n, output_n)
            batch_diag_ggn[out_idx] = diag_ggn_n

        # rearrange to [param_idx][sample_idx]
        batch_diag_ggn = list(zip(*batch_diag_ggn))

        return [torch.stack(p_batch_diag_ggn) for p_batch_diag_ggn in batch_diag_ggn]

    def diag_h(self):
        _, _, loss = self.problem.forward_pass()

        def hvp(df_dx, x, v):
            Hv = R_op(df_dx, x, v)
            return [j.detach() for j in Hv]

        def extract_ith_element_of_diag_h(i, p, df_dx):
            v = torch.zeros(p.numel()).to(self.problem.device)
            v[i] = 1.0
            vs = vector_to_parameter_list(v, [p])

            Hvs = hvp(df_dx, [p], vs)
            Hv = torch.cat([g.detach().view(-1) for g in Hvs])

            return Hv[i]

        diag_hs = []
        for p in list(self.problem.model.parameters()):
            diag_h_p = torch.zeros_like(p).view(-1)

            df_dx = torch.autograd.grad(loss, [p], create_graph=True, retain_graph=True)
            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_h(parameter_index, p, df_dx)
                diag_h_p[parameter_index] = diag_value

            diag_hs.append(diag_h_p.view(p.size()))

        return diag_hs
