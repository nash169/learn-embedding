#!/usr/bin/env python

import torch
import torch.nn as nn

from .kernel_machine import KernelMachine


class Embedding(nn.Module):
    def __init__(self, dim, structure=[10, 10]):
        super(Embedding, self).__init__()

        self.dim = dim

        # structure = [dim] + structure

        # layers = nn.ModuleList()

        # for i, _ in enumerate(structure[:-1]):
        #     layers.append(nn.Linear(structure[i], structure[i+1]))
        #     layers.append(nn.Tanh())

        # layers.append(nn.Linear(structure[-1], 1))

        # self.net_ = nn.Sequential(*(layers[i] for i in range(len(layers))))

        self.net_ = KernelMachine(self.dim, 500, 1, length=0.45)

    def forward(self, x):
        # self.net_.prediction_.weight.data = torch.abs(
        #     self.net_.prediction_.weight.data)
        return torch.concat((x, self.net_(x)), axis=1)

    def jacobian(self, x, y):
        jac = torch.empty(x.size(0), y.size(1), x.size(1)).to(x.device)

        for i in range(y.size(1)):
            jac[:, i, :] = torch.autograd.grad(
                y[:, i], x, grad_outputs=torch.ones_like(y[:, i]), create_graph=True)[0]

        return jac

    def hessian(self, x, jac):
        hess = torch.empty(jac.size(0), jac.size(
            1), jac.size(2), x.size(1)).to(x.device)

        for i in range(jac.size(1)):
            for j in range(jac.size(2)):
                hess[:, i, j, :] = torch.autograd.grad(
                    jac[:, i, j], x, grad_outputs=torch.ones_like(jac[:, i, j]), create_graph=True)[0]

        return hess

    def metric(self, jac):
        return torch.matmul(jac.permute(0, 2, 1), jac)

    def christoffel(self, x, m):
        im = m.inverse()
        dm = torch.empty(m.size(0), m.size(
            1), m.size(2), x.size(1)).to(x.device)

        for i in range(m.size(1)):
            for j in range(m.size(2)):
                dm[:, i, j, :] = torch.autograd.grad(
                    m[:, i, j], x, grad_outputs=torch.ones_like(m[:, i, j]), create_graph=True)[0]

        return 0.5 * (torch.einsum('bqm,bmji->bqji', im, dm + dm.permute(0, 1, 3, 2)) - torch.einsum('bqm,bijm->bqij', im, dm))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            nn.init.constant_(m.bias, 0.1)

    # Dimension setter/getter
    @property
    def dim(self):
        return self.dim_

    @dim.setter
    def dim(self, value):
        self.dim_ = value
