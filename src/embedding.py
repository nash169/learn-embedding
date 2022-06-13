#!/usr/bin/env python

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, approximator):
        super(Embedding, self).__init__()

        self.net_ = approximator

    def forward(self, x):
        return torch.concat((x, self.net_(x)), axis=1)
        # return self.net_(x)

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

    def pullmetric(self, y, jac):
        return torch.bmm(jac.permute(0, 2, 1), torch.bmm(self.metric(y), jac))
        # return torch.matmul(jac.permute(0, 2, 1), jac)

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

    # Ambient space metric
    @property
    def metric(self):
        return self.metric_

    @metric.setter
    def metric(self, value):
        self.metric_ = value

    # Obstacles
    @property
    def obstacles(self):
        return self.obstacles_

    @obstacles.setter
    def obstacles(self, value):
        self.obstacles_ = value
