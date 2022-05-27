#!/usr/bin/env python

import torch
import torch.nn as nn

from .embedding import Embedding


class Dynamics(nn.Module):
    def __init__(self, dim, attractor, structure=[10, 10]):
        super(Dynamics, self).__init__()

        self.attractor_ = attractor

        self.stiffness_ = torch.nn.Linear(dim, dim, bias=False)

        self.dissipation_ = torch.nn.Linear(dim, dim, bias=False)

        self.embedding = Embedding(dim, structure)

    # Forward network pass
    def forward(self, X):
        # data
        x = X[:, :self.embedding.dim]
        v = X[:, self.embedding.dim:]

        # embedding
        f = self.embedding(x)

        # jacobian
        j = self.embedding.jacobian(x, f)

        # metric
        m = self.embedding.metric(j)

        # christoffel
        g = self.embedding.christoffel(x, m)

        return (torch.bmm(m.inverse(), -(self.dissipation(v)+self.stiffness(x)).unsqueeze(2)) - torch.bmm(torch.einsum('bqij,bi->bqj', g, v), v.unsqueeze(2))).squeeze()

    # Potential function
    def potential(self, x):
        d = (x - self.attractor).unsqueeze(2)

        return (self.stiffness.weight.matmul(d) * d).sum(axis=1)

    # Attractor setter/getter
    @property
    def attractor(self):
        return self.attractor_

    @attractor.setter
    def attractor(self, value):
        self.attractor_ = value

    # Stiffness matrix setter/getter
    @property
    def stiffness(self):
        return self.stiffness_

    @stiffness.setter
    def stiffness(self, value):
        self.stiffness_.weight = nn.Parameter(value[0], requires_grad=value[1])

    # Dissipative matrix setter/getter
    @property
    def dissipation(self):
        return self.dissipation_

    @dissipation.setter
    def dissipation(self, value):
        self.dissipation_.weight = nn.Parameter(
            value[0], requires_grad=value[1])

    # Diffeomorphism setter/getter
    @property
    def embedding(self):
        return self.embedding_

    @embedding.setter
    def embedding(self, value):
        self.embedding_ = value
