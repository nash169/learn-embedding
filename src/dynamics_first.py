#!/usr/bin/env python

import torch
import torch.nn as nn


class DynamicsFirst(nn.Module):
    def __init__(self, attractor, stiffness, embedding):
        super(DynamicsFirst, self).__init__()

        self.attractor = attractor

        self.stiffness = stiffness

        self.embedding = embedding

    # Forward network pass
    def forward(self, x):
        # embedding
        y = self.embedding(x)

        # jacobian
        j = self.embedding.jacobian(x, y)

        # metric
        m = self.embedding.pullmetric(y, j)

        return (torch.bmm(m.inverse(), -self.stiffness(x-self.attractor).unsqueeze(2))).squeeze(2)

    # Potential function
    def potential(self, x):
        d = x - self.attractor

        return (d*self.stiffness(d)).sum(axis=1)

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
        self.stiffness_ = value

    # Diffeomorphism setter/getter
    @property
    def embedding(self):
        return self.embedding_

    @embedding.setter
    def embedding(self, value):
        self.embedding_ = value
