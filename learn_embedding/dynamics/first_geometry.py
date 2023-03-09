#!/usr/bin/env python

import torch
import torch.nn as nn

from ..covariances.spherical import Spherical


class FirstGeometry(nn.Module):
    def __init__(self, embedding, attractor, stiffness=None):
        super(FirstGeometry, self).__init__()

        # Embedding
        self._embedding = embedding

        # Attractor
        self._attractor = attractor

        # Stiffness matrix
        if stiffness is not None:
            self._stiffness = stiffness
        else:
            self._stiffness = Spherical(grad=False)

    # Forward Dynamics
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

    # Embedding setter/getter
    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    # Attractor setter/getter
    @property
    def attractor(self):
        return self._attractor

    @attractor.setter
    def attractor(self, value):
        self._attractor = value

    # Stiffness matrix setter/getter
    @property
    def stiffness(self):
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value):
        self._stiffness = value
