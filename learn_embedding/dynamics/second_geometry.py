#!/usr/bin/env python

import torch
import torch.nn as nn

from ..covariances.spherical import Spherical


# Default desired velocity
class ZeroVelocity(nn.Module):
    def __init__(self):
        super(ZeroVelocity, self).__init__()

    def forward(self, x):
        return 0


class SecondGeometry(nn.Module):
    def __init__(self, embedding, attractor, stiffness=None,  dissipation=None, velocity=None):
        super(SecondGeometry, self).__init__()

        # Embedding
        self._embedding = embedding

        # Attractor
        self._attractor = attractor

        # Stiffness matrix
        if stiffness is not None:
            self._stiffness = stiffness
        else:
            self._stiffness = Spherical(grad=False)

        # Dissipation matrix
        if dissipation is not None:
            self._dissipation = dissipation
        else:
            self._dissipation = Spherical(grad=False)

        # Reference velocity field
        if velocity is not None:
            self.velocity_ = velocity
        else:
            self.velocity_ = ZeroVelocity()

    # Forward Dynamics
    def forward(self, x):
        # data
        pos = x[:, :int(x.shape[1]/2)]
        vel = x[:, int(x.shape[1]/2):]
        # embedding
        y = self.embedding(pos)
        # jacobian
        j = self.embedding.jacobian(pos, y)
        # metric
        m = self.embedding.pullmetric(y, j)
        # christoffel
        g = self.embedding.christoffel(pos, m)
        # desired state
        xd = pos - self.attractor
        vd = vel - self.velocity(pos)

        return (torch.bmm(m.inverse(), -(self.dissipation(vd)+self.stiffness(xd)).unsqueeze(2))
                - torch.bmm(torch.einsum('bqij,bi->bqj', g, vd), vd.unsqueeze(2))).squeeze(2)

    def geodesic(self, x):
        # data
        pos = x[:, :int(x.shape[1]/2)]
        vel = x[:, int(x.shape[1]/2):]
        # embedding
        y = self.embedding(pos)
        # jacobian
        j = self.embedding.jacobian(pos, y)
        # metric
        m = self.embedding.pullmetric(y, j)
        # christoffel
        g = self.embedding.christoffel(pos, m)
        # desired state
        vd = vel - self.velocity(pos)

        return -torch.bmm(torch.einsum('bqij,bi->bqj', g, vd), vd.unsqueeze(2)).squeeze(2)

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

    # Desired reference field setter/getter
    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = value

    # Dissipative matrix setter/getter
    @property
    def dissipation(self):
        return self._dissipation

    @dissipation.setter
    def dissipation(self, value):
        self._dissipation = value
