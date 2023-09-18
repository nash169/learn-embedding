#!/usr/bin/env python

from typing import Optional

import torch
import torch.nn as nn

from ..covariances.spherical import Spherical


class SecondGeometry(nn.Module):
    def __init__(self, embedding, attractor, stiffness: Optional[nn.Module] = None,  dissipation: Optional[nn.Module] = None, field: Optional[nn.Module] = None, field_weight: Optional[float] = 1.0):
        super(SecondGeometry, self).__init__()

        # Embedding
        self.embedding = embedding

        # Attractor
        self._attractor = attractor

        # Stiffness matrix
        if stiffness is not None:
            self.stiffness = stiffness
        else:
            self.stiffness = Spherical(grad=False)

        # Dissipation matrix
        if dissipation is not None:
            self.dissipation = dissipation
        else:
            self.dissipation = Spherical(grad=False)

        # Reference velocity field
        if field is not None:
            self.field = field
            self.field_weight = field_weight

        # Velocity Dependent Embedding
        self._velocity_embedding = False

    # Forward Dynamics
    def forward(self, x):
        # data
        pos = x[:, :int(x.shape[1]/2)]
        vel = x[:, int(x.shape[1]/2):]
        # embedding
        y = self.embedding(pos, vel) if self.velocity_embedding else self.embedding(pos)
        # jacobian
        j = self.embedding.jacobian(pos, y)
        # metric
        m = self.embedding.pullmetric(y, j)
        # christoffel
        g = self.embedding.christoffel(pos, m)
        # desired state
        xd = pos - self.attractor
        vd = vel - self.field_weight*self.field(pos) if hasattr(self, 'field') else vel
        # dynamics harmonic components
        if hasattr(self.embedding, 'local_deformation'):
            harmonic_weight = torch.ones(x.shape[0], 1).to(x.device)
            harmonic_weight[self.embedding.local_deformation(pos, vel) >= 0.01] = 0.0
            # harmonic_weight[self.embedding.local_deformation(pos) >= 0.01] = 0.0
            # print(harmonic_weight)

        # harmonic_weight = 0.0

        return (torch.bmm(m.inverse(), -(harmonic_weight*self.dissipation(vd)+harmonic_weight*self.stiffness(xd)).unsqueeze(2))
                - torch.bmm(torch.einsum('bqij,bi->bqj', g, vel), vel.unsqueeze(2))).squeeze(2)

    def geodesic(self, x):
        # data
        pos = x[:, :int(x.shape[1]/2)]
        vel = x[:, int(x.shape[1]/2):]
        # embedding
        y = self.embedding(pos)
        # y = self.embedding(pos, vel) if self.velocity_embedding else self.embedding(pos)
        # jacobian
        j = self.embedding.jacobian(pos, y)
        # metric
        m = self.embedding.pullmetric(y, j)
        # christoffel
        g = self.embedding.christoffel(pos, m)
        # desired state
        # vd = vel - self.field(pos) if hasattr(self, 'field') else vel

        return -torch.bmm(torch.einsum('bqij,bi->bqj', g, vel), vel.unsqueeze(2)).squeeze(2)

    # Potential function
    def potential(self, x):
        d = x - self.attractor
        return (d*self.stiffness(d)).sum(axis=1)

    # Attractor setter/getter
    @property
    def attractor(self) -> torch.Tensor:
        return self._attractor

    @attractor.setter
    def attractor(self, value: torch.Tensor):
        self._attractor = value

    # Attractor setter/getter
    @property
    def velocity_embedding(self) -> torch.Tensor:
        return self._velocity_embedding

    @velocity_embedding.setter
    def velocity_embedding(self, value: torch.Tensor):
        self._velocity_embedding = value
