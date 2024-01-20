#!/usr/bin/env python

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import normalize

from ..covariances.spherical import Spherical
from ..utils.torch_helper import TorchHelper


class SecondGeometry(nn.Module):
    def __init__(self, embedding, attractor,
                 stiffness: Optional[nn.Module] = Spherical(grad=False),
                 dissipation: Optional[nn.Module] = Spherical(grad=False)):
        super(SecondGeometry, self).__init__()

        # Embedding
        self.embedding = embedding

        # Attractor
        self._attractor = attractor

        # Stiffness matrix
        self.stiffness = stiffness

        # Dissipation matrix
        self.dissipation = dissipation

        # Velocity Dependent Embedding
        self._velocity_embedding = False

    # Forward Dynamics
    def forward(self, x):
        # data
        p = x[:, :int(x.shape[1]/2)]
        v = x[:, int(x.shape[1]/2):]

        # embedding
        y = self.embedding(p, v) if self.velocity_embedding else self.embedding(p)

        # jacobian
        j = self.embedding.jacobian(p, y)

        # metric
        m = self.embedding.pullmetric(y, j)

        # christoffel
        g = self.embedding.christoffel(p, m)

        # potential energy
        f = self.stiffness(p - self.attractor)

        # metric free forces
        f_d = torch.zeros_like(p).to(x.device)

        # dissipation energy
        f_d += self.dissipation(v)

        # directional dissipation
        if hasattr(self, 'field'):
            # f += self.field_weight*(v - self.field(p))
            f += self.field_weight*(normalize(v, p=2, dim=1) - normalize(self.field(p), p=2, dim=1))

        # exponential dissipation
        if hasattr(self, 'exp_dissipation'):
            f_d += self.exp_dissipation_weight*self.exp_dissipation(p, self.attractor.unsqueeze(0))*v

        # dynamic harmonic components
        if hasattr(self.embedding, 'local_deformation') and hasattr(self, 'harmonic_start'):
            with torch.no_grad():
                d = self.embedding.local_deformation(p, v)
                harmonic_weight = TorchHelper.generalized_sigmoid(d, b=self.harmonic_growth, a=1.0, k=0.0, m=self.harmonic_start)
        else:
            harmonic_weight = 1.0

        f *= harmonic_weight
        f_d *= harmonic_weight

        return (torch.bmm(m.inverse(), -f.unsqueeze(2)) - torch.bmm(torch.einsum('bqij,bi->bqj', g, v), v.unsqueeze(2))).squeeze(2) - f_d
    
    def forward_fast(self, x):
        p = x[:, :int(x.shape[1]/2)]
        v = x[:, int(x.shape[1]/2):]

        y = self.embedding(p, v) if self.velocity_embedding else self.embedding(p)
        j = self.embedding.jacobian(p, y)
        m = self.embedding.pullmetric(y, j)
        g = self.embedding.christoffel_first(p, m)

        f_k = self.stiffness(self.attractor - p)
        f_d = self.dissipation(v)
        if hasattr(self, 'field'):
            f_d += self.field_weight*(normalize(v, p=2, dim=1) - normalize(self.field(p), p=2, dim=1))
        if hasattr(self, 'exp_dissipation'):
            f_d += self.exp_dissipation_weight*self.exp_dissipation(p, self.attractor.unsqueeze(0))*v

        if hasattr(self.embedding, 'local_deformation') and hasattr(self, 'harmonic_start'):
            with torch.no_grad():
                harmonic_weight = TorchHelper.generalized_sigmoid(self.embedding.local_deformation(p, v), b=self.harmonic_growth, a=1.0, k=0.0, m=self.harmonic_start)
        else:
            harmonic_weight = 1.0

        f_k *= harmonic_weight
        f_d *= harmonic_weight

        return torch.linalg.solve(m, f_k.unsqueeze(2) - torch.bmm(torch.einsum('bqij,bi->bqj', g, v), v.unsqueeze(2))).squeeze(2) - f_d


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
