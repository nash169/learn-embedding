#!/usr/bin/env python

from turtle import forward
import torch
import torch.nn as nn


# Default desired velocity
class ZeroVelocity(nn.Module):
    def __init__(self):
        super(ZeroVelocity, self).__init__()

    def forward(self, x):
        return 0


class DynamicsSecond(nn.Module):
    def __init__(self, attractor, stiffness, dissipation, embedding):
        super(DynamicsSecond, self).__init__()

        self.attractor = attractor

        self.stiffness = stiffness

        self.dissipation = dissipation

        self.embedding = embedding

        self.velocity_ = ZeroVelocity()

    # Forward network pass
    def forward(self, X):
        # data
        x = X[:, :int(X.shape[1]/2)]
        v = X[:, int(X.shape[1]/2):]

        # embedding
        y = self.embedding(x)

        # jacobian
        j = self.embedding.jacobian(x, y)

        # metric
        m = self.embedding.pullmetric(y, j)

        # christoffel
        g = self.embedding.christoffel(x, m)

        # desired state
        xd = x - self.attractor
        vd = v - self.velocity(x)

        return (torch.bmm(m.inverse(), -(self.dissipation(vd)+self.stiffness(xd)).unsqueeze(2)) - torch.bmm(torch.einsum('bqij,bi->bqj', g, vd), vd.unsqueeze(2))).squeeze(2)

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

    # Dissipative matrix setter/getter
    @property
    def dissipation(self):
        return self.dissipation_

    @dissipation.setter
    def dissipation(self, value):
        self.dissipation_ = value

    # Diffeomorphism setter/getter
    @property
    def embedding(self):
        return self.embedding_

    @embedding.setter
    def embedding(self, value):
        self.embedding_ = value

    # Desired velocity setter/getter
    @property
    def velocity(self):
        return self.velocity_

    @velocity.setter
    def velocity(self, value):
        self.velocity_ = value
