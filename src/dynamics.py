#!/usr/bin/env python

import torch
import torch.nn as nn


# Default desired velocity
class ZeroVelocity(nn.Module):
    def __init__(self):
        super(ZeroVelocity, self).__init__()

    def forward(self, x):
        return 0


class Dynamics(nn.Module):
    def __init__(self, embedding, stiffness=None, attractor=None, dissipation=None, velocity=None):
        super(Dynamics, self).__init__()

        # Embedding
        self.embedding_ = embedding

        # Stiffness matrix
        if stiffness is not None:
            self.stiffness_ = stiffness

        # Attractor
        if attractor is not None:
            self.attractor_ = attractor
        else:
            self.attractor_ = 0

        # Dissipation matrix
        if dissipation is not None:
            self.dissipation_ = dissipation

        # Reference velocity field
        if velocity is not None:
            self.velocity_ = velocity
        else:
            self.velocity_ = ZeroVelocity()

    # Forward network pass
    def dissipative_system(self, X):
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

        return (torch.bmm(m.inverse(), -(self.dissipation(vd)+self.stiffness(xd)).unsqueeze(2)) - torch.bmm(torch.einsum('bqij,bi->bqj', g, v), v.unsqueeze(2))).squeeze(2)

    def geodesic(self, X):
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
        vd = v - self.velocity(x)

        return -torch.bmm(torch.einsum('bqij,bi->bqj', g, vd), vd.unsqueeze(2)).squeeze(2)

    def gradient_system(self, X):
        # embedding
        y = self.embedding(X)

        # jacobian
        j = self.embedding.jacobian(X, y)

        # metric
        m = self.embedding.pullmetric(y, j)

        return (torch.bmm(m.inverse(), -self.stiffness(X-self.attractor).unsqueeze(2))).squeeze(2)

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

    # Desired reference field setter/getter
    @property
    def velocity(self):
        return self.velocity_

    @velocity.setter
    def velocity(self, value):
        self.velocity_ = value

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

    # Embedding setter/getter
    @property
    def embedding(self):
        return self.embedding_

    @embedding.setter
    def embedding(self, value):
        self.embedding_ = value


class FirstOrder(Dynamics):
    def __init__(self, embedding, stiffness, attractor):
        super(FirstOrder, self).__init__(embedding, stiffness, attractor)

    def forward(self, x):
        return self.gradient_system(x)

    def integrate(self, x, T, dt):
        dim = int(x.shape[1]/2)
        steps = int(T/dt)
        traj = torch.zeros(steps, x.shape[0], x.shape[1])
        traj[0, :, :] = x.requires_grad_(True)

        for i in range(steps-1):
            traj[i, :, dim:] = self.gradient_system(traj[i, :, :dim])
            traj[i+1, :, :dim] = traj[i, :, :dim] + traj[i, :, dim:]*dt

        return traj


class SecondOrder(Dynamics):
    def __init__(self, embedding, stiffness, attractor, dissipation, velocity=None):
        super(SecondOrder, self).__init__(
            embedding, stiffness, attractor, dissipation, velocity)

    def forward(self, x):
        return self.dissipative_system(x)

    def integrate(self, x, T, dt):
        dim = int(x.shape[1]/2)
        steps = int(T/dt)
        traj = torch.zeros(steps, x.shape[0], x.shape[1])
        traj[0, :, :] = x.requires_grad_(True)

        for i in range(steps-1):
            traj[i+1, :, dim:] = traj[i, :, dim:] + \
                self.dissipative_system(traj[i, :, :])*dt
            traj[i+1, :, :dim] = traj[i, :, :dim] + traj[i+1, :, dim:]*dt

        return traj
