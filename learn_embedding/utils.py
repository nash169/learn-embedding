#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import numpy as np
# import copy

from .dynamics import FirstOrder, SecondOrder
from .approximators import *
from .embedding import Embedding
from .covariances import *


def fix_params(model):
    for param in model.parameters():
        param.requires_grad = False


def create_model(X, order="first"):
    if order == "first":
        # Space dimension
        dim = int(X.shape[1])

        # Exact Kernel Expansion
        approximator = KernelExpansion(X)

        # Approximated Kernel Expansion
        # approximator = KernelMachine(dim, 1000, 1, length=0.4)

        # Neural Network
        # approximator = FeedForward(dim, [64], 1)

        # Embedding
        embedding = Embedding(approximator)
        # embedding.apply(embedding.init_weights)

        # Stiffness matrix
        stiffness = SPD(dim)

        # Attractor
        attractor = X[-1, :]

        return FirstOrder(embedding, stiffness, attractor).to(X.device)

    elif order == "second":
        # Space dimension
        dim = int(X.shape[1]/2)

        # Exact Kernel Expansion
        # approximator = KernelExpansion(X[:, :dim])

        # Approximated Kernel Expansion
        # approximator = KernelMachine(dim, 1000, 1, length=0.4)

        # Neural Network
        approximator = FeedForward(dim, [64], 1)

        # Embedding
        embedding = Embedding(approximator)
        # embedding.apply(embedding.init_weights)

        # Stiffness matrix
        stiffness = SPD(dim)

        # Attractor
        attractor = X[-1, :dim]

        # Dissipation matrix
        dissipation = SPD(dim)

        # Velocity reference field
        velocity = FirstOrder(Embedding(FeedForward(dim, [32], 1)),
                              Spherical(False), attractor).to(X.device)
        velocity.embedding.apply(velocity.embedding.init_weights)
        velocity.stiffness.spherical = (torch.tensor(0.1), False)
        fix_params(velocity)  # copy.deepcopy(stiffness)

        return SecondOrder(embedding, stiffness, attractor, dissipation, velocity).to(X.device)

    elif order == "first_flat":
        # Space dimension
        dim = int(X.shape[1])

        # Neural Network
        approximator = FeedForward(dim, [64], 1)

        # Embedding
        embedding = Embedding(approximator)
        embedding.apply(embedding.init_weights)
        fix_params(embedding)

        # Stiffness matrix
        stiffness = Spherical(False)

        # Attractor
        attractor = X[-1, :]

        return FirstOrder(embedding, stiffness, attractor).to(X.device)

    elif order == "second_flat":
        # Space dimension
        dim = int(X.shape[1]/2)

        # Neural Network
        approximator = FeedForward(dim, [64], 1)

        # Embedding
        embedding = Embedding(approximator)
        embedding.apply(embedding.init_weights)
        fix_params(embedding)

        # Stiffness matrix
        stiffness = Spherical(False)
        stiffness.spherical = (torch.tensor(1), False)

        # Attractor
        attractor = X[-1, :dim]

        # Dissipation matrix
        dissipation = Spherical(False)
        dissipation.spherical = (torch.tensor(2), False)

        # Velocity reference field
        velocity = FirstOrder(Embedding(FeedForward(dim, [32], 1)),
                              Spherical(False), attractor).to(X.device)
        velocity.embedding.apply(velocity.embedding.init_weights)
        velocity.stiffness.spherical = (torch.tensor(0.1), False)
        fix_params(velocity)  # copy.deepcopy(stiffness)

        return SecondOrder(embedding, stiffness, attractor, dissipation).to(X.device)

    else:
        print("Case not found.")


# Map dataset in between un upper and lower
# (should be expanded to n-dimension and maybe by default a square box)
def linear_map(x, xmin, xmax, ymin, ymax):
    m = (ymin - ymax)/(xmin-xmax)
    q = ymin - m*xmin

    y = m*x + q

    return y


# RBF kernel
def squared_exp(x, y, sigma=1, eta=1):
    l = -.5 / sigma**2
    xx = torch.einsum('ij,ij->i', x, x).unsqueeze(1)
    yy = torch.einsum('ij,ij->i', y, y).unsqueeze(0)
    k = -2 * torch.mm(x, y.T) + xx + yy
    k *= l
    return eta*torch.exp(k)


# Control barrier function
def infty_exp(x, y, r=1, a=1, b=2):
    xx = torch.einsum('ij,ij->i', x, x).unsqueeze(1)
    yy = torch.einsum('ij,ij->i', y, y).unsqueeze(0)
    k = -2 * torch.mm(x, y.T) + xx + yy
    return torch.exp(a/(b*torch.pow(k.sqrt() - r, b))) - 1


def paraboloid(x, y):
    xx = torch.einsum('ij,ij->i', x, x).unsqueeze(1)
    yy = torch.einsum('ij,ij->i', y, y).unsqueeze(0)
    k = -2 * torch.mm(x, y.T) + xx + yy
    return k.pow(2)


# Metrics
def metric_exp(y, y_obs, sigma, eta=1):
    d = y-y_obs
    k = eta*torch.exp(-0.5*torch.sum(d.pow(2), dim=1) /
                      sigma ** 2).unsqueeze(1).unsqueeze(2)
    return torch.bmm(d.unsqueeze(2), d.unsqueeze(1)) * k.pow(2)/np.power(sigma, 4) + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)


def metric_infty(y, y_obs, a, b, r=0):
    d = y-y_obs
    k = (torch.norm(d, dim=1) - r).unsqueeze(1).unsqueeze(2)
    k = torch.exp(a/(b*torch.pow(k, b)))
    return torch.bmm(d.unsqueeze(2), d.unsqueeze(1)) * (k-1) + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)

# def metric_infty(y):
#     xbar = y-y_obs
#     xnorm = (torch.norm(xbar, dim=1)).unsqueeze(1).unsqueeze(2)
#     d = xnorm - r_obs
#     k = -a_obs*torch.pow(d, -b_obs-1)/xnorm*torch.exp(a_obs /
#                                                       (b_obs*torch.pow(d, b_obs)))
#     return torch.bmm(xbar.unsqueeze(2), xbar.unsqueeze(1)) * k.pow(2) + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)


# Save model/dict
def save(file, model, dict=True):
    assert issubclass(type(model), torch.nn.Module)
    if dict:
        if not os.path.exists('dicts'):
            os.makedirs('dicts')
        torch.save(model.state_dict(), os.path.join(
            'dicts', '{}.pt'.format(file)))
    else:
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model, os.path.join(
            'models', '{}.pt'.format(file)))


# Load model/dict
def load(file, model, device):
    model.load_state_dict(torch.load(
        os.path.join('models', '{}.pt'.format(file)), map_location=torch.device(device)))
    model.eval()


def load(file, device):
    model = torch.load(
        os.path.join('models', '{}.pt'.format(file)), map_location=torch.device(device))
    model.eval()
    return model


class Rotation(nn.Module):
    def __init__(self):
        super(Rotation, self).__init__()

        self.rotation_ = nn.Parameter(torch.tensor(0), requires_grad=False)

    def forward(self, x):
        R = torch.tensor([[self.rotation.cos(), -self.rotation.sin()],
                         [self.rotation.sin(), self.rotation.cos()]]).to(x.device)
        return nn.functional.linear(x, R)

    # Params
    @property
    def rotation(self):
        return self.rotation_

    @rotation.setter
    def rotation(self, value):
        self.rotation_ = nn.Parameter(value, requires_grad=False)
