#!/usr/bin/env python

import os
import numpy as np
import torch


def function(x):
    y = torch.empty(x.size(0), x.size(1)+1).to(x.device)

    y[:, 0] = x[:, 0].cos()
    y[:, 1] = x[:, 0].sin()*x[:, 1].cos()
    y[:, 2] = x[:, 0].sin()*x[:, 1].sin()

    return y


def jacobian(x):
    y = function(x)
    jac = torch.empty(x.size(0), y.size(1), x.size(1)).to(x.device)

    for i in range(y.size(1)):
        jac[:, i, :] = torch.autograd.grad(
            y[:, i], x, grad_outputs=torch.ones_like(y[:, i]), create_graph=True)[0]

    return jac


def hessian(x):
    jac = jacobian(x)
    hess = torch.empty(jac.size(0), jac.size(
        1), jac.size(2), x.size(1)).to(x.device)

    for i in range(jac.size(1)):
        for j in range(jac.size(2)):
            hess[:, i, j, :] = torch.autograd.grad(
                jac[:, i, j], x, grad_outputs=torch.ones_like(jac[:, i, j]), create_graph=True)[0]

    return hess


def metric(x):
    jac = jacobian(x)
    return torch.matmul(jac.permute(0, 2, 1), jac)


def metric_grad(x):
    m = metric(x)
    dm = torch.empty(m.size(0), m.size(
        1), m.size(2), x.size(1)).to(x.device)

    for i in range(m.size(1)):
        for j in range(m.size(2)):
            dm[:, i, j, :] = torch.autograd.grad(
                m[:, i, j], x, grad_outputs=torch.ones_like(m[:, i, j]), create_graph=True)[0]

    return dm


def christoffel(x):
    m = metric(x)
    im = m.inverse()
    dm = torch.empty(m.size(0), m.size(
        1), m.size(2), x.size(1)).to(x.device)

    for i in range(m.size(1)):
        for j in range(m.size(2)):
            dm[:, i, j, :] = torch.autograd.grad(
                m[:, i, j], x, grad_outputs=torch.ones_like(m[:, i, j]), create_graph=True)[0]

    return 0.5 * (torch.einsum('bqm,bmji->bqji', im, dm + dm.permute(0, 1, 3, 2)) - torch.einsum('bqm,bijm->bqij', im, dm))

    # return 0.5*(torch.tensordot(im, dm + dm.permute(0, 1, 3, 2), dims=([2], [1])) - torch.tensordot(im, dm, dims=([2], [3])))


# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# x = torch.tensor([[1, 1], [3, 2]], dtype=float).requires_grad_(True)
x = torch.tensor([[0.933993247757551, 0.678735154857773], [1, 1], [2, 2]],
                 dtype=float).requires_grad_(True)


print("Embedding")
print(function(x))

print("Jacobian")
print(jacobian(x))

print("Hessian")
print(hessian(x))

print("Metric")
print(metric(x))

print("Metric Grad")
print(metric_grad(x))

print("Christoffel")
print(christoffel(x))
