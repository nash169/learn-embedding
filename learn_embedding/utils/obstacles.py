#!/usr/bin/env python

from typing import Optional

import torch
import torch.nn as nn

from ..kernels.squared_exp import SquaredExp
from .torch_helper import TorchHelper


class Obstacles:
    @staticmethod
    def semi_circle(radius, center, rot=0, res=10, angle=torch.pi):
        theta = torch.linspace(0, angle, res)
        rot_mat = torch.tensor([[torch.cos(rot), -torch.sin(rot)], [torch.sin(rot), torch.cos(rot)]])

        return center + radius*torch.mm(torch.stack((theta.cos(), theta.sin()), axis=1), rot_mat)

    @staticmethod
    def square(center, a, b, res=10):
        y, x = torch.meshgrid(torch.linspace(0, b, res), torch.linspace(0, a, res))
        return center + torch.stack((torch.cat((x[0, :], x[-1, :], x[:, 0], x[:, -1])), torch.cat((y[0, :], y[-1, :], y[:, 0], y[:, -1]))), dim=1).unique(dim=0)


class KernelDeformation(nn.Module):
    def __init__(self, samples, 
                 kernel: Optional[nn.Module] = SquaredExp(),
                 weights: Optional[torch.Tensor] = None, 
                 requires_grad: Optional[bool] = False,
                 tolerance: Optional[float] = 0.0):
        super(KernelDeformation, self).__init__()

        self._samples = samples

        self._kernel = kernel

        if weights is not None:
            self._weights = nn.Parameter(weights, requires_grad=requires_grad)
        else:
            self._weights = nn.Parameter(torch.ones(samples.shape[0]), requires_grad=requires_grad)

        self._tol = tolerance # used only for velocity dependent local deformation

    def forward(self, x, v=None):
        if v is not None:
            with torch.no_grad():
                dist = self.samples.unsqueeze(1) - x
                cos_kernel = torch.nan_to_num(torch.einsum('kij,ij->ki', dist.div(torch.linalg.norm(dist, dim=2).unsqueeze(-1)), v.div(v.norm(dim=1).unsqueeze(-1))), nan=-1)
                alphas = TorchHelper.generalized_sigmoid(cos_kernel, b=1.0e6, m=self.tol)*self.weights.view(-1, 1)
            return torch.sum(self.kernel(self._samples, x)*alphas, axis=0).view(-1, 1)
        else:
            return torch.mv(self.kernel(self._samples, x).T, self.weights).unsqueeze(1)

    @ property
    def kernel(self):
        return self._kernel

    @ kernel.setter
    def kernel(self, value):
        self._kernel = value

    @ property
    def weights(self):
        return self._weights

    @ weights.setter
    def weights(self, value):
        self._weights = nn.Parameter(value)

    @ property
    def samples(self):
        return self._samples

    @ samples.setter
    def samples(self, value):
        self._samples = nn.Parameter(value)

    @ property
    def tol(self):
        return self._tol

    @ tol.setter
    def tol(self, value):
        self._tol = nn.Parameter(value)
