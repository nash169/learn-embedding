#!/usr/bin/env python

import torch
import torch.nn as nn


class Spherical(nn.Module):
    def __init__(self, grad=True):
        super(Spherical, self).__init__()

        self.spherical_ = nn.Parameter(torch.rand(1), requires_grad=grad)

    def forward(self, x):
        return nn.functional.linear(x, self.spherical_.exp()*torch.eye(x.shape[1]).to(x.device))

    # Params
    @property
    def spherical(self):
        return self.spherical_

    @spherical.setter
    def spherical(self, value):
        self.spherical_ = nn.Parameter(value[0], requires_grad=value[1])
