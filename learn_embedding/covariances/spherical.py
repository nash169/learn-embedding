#!/usr/bin/env python

import torch
import torch.nn as nn


class Spherical(nn.Module):
    def __init__(self, eval=1, grad=True):
        super(Spherical, self).__init__()

        # self.eval = eval
        self._eval = nn.Parameter(torch.tensor(eval).log(), requires_grad=grad)

    def forward(self, x):
        return self.eval * x

    # Params
    @property
    def eval(self):
        return self._eval.exp()

    @eval.setter
    def eval(self, value):
        self._eval = nn.Parameter(torch.tensor(value).log(), requires_grad=True)
