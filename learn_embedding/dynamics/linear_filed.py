#!/usr/bin/env python

import torch
import torch.nn as nn

from ..covariances import Spherical


class LinearField(nn.Module):
    def __init__(self, attractor, covariance=None):
        super(LinearField, self).__init__()

        self.attractor = attractor

        if covariance is not None:
            self.covariance = covariance
        else:
            self.covariance = Spherical(1., False)

    def forward(self, x):
        return -self.covariance(x - self.attractor)

    # Attractor setter/getter
    @property
    def attractor(self) -> torch.Tensor:
        return self._attractor

    @attractor.setter
    def attractor(self, value: torch.Tensor):
        self._attractor = value
