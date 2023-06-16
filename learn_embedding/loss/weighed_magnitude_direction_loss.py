#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class WeighedMagnitudeDirectionLoss(_Loss):

    __constants__ = ['reduction']

    def __init__(self, magnitude_weight=0.5, direction_weight=0.5, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self._magnitude_weight = magnitude_weight
        self._direction_weight = direction_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-08)

        return (self._magnitude_weight * (target.norm(dim=1) - input.norm(dim=1)).square() + self._direction_weight*cosine_similarity(target,  input).square()).mean()
