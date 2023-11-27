#!/usr/bin/env python

import os
import torch


class TorchHelper():
    @staticmethod
    def save(model, path):
        torch.save(model.state_dict(), os.path.join('', '{}.pt'.format(path)))

    @staticmethod
    def load(model, path, device):
        model.load_state_dict(torch.load(os.path.join('', '{}.pt'.format(path)), map_location=torch.device(device)))

    @staticmethod
    def set_grad(model, grad):
        for param in model.parameters():
            param.requires_grad = grad

    @staticmethod
    def set_zero(model):
        for p in model.parameters():
            p.data.fill_(0)

    @staticmethod
    def grid_uniform(center, size, samples=1):
        return (torch.rand(samples, center.shape[0])-0.5)*size + center

    @staticmethod
    def generalized_sigmoid(x, b=1.0, a=0.0, k=1.0, m=0.0):
        # b: growth rate
        # a: left horizontal asymptote
        # k: right horizontal asymptote
        # m: start
        return (k-a) / (1. + torch.exp(-b * (x - m))) + a
