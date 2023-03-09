#!/usr/bin/env python

import os
import torch


class TorchHelper():
    @staticmethod
    def save(model, path):
        torch.save(model.state_dict(), os.path.join('', '{}.pt'.format(path)))

    @staticmethod
    def load(model, path):
        model.load_state_dict(torch.load(os.path.join('', '{}.pt'.format(path)), map_location=torch.device(model.device)))
        model.eval()

    @staticmethod
    def set_grad(model, grad):
        for param in model.parameters():
            param.requires_grad = grad

    @staticmethod
    def set_zero(model):
        for p in model.parameters():
            p.data.fill_(0)

    @staticmethod
    def grid_uniform(center, length, samples=1):
        a = [center[0] - length, center[1] - length]
        b = [center[0] + length, center[1] + length]
        return torch.cat((torch.FloatTensor(samples, 1).uniform_(a[0], b[0]), torch.FloatTensor(samples, 1).uniform_(a[1], b[1])), dim=1)
