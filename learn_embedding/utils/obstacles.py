#!/usr/bin/env python

import torch


class Obstacles:
    @staticmethod
    def semi_circle(radius, center, rot=0, res=10):
        theta = torch.linspace(0, torch.pi, res)
        rot_mat = torch.tensor([[torch.cos(rot), -torch.sin(rot)], [torch.sin(rot), torch.cos(rot)]])

        return center + radius*torch.mm(torch.stack((theta.cos(), theta.sin()), axis=1), rot_mat)

    @staticmethod
    def square(center, a, b, res=10):
        y, x = torch.meshgrid(torch.linspace(0, b, res), torch.linspace(0, a, res))
        return center + torch.stack((torch.cat((x[0, :], x[-1, :], x[:, 0], x[:, -1])), torch.cat((y[0, :], y[-1, :], y[:, 0], y[:, -1]))), dim=1).unique(dim=0)
