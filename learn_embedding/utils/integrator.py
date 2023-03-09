#!/usr/bin/env python

import torch


class Integrator():
    @staticmethod
    def first_order(dynamics, x0, T, dt):
        dim = x0.shape[1]
        steps = int(T/dt)
        x = torch.zeros(steps, x0.shape[0], 2*x0.shape[1]).to(x0.device)
        x[0, :, :] = torch.cat((x0, torch.zeros(1, dim).to(x0.device)), axis=1)

        for i in range(steps-1):
            x[i, :, dim:] = dynamics(x[i, :, :dim])
            x[i+1, :, :dim] = x[i, :, :dim] + x[i, :, dim:]*dt

        return x

    @staticmethod
    def second_order(dynamics, x0, T, dt):
        dim = int(x0.shape[1]/2)
        steps = int(T/dt)
        x = torch.zeros(steps, x0.shape[0], x0.shape[1]).to(x0.device)
        x[0, :, :] = x0

        for i in range(steps-1):
            x[i+1, :, dim:] = x[i, :, dim:] + dynamics(x[i, :, :])*dt
            x[i+1, :, :dim] = x[i, :, :dim] + x[i+1, :, dim:]*dt

        return x
