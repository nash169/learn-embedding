import torch
import matplotlib.pyplot as plt
from learn_embedding.utils.torch_helper import TorchHelper


def sigmoid_k(x, a=0.0, b=1.0, k=1.0, m=0.0):
    return (k-a) / (1. + torch.exp(-b * (x - m))) + a


x = torch.arange(-1, 1, 0.01)

# y = sigmoid_k(x, a=1.0, k=0.0, b=10.0, m=-5.0)

y = TorchHelper.generalized_sigmoid(x, b=1.0e6, m=0.25)

plt.plot(x, y)
plt.show(block=False)
