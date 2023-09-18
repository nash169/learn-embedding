import torch
import torch.nn as nn
import faiss
import faiss.contrib.torch_utils


def bump_above(x, max, decay=1.0):
    bump = torch.zeros_like(x)
    bump[x.abs() < max] = x[x.abs() < max].square().sub(max.square()).pow(-1).mul(decay).exp().div(max.square().pow(-1).mul(-decay).exp())
    return bump


def bump_below(x, min, decay=1.0):
    bump = torch.zeros_like(x)
    bump[x.abs() > min] = x[x.abs() > min].square().sub(min.square()).pow(-1).mul(decay).exp().div(min.square().pow(-1).mul(-decay).exp())
    return bump


class BumpKNN(nn.Module):
    def __init__(self, data, radius=1.0, decay=1.0):
        super(BumpKNN, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.radius = torch.tensor(radius).to(device)
        self.decay = torch.tensor(decay).to(device)

        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.knn = faiss.GpuIndexIVFFlat(res, data.shape[1], 1, faiss.METRIC_L2)  # as for gmm think about knn in phase space
        else:
            self.knn = faiss.IndexFlatL2(data.shape[1])

        self.knn.train(data)
        self.knn.add(data)

    def forward(self, x):
        return bump_above(self.knn.search(x, 1)[0].sqrt().squeeze(), self.radius, self.decay)
