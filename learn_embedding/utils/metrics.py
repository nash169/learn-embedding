import torch


def metric_exp(y, y_obs, sigma, eta=1):
    d = y-y_obs
    k = eta*torch.exp(-0.5*torch.sum(d.pow(2), dim=1) /
                      sigma ** 2).unsqueeze(1).unsqueeze(2)
    return torch.bmm(d.unsqueeze(2), d.unsqueeze(1)) * k.pow(2)/np.power(sigma, 4) + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)


def metric_infty(y, y_obs, a, b, r=0):
    d = y-y_obs
    k = (torch.norm(d, dim=1) - r).unsqueeze(1).unsqueeze(2)
    k = torch.exp(a/(b*torch.pow(k, b)))
    return torch.bmm(d.unsqueeze(2), d.unsqueeze(1)) * (k-1) + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)
