import torch
from torch.nn import functional as F

EPS = 1e-6

def lorentzian_product(x, y, keepdim=True):
    y0, y1 = torch.split(y, [1, y.shape[-1] - 1], dim=-1)
    y_neg_first = torch.cat((-y0, y1), dim=-1)
    return torch.sum(x * y_neg_first, dim=-1, keepdim=keepdim)


def lorentzian_product_mat(x, y):
    y0, y1 = torch.split(y, [1, y.shape[-1] - 1], dim=-1)
    y_neg_first = torch.cat((-y0, y1), dim=-1)
    return torch.matmul(x, y_neg_first.T)


def parallel_transport(x, m1, m2):
    alpha = -lorentzian_product(m1, m2)
    coef = lorentzian_product(m2, x) / (alpha + 1.0)

    return x + coef * (m1 + m2)


def lorentz_distance(x, y):
    xy_inner = lorentzian_product(x, y)
    return torch.acosh(torch.clamp(-xy_inner, min=1.+EPS))


def lorentz_distance_mat(x, y):
    xy_inner = lorentzian_product_mat(x, y)
    return torch.acosh(torch.clamp(-xy_inner, min=1.+EPS))


def exp_map(x, mu, c=1.):
    res = lorentzian_product(x, x)
    res = torch.sqrt(torch.clamp(res, min=EPS))
    res = torch.clamp(res, 0, 32)

    return torch.cosh(res) * mu + torch.sinh(res) * x / res


def inv_exp_map(x, mu):
    alpha = -lorentzian_product(x, mu) 
    alpha = torch.clamp(alpha, min=1.+EPS)

    tmp = lorentz_distance(x, mu) / torch.sqrt(alpha+1) / torch.sqrt(alpha-1)

    return tmp * (x - alpha * mu)


def lorentz2poincare(x):
    d = x.size(-1) -1
    return (x.narrow(-1, 1, d) * 1) / (x.narrow(-1, 0, 1) + 1)


def poincare2lorentz(x):
    x_norm_square = torch.sum(x * x, dim=1, keepdim=True)
    return torch.cat((1 + x_norm_square, 2 * x), dim=1) / (1 - x_norm_square + EPS)
