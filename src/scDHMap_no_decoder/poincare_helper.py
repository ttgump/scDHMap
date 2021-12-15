import torch
import numpy as np

def sq_distances(x):
    sq_norms_x = torch.sum(x**2, dim=1, keepdim=True)
    sq_norms_y = torch.transpose(sq_norms_x, 0, 1)
    dotprods = torch.matmul(x, torch.transpose(x, 0, 1))
    d = sq_norms_x + sq_norms_y - 2. * dotprods
    d_scaled = 1+2*d/((1-sq_norms_x)*(1-sq_norms_y))
    return sq_norms_x, sq_norms_y, d_scaled

def hyp_distances(x, q=1, c=1.0):
    sq_norms_x, sq_norms_y, d_xy = sq_distances(x)
    hyp_dist = torch.acosh(d_xy)**q
    return sq_norms_x, d_xy, hyp_dist


def sq_distances_2(x,y):
    sq_norms_x = torch.sum(x**2, dim=1, keepdim=True)
    sq_norms_y = torch.transpose(torch.sum(y**2, dim=1, keepdim=True), 0, 1)
    dotprods = torch.matmul(x, torch.transpose(y, 0, 1))
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*||x||*||y||
    d = sq_norms_x + sq_norms_y - 2. * dotprods
    d_scaled = 1+2*d/((1-sq_norms_x)*(1-sq_norms_y))
    return sq_norms_x, sq_norms_y, d_scaled

def hyp_distances_2(x,y,q=1,c=1.0):
    sq_norms_x, sq_norms_y, d_xy = sq_distances_2(x,y)
    hyp_dist = torch.acosh(d_xy)**q
    return sq_norms_x, d_xy, hyp_dist

# poincare distance to orign
def poincare_norm(x):
    sq_norms_x = torch.sum(x**2, dim=1, keepdim=True)
    d = 1+2*sq_norms_x/(1-sq_norms_x+1e-10)
    return torch.acosh(d)