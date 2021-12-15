import math
import torch
from torch.nn import functional as F
from torch.distributions import Normal, Independent
from numbers import Number
from torch.distributions.utils import _standard_normal, broadcast_all
from lorentzian_helper import *


class HyperbolicWrappedNorm(torch.distributions.Distribution):

    arg_constraints = {'_loc': torch.distributions.constraints.real,
                       '_scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real

    @property
    def mean(self):
        return self._loc

    @property
    def stddev(self):
        return self._scale

    def __init__(self, loc, scale, c=1., validate_args=None, softplus=False):
        self._loc, self._scale =  loc, scale
        self._base_dist = Normal(torch.zeros_like(scale).to(scale.device), scale)
        self._dim = loc.shape[1] - 1
        self.eps = 1e-6
        super(HyperbolicWrappedNorm, self).__init__()


    def sample(self):
        return self._sample_n(1).squeeze()


    def _sample_n(self, n):
        shape = torch.cat((torch.tensor([n]), torch.tensor(self._scale.shape)), dim=0)
        zn = torch.randn(torch.Size(shape)).to(self._scale.device)
        zn *= self._scale

        shape1 = [n, self._loc.shape[0], 1]
        z0 = torch.cat((torch.zeros(shape1).to(self._scale.device), zn), dim=-1)

        loc0 = self._lorentzian_orig(torch.Size(shape1), torch.Size(shape)).to(self._scale.device)
        tmp = torch.unsqueeze(self._loc, 0)

        shape2 = (n, 1, 1)
        zt = parallel_transport(z0, loc0, tmp.tile(shape2))
        z = exp_map(zt, tmp.tile(shape2))
        return z


    @staticmethod
    def _lorentzian_orig(s1, s0):
        x1 = torch.ones(s1)
        x0 = torch.zeros(s0)

        x_orig = torch.cat((x1, x0), dim=-1)

        return x_orig


    def log_prob(self, x):
        v = inv_exp_map(x, self._loc)
        tmp = lorentzian_product(v, v)
        x_norm = torch.sqrt(torch.clamp(tmp, min=self.eps))

#        x_norm = torch.clamp(x_norm, 0, 32)
        res = (self._dim - 1.0) * torch.log(torch.sinh(x_norm) / x_norm + self.eps)

        shape = self._scale.shape
        shape1 = [self._loc.shape[0], 1]

        loc0 = self._lorentzian_orig(shape1, shape).to(self._scale.device)
        v1 = parallel_transport(v, self._loc, loc0)
        xx = v1[..., 1:]
        log_base_prob = torch.sum(self._base_dist.log_prob(xx), dim=-1)

        return log_base_prob - res.squeeze()


    def prob(self, x):
        return torch.exp(self.log_prob(x))
