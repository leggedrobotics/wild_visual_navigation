#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.utils import Data
import copy
import torch
from torch import nn, distributions


class LinearBatchNorm(nn.Module):
    """
    An (invertible) batch normalization layer.
    This class is mostly inspired from this one:
    https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py
    """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, **kwargs):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)

            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        log_det = self.log_gamma - 0.5 * torch.log(var + self.eps)

        return y, log_det.expand_as(x).sum(1)

    def backward(self, x, **kwargs):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_det = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_det.expand_as(x).sum(1)


class LinearCouplingLayer(nn.Module):
    """
    Linear coupling layer.
        (i) Split the input x into 2 parts x1 and x2 according to a given mask.
        (ii) Compute s(x2) and t(x2) with given neural network.
        (iii) Final output is [exp(s(x2))*x1 + t(x2); x2].
    The inverse is trivially [(x1 - t(x2))*exp(-s(x2)); x2].
    """

    def __init__(
        self,
        input_size,
        mask,
        network_topology,
        conditioning_size=None,
        single_function=True,
    ):
        super().__init__()

        if conditioning_size is None:
            conditioning_size = 0

        if network_topology is None or len(network_topology) == 0:
            network_topology = [input_size]

        self.register_buffer("mask", mask)

        self.dim = input_size

        self.s = [
            nn.Linear(input_size + conditioning_size, network_topology[0]),
            nn.ReLU(),
        ]

        for i in range(len(network_topology)):
            t = network_topology[i]
            t_p = network_topology[i - 1]
            self.s.extend([nn.Linear(t_p, t), nn.ReLU()])

        if single_function:
            input_size = input_size * 2

        ll = nn.Linear(network_topology[-1], input_size)

        self.s.append(ll)
        self.s = nn.Sequential(*self.s)

        if single_function:
            self.st = lambda x: (self.s(x).chunk(2, 1))
        else:
            self.t = copy.deepcopy(self.s)
            self.st = lambda x: (self.s(x), self.t(x))

    def backward(self, x, y=None):
        mx = x * self.mask

        if y is not None:
            _mx = torch.cat([y, mx], dim=1)
        else:
            _mx = mx

        s, t = self.st(_mx)
        s = torch.tanh(s)  # Adding an activation function here with non-linearities

        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)

        log_abs_det_jacobian = -(1 - self.mask) * s

        return u, log_abs_det_jacobian.sum(1)

    def forward(self, u, y=None):
        mu = u * self.mask

        if y is not None:
            _mu = torch.cat([y, mu], dim=1)
        else:
            _mu = mu

        s, t = self.st(_mu)
        s = torch.tanh(s)

        x = mu + (1 - self.mask) * (u * s.exp() + t)

        log_abs_det_jacobian = (1 - self.mask) * s

        return x, log_abs_det_jacobian.sum(1)


class Permutation(nn.Module):
    """
    A permutation layer.
    """

    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        self.register_buffer("p", torch.randperm(in_ch))
        self.register_buffer("invp", torch.argsort(self.p))

    def forward(self, x, y=None):
        assert x.shape[1] == self.in_ch
        out = x[:, self.p]
        return out, 0

    def backward(self, x, y=None):
        assert x.shape[1] == self.in_ch
        out = x[:, self.invp]
        return out, 0


class SequentialFlow(nn.Sequential):
    """
    Utility class to build a normalizing flow from a sequence of base transformations.
    During forward and inverse steps, aggregates the sum of the log determinants of the Jacobians.
    """

    def forward(self, x, y=None):
        log_det = 0
        for module in self:
            x, _log_det = module(x, y=y)
            log_det = log_det + _log_det
        return x, log_det

    def backward(self, u, y=None):
        log_det = 0
        for module in reversed(self):
            u, _log_det = module.backward(u, y=y)
            log_det = log_det + _log_det
        return u, log_det

    def forward_steps(self, x, y=None):
        log_det = 0
        xs = [x]
        for module in self:
            x, _log_det = module(x, y=y)
            xs.append(x)
            log_det = log_det + _log_det
        return xs, log_det

    def backward_steps(self, u, y=None):
        log_det = 0
        us = [u]
        for module in reversed(self):
            u, _log_det = module.backward(u, y=y)
            us.append(u)
            log_det = log_det + _log_det
        return us, log_det


class LinearRnvp(nn.Module):
    """
    Main RNVP model, alternating affine coupling layers
    with permutations and/or batch normalization steps.
    """

    def __init__(
        self,
        input_size,
        coupling_topology,
        flow_n=2,
        use_permutation=False,
        batch_norm=False,
        mask_type="odds",
        conditioning_size=None,
        single_function=False,
        **kwargs
    ):
        super().__init__()

        self.register_buffer("prior_mean", torch.zeros(input_size))  # Normal Gaussian with zero mean
        self.register_buffer("prior_var", torch.ones(input_size))  # Normal Gaussian with unit variance

        if mask_type == "odds":
            mask = torch.arange(0, input_size).float() % 2
        elif mask_type == "half":
            mask = torch.zeros(input_size)
            mask[: input_size // 2] = 1
        else:
            assert False

        if coupling_topology is None:
            coupling_topology = [input_size // 2, input_size // 2]

        blocks = []

        for i in range(flow_n):
            blocks.append(
                LinearCouplingLayer(
                    input_size,
                    mask,
                    network_topology=coupling_topology,
                    conditioning_size=conditioning_size,
                    single_function=single_function,
                )
            )
            if use_permutation:
                blocks.append(Permutation(input_size))
            else:
                mask = 1 - mask

            if batch_norm:
                blocks.append(LinearBatchNorm(input_size))

        self.flows = SequentialFlow(*blocks)

    def logprob(self, x):
        return self.prior.log_prob(x)  # Compute log probability of the input at the Gaussian distribution

    @property
    def prior(self):
        return distributions.Normal(self.prior_mean, self.prior_var)  # Normal Gaussian with zero mean and unit variance

    def forward(self, data: Data):
        x = data.x
        z, log_det = self.flows.forward(x, y=None)
        log_prob = self.logprob(z)
        return {"z": z, "log_det": log_det, "logprob": log_prob}

    def backward(self, u, y=None, return_step=False):
        if return_step:
            return self.flows.backward_steps(u, y)
        return self.flows.backward(u, y)

    def sample(self, samples=1, y=None, return_step=False, return_logdet=False):
        u = self.prior.sample((samples,))
        z, d = self.backward(u, y=y, return_step=return_step)
        if return_logdet:
            d = self.logprob(u).sum(1) + d
            return z, d
        return z
