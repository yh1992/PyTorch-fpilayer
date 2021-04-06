"""
Core algorithms of the Fixed-Point Iteration Layers.

Ref: Younghan Jeon, Minsik Lee, and Jin Young Choi,
    "Differentiable Forward and Backward Fixed-point Iteration Layers,"
    IEEE Access, January 22, 2021.
License: GPLv3

Copyright (C) 2021 Younghan Jeon, Minsik Lee
This file is part of PyTorch-fpilayer.

PyTorch-fpilayer is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

PyTorch-fpilayer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyTorch-fpilayer. If not, see <http://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn


def default_term_cond(cnt, x, px, threshold=1e-10, epsilon=1e-30, normalized=True, display=False, minIter=0, maxIter=0):
    if normalized:
        val = (x[0] - px[0]).norm() ** 2 / (epsilon + px[0].norm() ** 2)
    else:
        val = (x[0] - px[0]).norm() ** 2
    if display and cnt % display == 0:
        print(f'{cnt}: {val}')
    if cnt <= minIter:
        return False
    elif cnt >= maxIter & maxIter > 0:
        return False
    else:
        return val > threshold


def create_independent_nodes(x, requires_grad=None):
    # x must be a sequence of tensors
    # if requires_grad is None, it is determined based on that of each element of x
    # if it is a tuple or list, that sets the requires_grad for each element
    if type(requires_grad) in (tuple, list):
        return tuple(item if not item.requires_grad and not rg else
                     item.detach().requires_grad_(requires_grad=rg) for item, rg in zip(x, requires_grad))
    elif requires_grad is None:
        return tuple(item if not item.requires_grad else
                     item.detach().requires_grad_(requires_grad=item.requires_grad) for item in x)
    else:
        return tuple(item if not item.requires_grad and not requires_grad else
                     item.detach().requires_grad_(requires_grad=requires_grad) for item in x)


def construct_independent_graph(fun, x, y, _x=None, _y=None, requires_grad__y=None):
    # create independent nodes
    if _x is None:
        _x = create_independent_nodes(x, requires_grad=True)
    if _y is None:
        _y = create_independent_nodes(y, requires_grad=requires_grad__y)
    _x, _y = tuple(_x), tuple(_y)

    # compute fun on a new graph
    with torch.enable_grad():
        _f = fun(_x, _y)
    _f = tuple(f_.requires_grad_() for f_ in _f)
    return _x, _y, _f


def _combine_upstream_gradient(_f):
    def _combine(x, uy):
        return tuple(u_.flatten().dot(f_.flatten()) for u_, f_ in zip(uy, _f))
    return _combine


def _chain_rule_on_independent_graph(upstream_grad, _f, x, _x, y=tuple(), _y=tuple()):
    _u = create_independent_nodes(upstream_grad)
    _combine = _combine_upstream_gradient(_f)
    return partial(_combine, x=x, y=upstream_grad + y, _x=_x, _y=_u + _y)


class PartialOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *xypmisc):
        _dfdx = xypmisc[-1][-1]
        dfdx = tuple(df_.detach() for df_ in _dfdx)
        ctx.save_for_backward(*xypmisc[:-1], *dfdx)
        ctx.others = xypmisc[-1]
        return dfdx

    @staticmethod
    def backward(ctx, *dlddfdx):
        _xyp, _dfdx = ctx.others
        xyp = ctx.saved_tensors[:-len(dlddfdx)]

        xyp = tuple(xyp_ for xyp_, need in zip(xyp, ctx.needs_input_grad) if need)
        _xyp = tuple(xyp_ for xyp_, need in zip(_xyp, ctx.needs_input_grad) if need)
        dldxy = _chain_rule_on_independent_graph(dlddfdx, _dfdx, xyp, _xyp)
        dldxy = list(dldxy)
        dldxy = tuple(dldxy.pop(0) if need else None for need in ctx.needs_input_grad)
        return dldxy


def partial(fun, x, y, param=(), _x=None, _y=None):
    """
    partial derivatives of fun w.r.t x
    note that the differentiation is strictly done for the parameters of fun,
    not following the entire computational graph of pytorch
    creates another independent computational graph to compute the result
    :param fun: function to calculate a partial derivative. must receive two tuples of tensors
    e.g., fun(x, y)
    fun can be an nn.Module. See fixed_point_iteration for explanation. If it is not an nn.Module and it has internal parameters,
    they have to be included as param. Other than x, y, and the internal parameters, fun must not depend on any tensors connected
    to other computational graphs.
    :param x: a sequence of variables to differentiate
    :param y: a sequence of miscellaneous variables. It can be an empty tuple.
    (not differentiated but connected to the computational graph)
    :param _x: see the below explanation
    :param _y: see the below explanation
    you might precompute some parts of the independent computational graph and include it in the definition of fun
    _x and _y are the starting leaf variables in that graph corresponding to x and y, respectively
    :return: partial derivatives
    """

    # replace lists to tuples
    x, y, param = tuple(x), tuple(y), tuple(param)

    if issubclass(type(fun), nn.Module):
        assert not param
        param = tuple(fun.parameters())

    # construct an independent graph
    _x, _y, _f = construct_independent_graph(fun, x=x, y=y, _x=_x, _y=_y)

    # derive partial derivatives in the graph. we want retain the independent graph since some parts might have been predefined outside
    _dfdx = torch.autograd.grad(outputs=_f, inputs=_x, retain_graph=True, create_graph=torch.is_grad_enabled(), allow_unused=True)
    # Nones are replaced to zeros
    _dfdx = tuple(torch.zeros_like(x_) if df_ is None else df_ for df_, x_ in zip(_dfdx, _x))

    # attach the independent graph as a layer in the existing graph
    return PartialOp.apply(*x, *y, *param, (_x + _y + param, _dfdx))


def _vectorize(x):
    return torch.cat([item.flatten() for item in x]).reshape(-1, 1)


def _backward_operator(_g, _x, _y, _param):
    nx, ny = len(_x), len(_y)
    _yp = _y + _param

    def _operator(c, xydldx):
        x = xydldx[:nx]
        y = xydldx[nx:nx + ny]
        dldx = xydldx[nx + ny:]
        dhdx = _chain_rule_on_independent_graph(c, _g, x, _x, y + _param, _yp)
        return tuple(dh_ + dl_ for dh_, dl_ in zip(dhdx, dldx))
    return _operator


class FixedPointIterationOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *yparammisc):
        ny, x0, g, forward_cond, backward_cond, m_anderson, eps = yparammisc[-1]
        px = tuple(item.detach() for item in x0)
        y = yparammisc[:ny]
        x = g(px, y)
        if m_anderson > 0:
            gx = [x]
            f = _vectorize(x) - _vectorize(px)
            df = []
        cnt = 0
        while True:

            px = x
            x = g(x, y)

            if m_anderson > 0:
                pf = f
                gx.append(x)
                f = _vectorize(x) - _vectorize(px)
                df.append(f - pf)
                if len(df) > m_anderson:
                    gx = gx[1:]
                    df = df[1:]

                dfc = torch.cat(df, 1)
                dfct = dfc.t()
                A = dfct @ dfc + torch.eye(n=m_anderson, out=dfc.new(m_anderson, m_anderson)) * eps
                gamma = (dfct @ f).solve(A).solution.flatten()
                if gamma.numel() > 1:
                    alpha = (gamma[[0]],) + (gamma[1:] - gamma[:-1]).split(1) + (1 - gamma[[-1]],)
                else:
                    alpha = (gamma[[0]], 1 - gamma[[-1]])
                x = tuple(torch.stack([alphai * gxi for alphai, gxi in zip(alpha, gx_elem)]).sum(0) for gx_elem in zip(*gx))

            cnt += 1
            if not forward_cond(cnt, x, px):
                break

        ctx.save_for_backward(*x, *yparammisc[:-1])
        ctx.others = (len(x0), ny, g, forward_cond, backward_cond, m_anderson, eps)
        return tuple(x)

    @staticmethod
    def backward(ctx, *dldx):
        nx, ny, g, forward_cond, backward_cond, m_anderson, eps = ctx.others
        x = ctx.saved_tensors[:nx]
        y = ctx.saved_tensors[nx:nx+ny]
        # parameters of g are considered to be already on an independent graph,
        # since they are not dependent to any other entities.
        _param = ctx.saved_tensors[nx+ny:]

        # pre-compute some parts of the independent graph for partial
        _x, _y, _g = construct_independent_graph(g, x=x, y=y)

        c = fixed_point_iteration(x + y + dldx, dldx, _backward_operator(_g, _x, _y, _param), backward_cond, forward_cond, m_anderson, eps, _param)

        yp = tuple(yp_ for yp_, need in zip(y + _param, ctx.needs_input_grad) if need)
        _yp = tuple(yp_ for yp_, need in zip(_y + _param, ctx.needs_input_grad) if need)
        dldyp = _chain_rule_on_independent_graph(c, _g, yp, _yp, x, _x)
        dldyp = list(dldyp)
        dldyp = tuple(dldyp.pop(0) if need else None for need in ctx.needs_input_grad)
        return dldyp


def fixed_point_iteration(y, x0, g, forward_cond=default_term_cond, backward_cond=default_term_cond, m_anderson=0, eps=1e-15, param=()):
    """
    fixed_point_iteration_layer
    :param y: input tensors of the layer
    :param x0: initial tensors of the fixed-point-iteration variable
    :param g: fixed-point operator. It can be a function or an nn.Module
    g must receive all the tensors that need to be connected to the computational graph as arguments explicitly.
    g must be in the form of g(x, y). If g is an nn.Module, it can have some internal parameters that are independent
    from the outside world. Note that other than x, y, and the internal parameters, g must not depend on tensors derived
    from any outside parameters, i.e., g must be isolated from the outside world except for x, y.
    :param forward_cond: loop condition for forward (if this is not satisfied, terminates the fixed-point iteration)
    :param backward_cond: loop condition for backward (if this is not satisfied, terminates the fixed-point iteration)
    :param m_anderson: m for Anderson acceleration
    :param eps: eps for approximate pseudo-inverse
    :param param: If g is a function and there are any independent parameters in g, they should be included here as a
    tuple. Note that they are considered independent to x and y in computing g.
    :return: x - converged x
    """
    ny = len(y)

    if issubclass(type(g), nn.Module):
        assert not param
        param = tuple(g.parameters())

    return FixedPointIterationOp.apply(*y, *param, (ny, x0, g, forward_cond, backward_cond, m_anderson, eps))


# Fixed-point iteration module (see the autograd.Function version)
class FixedPointIteration(nn.Module):
    def __init__(self, g, forward_cond=default_term_cond, backward_cond=default_term_cond, m_anderson=0, eps=1e-15, param=()):
        super(FixedPointIteration, self).__init__()
        self.g = g
        self.forward_cond = forward_cond
        self.backward_cond = backward_cond
        self.m_anderson = m_anderson
        self.eps = eps
        self.param = param

    def forward(self, y, x0):
        return fixed_point_iteration(y, x0, self.g, self.forward_cond, self.backward_cond, self.m_anderson, self.eps, self.param)

