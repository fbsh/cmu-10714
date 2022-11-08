"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    #def __str__(self):
    #    return str(self.__class__) + str(self.__dict__)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(self.out_features, 1, device=device, dtype=dtype, requires_grad=bias).reshape((1, self.out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight
        if self.bias.requires_grad:
            return y + ops.broadcast_to(self.bias, y.shape)
        return y
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        from functools import reduce
        batch = X.shape[0]
        dims = reduce(lambda x,y : x*y, X.shape[1:])
        return ops.reshape(X, (batch, dims))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #print('---->')
        #print(x)
        for m in self.modules:
            #print(m)
            x = m(x)
        #print('<----seq')
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n = logits.shape[1]        
        losses = ops.logsumexp(logits, axes = 1) - ops.summation(logits * init.one_hot(n, y), axes = 1)
        return ops.summation(losses / losses.shape[0]) 
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_size = x.shape[0]
            mean = ops.summation(x, axes = 0) / batch_size

            var = ops.summation((x - ops.broadcast_to(mean, x.shape))**2, axes = 0) / batch_size

            #mean2 = ops.summation(x*x, axes = 0) / batch_size
            #var = mean2 - mean * mean

            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var.data

            y = (x - ops.broadcast_to(mean, x.shape)) / ops.broadcast_to((var + self.eps)**0.5, x.shape) 
            return y * ops.broadcast_to(self.weight, y.shape) + ops.broadcast_to(self.bias, y.shape)
        else:
            y = (x - ops.broadcast_to(self.running_mean, x.shape)) / ops.power_scalar(ops.broadcast_to(self.running_var + self.eps, x.shape), 0.5)
            return y * ops.broadcast_to(self.weight, y.shape) + ops.broadcast_to(self.bias, y.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert self.dim == x.shape[1]
        mean = ops.summation(x, axes = 1) / self.dim
        var = ops.summation((x - ops.broadcast_to(mean.reshape((x.shape[0], 1)), x.shape))**2, axes=1) / self.dim
        var = ops.broadcast_to(var.reshape((x.shape[0], 1)), x.shape)
        y = (x - ops.broadcast_to(mean.reshape((x.shape[0], 1)), x.shape)) / (var + self.eps)**0.5
        return y * ops.broadcast_to(self.weight, y.shape) + ops.broadcast_to(self.bias, y.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x

        return x * init.randb(*x.shape, p = 1 - self.p) / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #print(self.fn)
        return self.fn(x) + x
        ### END YOUR SOLUTION



