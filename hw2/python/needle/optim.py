"""Optimization module"""
import needle as ndl
import numpy as np
import needle.init as init


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
        #print('1 global tensors', ndl.autograd.TENSOR_COUNTER)

    def step(self):
        ### BEGIN YOUR SOLUTION
        #print('2 global tensors', ndl.autograd.TENSOR_COUNTER)
        for p in self.params:
            if p not in self.u:
                self.u[p] = init.zeros(*p.shape, device=p.device, dtype=p.dtype, requires_grad=False)
            self.u[p].data = self.momentum * self.u[p].data + (1. - self.momentum) * (p.grad.data + self.weight_decay * p.data)
            p.data = p.data - self.lr * self.u[p].data
        #print('3 global tensors', ndl.autograd.TENSOR_COUNTER)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
        #print('1 global tensors', ndl.autograd.TENSOR_COUNTER)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        #print('2 global tensors', ndl.autograd.TENSOR_COUNTER)
        for p in self.params:
            if p not in self.m:
                self.m[p] = init.zeros(*p.shape, device=p.device, dtype=p.dtype, requires_grad=False)
                self.v[p] = init.zeros(*p.shape, device=p.device, dtype=p.dtype, requires_grad=False)
            
            delta = p.grad.data + self.weight_decay * p.data
            self.m[p].data = self.beta1 * self.m[p].data + (1 - self.beta1) * delta
            self.v[p].data = self.beta2 * self.v[p].data + (1 - self.beta2) * delta * delta

            m = self.m[p].data / (1 - self.beta1**self.t) if self.t > 0 else self.m[p].data
            v = self.v[p].data / (1 - self.beta2**self.t) if self.t > 0 else self.v[p].data
            p.data = p.data - self.lr * m / (v**0.5 + self.eps)
        #print('3 global tensors', ndl.autograd.TENSOR_COUNTER)
        ### END YOUR SOLUTION
