"""Optimization module"""
import needle as ndl
import numpy as np

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
        self.lr = np.float32(lr)
        self.momentum = np.float32(momentum)
        self.u = {}
        self.weight_decay = np.float32(weight_decay)

    def step(self):
        ### BEGIN YOUR SOLUTION
        for i, param in enumerate(self.params):
          delta = param.grad
          if self.weight_decay - np.float32(0) > 1e-5:
            delta = delta + self.weight_decay * param
          if self.momentum - np.float32(0) > 1e-5:
            ut = self.u.get(i)
            if ut != None:
              ut1 = self.momentum * ut + (np.float32(1)-self.momentum) * delta
            else:
              ut1 = (np.float32(1)-self.momentum) * delta
          else:
            ut1 = delta
          param = param - self.lr * ut1
          self.u[i] = ut1.detach()
          self.params[i].data = param
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped
            


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
        self.lr = np.float32(lr)
        self.beta1 = np.float32(beta1)
        self.beta2 = np.float32(beta2)
        self.eps = np.float32(eps)
        self.weight_decay = np.float32(weight_decay)
        self.t = np.float32(0)

        self.u = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t = self.t + np.float32(1)
        for i, param in enumerate(self.params):
          delta = param.grad + self.weight_decay * param
          ut = self.u.get(i)
          vt = self.v.get(i)
          if ut != None and vt != None:
            ut1 = self.beta1 * ut + (np.float32(1)-self.beta1) * delta
            vt1 = self.beta2 * vt + (np.float32(1)-self.beta2) * (delta**2)
          else:
            ut1 = (np.float32(1)-self.beta1) * delta
            vt1 = (np.float32(1)-self.beta2) * (delta**2)
          self.u[i] = ut1.detach()
          self.v[i] = vt1.detach()
          ut1 = ut1 / (np.float32(1) - np.power(self.beta1, self.t, dtype=self.beta1.dtype))
          vt1 = vt1 / (np.float32(1) - np.power(self.beta2, self.t, dtype=self.beta2.dtype))
          param = (param - self.lr * ut1/(vt1 ** 0.5 + self.eps))
          self.params[i].data = param
        ### END YOUR SOLUTION
