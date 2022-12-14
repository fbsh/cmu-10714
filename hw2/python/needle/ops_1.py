"""Operatpr table."""
# Global operator table.
from functools import reduce
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(input_a, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * (- lhs / (rhs * rhs))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        input_shape = a.shape 
        input_shape_len = len(input_shape)
        transpose_axes = list(range(len(input_shape)))
        if self.axes is None: 
            transpose_axes[-2] = input_shape_len - 1
            transpose_axes[-1] = input_shape_len - 2
        else: 
            transpose_axes[self.axes[0]] = self.axes[1]
            transpose_axes[self.axes[1]] = self.axes[0]  
        self.transpose_axes = transpose_axes
        return array_api.transpose(a, transpose_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        input_shape_len = len(input_shape) - 1
        self.reduce_dim = []
        for idx in range(len(out_grad.shape)-1, -1, -1): 
            if input_shape_len < 0: 
                self.reduce_dim.append(idx)
                continue
            broadcast_dim_size = self.shape[idx]
            input_dim_size = input_shape[input_shape_len]
            
            if broadcast_dim_size != input_dim_size: 
                self.reduce_dim.append(idx)
            input_shape_len -= 1
        return reshape(summation(out_grad, tuple(self.reduce_dim)), input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        broadcast_shape = list(input_shape)
        if self.axes: 
            for i in self.axes: 
                broadcast_shape[i] = 1
        else: 
            broadcast_shape = [1 for _ in range(len(broadcast_shape))] # (1, 1, 1, 1)
        out_grad = reshape(out_grad, broadcast_shape)
        return out_grad * array_api.ones(input_shape, dtype=array_api.float32)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_tmp_grad = matmul(out_grad, transpose(rhs))
        rhs_tmp_grad = matmul(transpose(lhs), out_grad)
        if lhs_tmp_grad.shape != lhs.shape: 
            # Need Reduce
            lhs_tmp_grad = summation(lhs_tmp_grad, axes=tuple(range(len(lhs_tmp_grad.shape) - 2)))
        if rhs_tmp_grad.shape != rhs.shape: 
            # Need Reduce
            rhs_tmp_grad = summation(rhs_tmp_grad, axes=tuple(range(len(rhs_tmp_grad.shape) - 2)))
        return lhs_tmp_grad, rhs_tmp_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return - out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        relu_mask = a > 0 
        return a * relu_mask
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        relu_mask = Tensor((node.inputs[0].cached_data > 0).astype(array_api.float32))
        return out_grad * relu_mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_val = array_api.max(Z, axis=self.axes, keepdims=True)
        no_keep_dims_max_val = array_api.max(Z, axis=self.axes, keepdims=False)
        exp_val = array_api.exp(Z - max_val)
        out = array_api.log(array_api.sum(exp_val, axis=self.axes)) + no_keep_dims_max_val
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_tensor = node.inputs[0].cached_data
        max_val = array_api.max(input_tensor, axis=self.axes, keepdims=True)
        exp_val = array_api.exp(input_tensor - max_val)
        sum_val = array_api.sum(exp_val, axis=self.axes)

        log_grad = out_grad.cached_data / sum_val 

        input_shape = input_tensor.shape
        broadcast_shape = list(input_shape)
        if self.axes: 
            for i in self.axes: 
                broadcast_shape[i] = 1
        else: 
            broadcast_shape = [1 for _ in range(len(broadcast_shape))] # (1, 1, 1, 1)
        reduce_sum_grad = array_api.reshape(log_grad, tuple(broadcast_shape))
        exp_grad = exp_val * reduce_sum_grad
        return Tensor(exp_grad)        
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
