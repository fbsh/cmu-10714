"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


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
                in_grad.append(init.zeros_like(value))
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
        return a + numpy.float32(self.scalar)

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
        return a * numpy.float32(self.scalar)

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
        return a ** numpy.float32(self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs
        if isinstance(a, tuple):
          a = a[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))
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
        a, b = node.inputs
        return (out_grad * (b ** -1), out_grad * a * (b ** -2) * numpy.float32(-1))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / numpy.float32(self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * numpy.float32(numpy.float32(1.0)/self.scalar),)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        _axes = list([x for x in range(len(a.shape))])
        if self.axes is not None:
          _axes[self.axes[0]], _axes[self.axes[1]] = \
          _axes[self.axes[1]], _axes[self.axes[0]]
        else:
          _axes[-1], _axes[-2] = \
          _axes[-2], _axes[-1]
        return a.permute(_axes).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        s = node.inputs[0].shape
        axis = []
        idx1 = len(s) - 1
        idx2 = len(self.shape) - 1
        while(idx1 >= 0 and idx2 >= 0):
          if(s[idx1] != self.shape[idx2] and 
            (s[idx1] == 1 or self.shape[idx2] == 1)):
            axis.append(idx2)
          idx1 -= 1
          idx2 -= 1
        while(idx2 >= 0):
          axis.append(idx2)
          idx2 -= 1
        return reshape(summation(out_grad, tuple(axis)), s)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        s1 = node.inputs[0].shape
        s = list(s1)
        if self.axes is None:
          axis = list(range(0,len(s1)))
        elif isinstance(self.axes, int):
          axis = (self.axes,)
        else:
          axis = self.axes
        for i in axis:
          s[i] = 1
        return broadcast_to(reshape(out_grad, s), s1)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        X_ = matmul(out_grad, transpose(W))
        W_ = matmul(transpose(X), out_grad)
        if len(X_.shape) > len(X.shape):
          X_ = summation(X_, tuple(range(0, len(X_.shape)-len(X.shape))))
        if len(W_.shape) > len(W.shape):
          W_ = summation(W_, tuple(range(0, len(W_.shape)-len(W.shape))))
        return (X_, W_)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * numpy.float32(-1.0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * numpy.float32(-1.0),)
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
        a = node.inputs[0]
        return out_grad * (a ** -1)
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
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        res = array_api.maximum(a, 0)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.numpy()
        a = numpy.multiply((a> 0), 1, dtype=node.dtype)
        res = out_grad * Tensor(NDArray(a, device=node.device), requires_grad=False, device=node.device)
        return res
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if(isinstance(axes, int)):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        y_squeezed = Z.max(axis=self.axes, keepdims=True)
        y = array_api.broadcast_to(y_squeezed, Z.shape)
        
        Z_ = array_api.log(array_api.summation(array_api.exp(Z-y), axis=self.axes, keepdims=True))
        # print('Z shape: ', Z_.shape)
        # print('y shape: ', y_squeezed.shape)
        res = (Z_ + y_squeezed).compact()
        if self.axes != None:
            new_axis = list()
            for x in res.shape:
                if x != 1:
                    new_axis.append(x)
            res = array_api.reshape(res, new_axis)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs
        if isinstance(Z, tuple):
          Z = Z[0]
        before_s = Z.shape
        s = list(Z.shape)
        axis = self.axes
        if self.axes == None:
          axis = [i for i in range(len(before_s))]
        # elif isinstance(self.axes, int):
        #     axis = (self.axes,)
        # print("axis: ", axis)
        for i in axis:
          s[i] = 1
        s = tuple(s)
        Z = Z - broadcast_to(reshape(node,s), before_s)
        expz = exp(Z)
        expsum = broadcast_to(reshape(summation(expz, self.axes),s), before_s)
        tmpa = broadcast_to(reshape(out_grad,s), before_s)
        tmpb = expz / expsum
        res = tmpa * tmpb
        return res
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        res = a.tanh()
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        tmp = -(tanh(a) ** 2) + numpy.float32(1.0)
        res = out_grad * tmp
        return res
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)

class Sigmoid(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (numpy.float32(1.0) + array_api.exp(-a))**-1
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        tmp = sigmoid(a) * (numpy.float32(1.0) - sigmoid(a))
        res = out_grad * tmp
        return res
        ### END YOUR SOLUTION

def sigmoid(a):
    return Sigmoid()(a)

class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        arrays = list()
        for x in args:
            arrays.append(x.numpy())
        res = numpy.stack(arrays, axis=self.axis)
        # print("stack shape: ", res.shape)
        return NDArray(res,device=args[0].device)
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_tuple =  split(out_grad, axis=self.axis)
        res = list()
        for x in out_tuple:
            s = []
            for i in range(len(x.shape)):
                if i != self.axis:
                    s.append(x.shape[i])
            s = tuple(s)
            res.append(reshape(x, s))
        return make_tuple(*res)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        tmp = A.numpy()
        res = numpy.split(tmp, tmp.shape[self.axis], axis=self.axis)
        # print("shape: ", res[0].shape)
        return tuple([NDArray(x,device=A.device) for x in res])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        res = reshape(stack(out_grad, self.axis), node.inputs[0].shape)
        return res
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.dilate(self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.undilate(self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

class Pad(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes
    
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.pad(self.axes)
        ### END YOUR SOLUTION
    
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

def pad(a, axes):
    return Pad(axes)(a)


class Permute(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.permute(self.axes).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        axis = self.axes
        idxs = [axis.index(i) for i in range(len(axis))]
        return permute(out_grad, idxs)
        ### END YOUR SOLUTION

def permute(a, axes):
    return Permute(axes)(a)


class SparseMatrixMultiply(TensorOp):
    def __init__(self, sparse_matrix: array_api.SparseMatrix):
        self.sparse_matrix = sparse_matrix
        self.sparse_matrix_T = array_api.sparse_transpose(sparse_matrix)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.spmm(self.sparse_matrix, a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return spmm(self.sparse_matrix_T, out_grad)
        ### END YOUR SOLUTION

def spmm(sparse_matrix, a):
    return SparseMatrixMultiply(sparse_matrix)(a)

class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # implement convolution
        # A shape is (N, H, W, C_in)
        # B shape is (K, K, Cin, C_out)
        # output shape is (N,H_out,W_out,C_out)
        A_pad = A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        N, H, W, C_in = A_pad.shape
        K, _, _, C_out = B.shape
        H_out = int((H - K) / self.stride + 1)
        W_out = int((W - K) / self.stride + 1)
        Ns, Hs, Ws, Cs = A_pad.strides

        inner_dim = K * K * C_in
        A_im2coled = A_pad.as_strided(shape = (N, H_out, W_out, K, K, C_in),
                                        strides = (Ns, self.stride* Hs, self.stride* Ws, Hs, Ws, Cs))\
                                            .compact()\
                                            .reshape((N, H_out, W_out, inner_dim))
        # print("A shape: ", A_pad.shape)
        # print("A_im2coled shape: ", A_im2coled.shape)
        # print("B shape: ", B.shape)
        B_reshaped = B.compact().reshape((inner_dim, C_out)).compact()
        # print("B_reshaped shape: ", B_reshaped.shape)
        out = NDArray.make((N,H_out,W_out,C_out), device=A.device)
        out.fill(0.0)
        for i in range(N):
            for j in range(H_out):
                for k in range(W_out):
                    left = A_im2coled[i,j,k,:].compact().reshape((1, inner_dim)).compact()
                    mult_res = (left @ B_reshaped).compact()
                    out[i,j,k,:] = mult_res.reshape((1,1,1,C_out)).compact()

        out = out.compact().reshape((N,H_out,W_out,C_out)).compact()
        # print("conv shape: ", out.shape)
        # print("-----------------------\n")
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs[0], node.inputs[1]
        K = W.shape[0]
        new_pad = K - self.padding - 1
        dX = conv(out_grad.dilate((1,2),self.stride-1), W.flip((0,1)).transpose(), padding=new_pad)
        # print("dX shape: ", dX.shape)
        # print("x shape: ", X.shape)
        dW = conv(X.permute((3,1,2,0)), out_grad.dilate((1,2),self.stride-1).permute((1,2,0,3)), padding=self.padding).permute((1,2,0,3))
        return dX, dW
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



