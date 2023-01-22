"""The module.
"""
from typing import List
from needle.autograd import Tensor, NDArray
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

def get_linear(X, h, W_ih, b_ih, W_hh, b_hh, hidden_size):
    Xw = X @ W_ih
    hw = h @ W_hh
    bih = ops.broadcast_to(ops.reshape(b_ih, (1, hidden_size)), Xw.shape)
    bhh = ops.broadcast_to(ops.reshape(b_hh, (1, hidden_size)), hw.shape)
    Xw = Xw + bih
    hw = hw + bhh
    return Xw + hw

def split_section(X, axis, hidden_size):
    if len(X.shape) == 2:
        X = ops.reshape(X, (X.shape[0], 4, hidden_size))
        res = [ops.reshape(i, (X.shape[0], hidden_size)) for i in ops.split(X, axis)]
    else:
        X = ops.reshape(X, (4, hidden_size))
        res = [ops.reshape(i, (hidden_size,)) for i in ops.split(X, axis)]
    return res
    
    return X

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


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True)
        self.bias = init.zeros(out_features,1, device=device, dtype=dtype, requires_grad=True)
        if bias:
          self.bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
        self.bias = ops.reshape(self.bias, (1, out_features))
        self.weight = Parameter(self.weight)
        self.bias = Parameter(self.bias)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        broadcast_bias = ops.broadcast_to(self.bias, X.shape[:-1]+ (self.out_features,))
        res = X @ self.weight + broadcast_bias
        return res
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        prod = 1
        if len(X.shape) == 1:
          return ops.reshape(X, (X.shape[0],1))
        if len(X.shape) == 2:
          return X
        for dim in X.shape[1:]:
          prod = prod * dim
        return ops.reshape(X, (X.shape[0], prod))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.sigmoid(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
          x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        r, n = logits.shape
        zy = ops.summation(logits * init.one_hot(n, y, device=logits.device),axes=1)
        res = ops.summation(ops.logsumexp(logits, (1,)) - zy) / np.float32(r)
        return res
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        self.bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        self.weight = Parameter(self.weight)
        self.bias = Parameter(self.bias)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, n = x.shape
        m = np.float32(self.momentum)
        im = np.float32(1)-np.float32(self.momentum)
        batch = np.float32(batch)

        cur_mean = ops.reshape(ops.summation(x, axes=0) / batch,(1,n))
        ex = ops.broadcast_to(cur_mean, x.shape)
        cur_var = ops.reshape(ops.summation((x - ex)**2, axes=0) / batch, (1,n))
        if self.training == True:
          run_mean = m * cur_mean + (im) * ops.reshape(self.running_mean, (1,n))
          self.running_mean = ops.reshape(run_mean, (n,)).detach()
          run_var = m * cur_var + (im) * ops.reshape(self.running_var, (1,n))
          self.running_var = ops.reshape(run_var, (n,)).detach()
        
        w = ops.broadcast_to(ops.reshape(self.weight, (1,n)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1,n)), x.shape)
        if self.training == False:
          cur_mean, cur_var = self.running_mean, self.running_var
        ex = ops.broadcast_to(cur_mean, x.shape)
        varx = ops.broadcast_to(cur_var, x.shape)

        # if self.training == False:
        #   res = ((x - ex)/((varx + self.eps)**0.5))
        # else:
        #   res = w * ((x - ex)/((varx + self.eps)**0.5)) + b
        # res = ((x - ex)/((varx + self.eps)**0.5)) * w + b
        res = (w * x)/((varx + self.eps)**0.5) + b - (w*ex)/((varx + self.eps)**0.5)
        return res
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        self.bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(self.weight)
        self.bias = Parameter(self.bias)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, n = x.shape
        ex = ops.broadcast_to(ops.reshape(ops.summation(x, axes=1) / n,(batch,1)), x.shape)
        varx = ops.broadcast_to(ops.reshape(ops.summation((x - ex)**2, axes=1) / n, (batch,1)), x.shape)
        w = ops.broadcast_to(ops.reshape(self.weight, (1,n)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1,n)), x.shape)
        return w * ((x - ex)/((varx + self.eps)**0.5)) + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = np.float32(p)

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == False:
          return x
        x = x / (np.float32(1)-self.p)
        mask = init.randb(x.shape[0], x.shape[1], device=x.device, p=1-self.p)
        return x * mask
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.fn != None:
          return self.fn(x) + x
        return x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(
            in_channels*kernel_size*kernel_size,
            out_channels*kernel_size*kernel_size,
            (kernel_size, kernel_size, in_channels, out_channels),
            device=device,
            dtype=dtype,
            requires_grad=True)
        self.weight = Parameter(self.weight)
        self.bias = init.zeros(out_channels, device=device, dtype=dtype, requires_grad=True)
        if bias:
            a = np.float32(1.0)/(in_channels * kernel_size**2)**0.5
            self.bias = init.rand(*(out_channels,), low=-a, high=a, device=device, dtype=dtype, requires_grad=True)
        self.bias = Parameter(self.bias)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # turn NCHW to NHWC
        x = ops.permute(x, (0,2,3,1))
        # Calculate padding
        pad = (self.kernel_size - 1) // 2
        # pad = (self.kernel_size) // 2
        # Convolve
        x_conv = ops.conv(x, self.weight, stride=self.stride, padding=pad)
        # turn NHWC to NCHW
        x_conv = ops.permute(x_conv, (0,3,1,2))
        # Add bias
        # print("b broadcast shape: ", x_conv.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1,self.out_channels, 1, 1)), x_conv.shape)
        x_conv = x_conv + b
        
        return x_conv
        ### END YOUR SOLUTION

class GraphConv(Module):
    """
    Graph Convolution Layer. 
    """
    def __init__(self, in_channels, out_channels, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(
            in_channels,
            out_channels,
            (in_channels, out_channels),
            device=device,
            dtype=dtype,
            requires_grad=True)
        self.weight = Parameter(self.weight)
        self.bias = init.zeros(out_channels, device=device, dtype=dtype, requires_grad=True)
        if bias:
            a = np.float32(1.0)/(in_channels)**0.5
            self.bias = init.rand(*(out_channels,), low=-a, high=a, device=device, dtype=dtype, requires_grad=True)
        self.bias = Parameter(self.bias)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor, adj) -> Tensor:
        sup = ops.matmul(x, self.weight)
        output = ops.spmm(adj, sup)
        b = ops.broadcast_to(ops.reshape(self.bias, (1, self.out_channels)), output.shape)
        output = output + b
        
        return output
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        a = np.float32(1.0)/hidden_size**0.5
        self.W_ih = init.rand(*(input_size, hidden_size), low=-a, high=a, device=device, dtype=dtype, requires_grad=True)
        self.W_hh = init.rand(*(hidden_size, hidden_size), low=-a, high=a, device=device, dtype=dtype, requires_grad=True)
        self.W_ih = Parameter(self.W_ih)
        self.W_hh = Parameter(self.W_hh)
        self.bias_ih = init.zeros(*(hidden_size,), device=device, dtype=dtype, requires_grad=False)
        self.bias_hh = init.zeros(*(hidden_size,), device=device, dtype=dtype, requires_grad=False)
        if bias:
            self.bias_ih = init.rand(*(hidden_size,), low=-a, high=a, device=device, dtype=dtype, requires_grad=True)
            self.bias_hh = init.rand(*(hidden_size,), low=-a, high=a, device=device, dtype=dtype, requires_grad=True)
        self.bias_ih = Parameter(self.bias_ih)
        self.bias_hh = Parameter(self.bias_hh)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h == None:
            h = init.zeros(*(X.shape[0], self.hidden_size), device=X.device, dtype=X.dtype)
        linear_res = get_linear(X, h, self.W_ih, self.bias_ih, self.W_hh, self.bias_hh, self.hidden_size)
        if self.nonlinearity == 'tanh':
            h_ = ops.tanh(linear_res)
        elif self.nonlinearity == 'relu':
            h_ = ops.relu(linear_res)
        else:
            raise ValueError("Invalid nonlinearity")
        return h_
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.rnn_cells = []
        for i in range(num_layers):
            if i == 0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h0 == None:
            h0 = init.zeros(*(self.num_layers, bs, self.hidden_size), device=X.device, dtype=X.dtype)
        h = ops.split(h0, axis=0)
        X = ops.split(X, axis=0)
        # print("x type split: ", type(X))
        output = []
        for i in range(seq_len):
            # print("sequence ", i)
            h_ = []
            for j in range(self.num_layers):
                # print("layer ", j)
                if j == 0:
                    # print("x cache type: ", type(X[i].cached_data))
                    h_.append(self.rnn_cells[j](X[i].reshape((bs, self.input_size)) , h[j].reshape((bs, self.hidden_size))))
                else:
                    h_.append(self.rnn_cells[j](h_[j-1].reshape((bs, self.hidden_size)), h[j].reshape((bs, self.hidden_size))))
            output.append(h_[-1])
            h = h_
        output = ops.stack(output, axis=0)
        h_n = ops.stack(h, axis=0)
        assert output.shape == (seq_len, bs, self.hidden_size)
        assert h_n.shape == (self.num_layers, bs, self.hidden_size)
        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        k=1/hidden_size**0.5
        self.W_ih = init.rand(*(input_size, 4*hidden_size), low=-k, high=k, device=device, dtype=dtype, requires_grad=True)
        self.W_hh = init.rand(*(hidden_size, 4*hidden_size), low=-k, high=k, device=device, dtype=dtype, requires_grad=True)
        self.bias_ih = init.zeros(*(4*hidden_size,), device=device, dtype=dtype, requires_grad=False)
        self.bias_hh = init.zeros(*(4*hidden_size,), device=device, dtype=dtype, requires_grad=False)
        if bias:
            self.bias_ih = init.rand(*(4*hidden_size,), low=-k, high=k, device=device, dtype=dtype, requires_grad=True)
            self.bias_hh = init.rand(*(4*hidden_size,), low=-k, high=k, device=device, dtype=dtype, requires_grad=True)
        self.W_ih = Parameter(self.W_ih)
        self.W_hh = Parameter(self.W_hh)
        self.bias_ih = Parameter(self.bias_ih)
        self.bias_hh = Parameter(self.bias_hh)

        ### END YOUR SOLUTION



    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h == None:
            h = (init.zeros(*(X.shape[0], self.hidden_size), device=X.device, dtype=X.dtype),
                 init.zeros(*(X.shape[0], self.hidden_size), device=X.device, dtype=X.dtype))
        h0, c0 = h

        Wii, Wif, Wig, Wio = split_section(self.W_ih, 1, self.hidden_size)
        Whi, Whf, Whg, Who = split_section(self.W_hh, 1, self.hidden_size)
        bii, bif, big, bio = split_section(self.bias_ih, 0, self.hidden_size)
        bhi, bhf, bhg, bho = split_section(self.bias_hh, 0, self.hidden_size)

        linear_res_i = get_linear(X, h0, Wii, bii, Whi, bhi, self.hidden_size)
        linear_res_f = get_linear(X, h0, Wif, bif, Whf, bhf, self.hidden_size)
        linear_res_g = get_linear(X, h0, Wig, big, Whg, bhg, self.hidden_size)
        linear_res_o = get_linear(X, h0, Wio, bio, Who, bho, self.hidden_size)

        i = ops.sigmoid(linear_res_i)
        f = ops.sigmoid(linear_res_f)
        g = ops.tanh(linear_res_g)
        o = ops.sigmoid(linear_res_o)
        c_ = f * c0 + i * g
        h_ = o * ops.tanh(c_)
        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.lstm_cells = []
        for i in range(num_layers):
            if i == 0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h == None:
            h = (init.zeros(*(self.num_layers, bs, self.hidden_size), device=X.device, dtype=X.dtype),
                 init.zeros(*(self.num_layers, bs, self.hidden_size), device=X.device, dtype=X.dtype))
        h0, c0 = h
        h0 = ops.split(h0, axis=0)
        c0 = ops.split(c0, axis=0)
        X = ops.split(X, axis=0)
        output = []
        for i in range(seq_len):
            h_ = []
            c_ = []
            for j in range(self.num_layers):
                if j == 0:
                    ht, ct = self.lstm_cells[j](X[i].reshape((bs, self.input_size)), (h0[j].reshape((bs, self.hidden_size)), c0[j].reshape((bs, self.hidden_size))))
                    
                else:
                    ht, ct = self.lstm_cells[j](h_[j-1].reshape((bs, self.hidden_size)), (h0[j].reshape((bs, self.hidden_size)), c0[j].reshape((bs, self.hidden_size))))
                h_.append(ht)
                c_.append(ct)
            output.append(h_[-1])
            h0 = h_
            c0 = c_
        output = ops.stack(output, axis=0)
        h_n = ops.stack(h0, axis=0)
        c_n = ops.stack(c0, axis=0)
        assert output.shape == (seq_len, bs, self.hidden_size)
        assert h_n.shape == (self.num_layers, bs, self.hidden_size)
        assert c_n.shape == (self.num_layers, bs, self.hidden_size)
        return output, (h_n, c_n)

        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype=dtype
        self.weight = init.randn(*(num_embeddings, embedding_dim), mean=0, std=1, device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(self.weight)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        input = x.numpy()
        output = np.zeros((seq_len, bs, self.num_embeddings), dtype=self.dtype)
        for i in range(seq_len):
            for j in range(bs):
                output[i][j][int(input[i][j])] = 1
        output = output.reshape((seq_len * bs, self.num_embeddings))
        output = Tensor(NDArray(output, device=x.device), device=x.device, dtype=x.dtype)
        output = ops.matmul(output, self.weight)
        output = output.reshape((seq_len, bs, self.embedding_dim))
        assert output.shape == (seq_len, bs, self.embedding_dim)
        return output
        ### END YOUR SOLUTION
