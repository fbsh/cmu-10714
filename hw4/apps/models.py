import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def convbn(a, b, k, s, device):
    return nn.Sequential(
        nn.Conv(a, b, k, stride=s, device=device),
        nn.BatchNorm2d(b, device=device),
        nn.ReLU()
    )


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device = device
        self.dtype = dtype

        fx1 = nn.Sequential(
            convbn(3, 16, 7, 4, device),
            convbn(16, 32, 3, 2, device)
        )
        fn1 = nn.Sequential(
            convbn(32, 32, 3, 1, device),
            convbn(32, 32, 3, 1, device)
        )
        res1 = nn.Residual(fn1)

        fx2 = nn.Sequential(
            convbn(32, 64, 3, 2, device),
            convbn(64, 128, 3, 2, device)
        )
        fn2 = nn.Sequential(
            convbn(128, 128, 3, 1, device),
            convbn(128, 128, 3, 1, device)
        )
        res2 = nn.Residual(fn2)
        self.model = nn.Sequential(
            fx1,
            res1,
            fx2,
            res2,
            nn.Flatten(),
            nn.Linear(128, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 10, device=device)
        )

        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.seq_model = seq_model
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            raise ValueError('seq_model must be rnn or lstm')
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x = self.embedding(x)
        x, h = self.model(x, h)
        x = x.reshape((seq_len*bs, self.hidden_size))
        x = self.linear(x)
        # sum = ndl.ops.summation(x, axes=(1,))
        # sum = sum.reshape((seq_len*bs, 1))
        # sum = sum.broadcast_to((seq_len*bs, self.output_size))
        # x = x / sum
        return x, h
        ### END YOUR SOLUTION

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out = 0.1, device=None, dtype="float32"):
        super(GCN, self).__init__()
        self.graph_conv1 = nn.GraphConv(input_size, hidden_size, device=device)
        self.relu = nn.ReLU()
        self.graph_conv2 = nn.GraphConv(hidden_size, output_size, device=device)
        self.dropout = nn.Dropout(drop_out)
        

    def forward(self, x, adj):
        x = self.relu(self.graph_conv1(x, adj))
        x = self.dropout(x)
        x = self.graph_conv2(x, adj)
        return x

if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)