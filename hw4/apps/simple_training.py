import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
      y_hat = np.argmax(y_hat, axis=1)
    cmp = y_hat == y.astype('int')
    return np.float32(np.sum(cmp))

def batch_loss(logits: nn.Tensor, y: nn.Tensor):
    r, n = logits.shape
    zy = ndl.ops.summation(logits * nn.init.one_hot(n, y, device=logits.device),axes=1)
    res = ndl.ops.summation(nn.ops.logsumexp(logits, (1,)) - zy)
    return res


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt == None:
        model.eval()
    else:
        model.train()
    nbatch, batch_size = data.shape
    avg_loss = np.float32(0)
    avg_acc = np.float32(0)
    sum_samples = np.float32(0)
    # for i in range(nbatch - seq_len):
    for i in range(0, nbatch - 1, seq_len):
        batch_x, batch_y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        sum_samples += batch_size * batch_x.shape[0]
        if opt == None:
            out, _ = model(batch_x)
            loss = loss_fn(out, batch_y)
        else:
            opt.reset_grad()
            out, _ = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            if getattr(opt, 'clip_grad_norm', None) is not None:
                if clip is not None:
                    opt.clip_grad_norm(clip)
                else:
                    opt.clip_grad_norm()
            opt.step()
        avg_loss += batch_loss(out, batch_y)
        avg_acc += accuracy(out.numpy(), batch_y.numpy())
    return avg_acc / np.float32(sum_samples), avg_loss.numpy() / np.float32(sum_samples)   
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=opt, clip=clip, device=device, dtype=dtype)
        # print("loss: ", avg_loss, "acc: ", avg_acc)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=None, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def epoch_cora(data, model, slices, loss_fn=nn.SoftmaxLoss(), opt=None, clip=None):
    def tensor_getitem(t, s):
        t = ndl.ops.split(t, axis=0)
        t = t.tuple()[s]
        t = ndl.ops.stack(t, axis=0)
        return t

    adj, features, labels = data
    nsamples = features.shape[0]

    if opt is not None:
        model.train()
        opt.reset_grad()
    else:
        model.eval()

    output = model(features, adj)
    #import pdb; pdb.set_trace()
    #loss = loss_fn(tensor_getitem(output, slices), tensor_getitem(labels, slices))
    loss = loss_fn(output, labels)

    if opt is not None:
        loss.backward()
        opt.step()
    
    return accuracy(output.numpy(), labels.numpy()) / nsamples, loss.detach().numpy()


def train_cora(data, model, slices, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.01, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: GCN instance
        data: a tuple of Tensors (adj, features, labels)
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    avg_loss = []
    avg_acc = []

    for epoch in range(n_epochs):
        #import pdb; pdb.set_trace()
        acc, loss = epoch_cora(data, model, slices, loss_fn=loss_fn(), opt=opt, clip=clip)
        avg_acc.append(acc)
        avg_loss.append(loss.item())
        #print(f"{epoch}: {avg_acc}, {avg_loss.item()}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_cora(data, model, slices, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: GCN instance
        data: a tuple of Tensors (adj, features, labels)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of testing
        avg_loss: average loss over dataset from last epoch of testing
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_cora(data, model, slices, loss_fn=loss_fn())
    return avg_acc, avg_loss.item()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
