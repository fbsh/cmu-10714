
import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    return nn.Sequential(nn.Residual(fn), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for i in range(num_blocks)], 
        nn.Linear(hidden_dim, num_classes))
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train() if opt else model.eval()
    loss_func = nn.SoftmaxLoss()
    losses = 0
    errors = 0
    ns = 0
    for i, batch in enumerate(dataloader):
        X, y = batch[0], batch[1]
        p = model(X)
        loss = loss_func(p, y)
        err = p.numpy().argmax(axis=1) - y.numpy()
        ns += len(err)
        errors += np.sum(err != 0)
        losses += loss.numpy() * len(err)
        if opt:
            loss.backward()
            opt.step()

    return errors/ns, losses / ns
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(\
            "./data/train-images-idx3-ubyte.gz",
            "./data/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True)
    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=False)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        tr_err, tr_loss = epoch(train_dataloader, model, opt)
        print(i, tr_err, tr_loss)
    te_err, te_loss = epoch(test_dataloader, model)
    print(te_err, te_loss)
    return (tr_err, tr_loss, te_err, te_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
