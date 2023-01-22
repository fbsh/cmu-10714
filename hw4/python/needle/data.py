import numpy as np
from .autograd import Tensor
import os
import pickle
import struct
import gzip
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
          return np.fliplr(img)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        x,y = img.shape[:2]
        shift_x, shift_y = -shift_x, -shift_y
        res = np.zeros(img.shape, dtype=img.dtype)
        if abs(shift_x) >= x or abs(shift_y) >= y:
          return res
        x_tl = max(0, 0 + shift_x)
        y_tl = max(0, 0 + shift_y)
        x_br = min(x, x + shift_x)
        y_br = min(y, y + shift_y)

        x_s = max(0, 0 - shift_x)
        y_s = max(0, 0 - shift_y)
        x_e = min(x, x - shift_x)
        y_e = min(y, y - shift_y)

        res[x_tl:x_br, y_tl:y_br] = img[x_s:x_e, y_s:y_e]
        return res

        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.iteridx = 0
        if self.shuffle:
          idx = np.arange(len(self.dataset))
          np.random.shuffle(idx)
          self.ordering = np.array_split(idx, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self) -> List[Tensor]:
        ### BEGIN YOUR SOLUTION
        if self.iteridx >= len(self.ordering):
          raise StopIteration
        sample = [self.dataset[int(i)] for i in self.ordering[self.iteridx]]
        sample = zip(*sample)
        # img = Tensor.make_const(np.array([s[0] for s in sample],dtype=np.float32))
        # label = Tensor.make_const(np.array([s[1] for s in sample],dtype=np.float32))
        self.iteridx = self.iteridx + 1
        return tuple(Tensor.make_const(nd.NDArray(s)) for s in sample)
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
       ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, 'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            nrows, ncols = struct.unpack('>II', f.read(8))
            buf = f.read(size * nrows * ncols)
            X = np.frombuffer(buf, dtype=np.dtype(np.uint8)).newbyteorder(">")
            X = X.reshape((size,nrows, ncols, 1)).astype('float32')
            X = X / np.float32(255.0)

        with gzip.open(label_filename, 'rb') as i:
            magic, size = struct.unpack('>II', i.read(8))
            buf = i.read(size)
            y = np.frombuffer(buf, dtype=np.dtype(np.uint8)).newbyteorder(">")
        self.image = X
        self.label = y
        
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if isinstance(index, int):
          img = self.image[index]
          img = self.apply_transforms(img)
        elif isinstance(index, slice):
          img = [self.apply_transforms(self.image[ii]) for ii in range(*index.indices(len(self)))]
          img = np.array(img, dtype=np.float32)
        else:
          raise TypeError
        label = self.label[index]
        return (img, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.image.shape[0]
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        X = []
        y = []
        if train:
            filename = os.path.join(base_folder, 'data_batch_')
            for i in range(1,6):
                with open(filename + str(i), 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    X.append(dict[b'data'])
                    y.append(dict[b'labels'])
        else:
            filename = os.path.join(base_folder, 'test_batch')
            with open(filename, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                X.append(dict[b'data'])
                y.append(dict[b'labels'])
        X = np.concatenate(X)
        y = np.concatenate(y)
        X = X.reshape((X.shape[0], 3, 32, 32)).astype("float32")
        X = X / np.float32(255.0)
        self.image = X
        self.label = y

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(index, int):
          img = self.image[index]
          img = self.apply_transforms(img)
        elif isinstance(index, slice):
          img = [self.apply_transforms(self.image[ii]) for ii in range(*index.indices(len(self)))]
          img = np.array(img, dtype=np.float32)
        else:
          raise TypeError
        label = self.label[index]
        return (img, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.image.shape[0]
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
          self.word2idx[word] = len(self.idx2word)
          self.idx2word.append(word)
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        # Add words to the dictionary
        max_lines_ = max_lines
        with open(path, 'r') as f:
            for line in f:
                for word in line.split() + ['<eos>']:
                    self.dictionary.add_word(word)
                if max_lines is not None:
                    max_lines -= 1
                    if max_lines == 0:
                        break
        # Tokenize file content
        max_lines = max_lines_
        ids = []
        with open(path, 'r') as f:
            for line in f:
                for word in line.split() + ['<eos>']:
                    ids.append(self.dictionary.word2idx[word])
                if max_lines is not None:
                    max_lines -= 1
                    if max_lines == 0:
                        break
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    nbatch = len(data) // batch_size
    data = data[:nbatch * batch_size]
    data = np.array(data, dtype=dtype)
    data = data.reshape((batch_size, -1)).T
    assert data.shape == (nbatch, batch_size)
    return data
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    seq_len = min(bptt, len(batches) - 1 - i)
    data = batches[i:i+seq_len]
    target = batches[i+1:i+1+seq_len].reshape(-1)
    data = Tensor(nd.NDArray(data, device=device), device=device, dtype=dtype)
    target = Tensor(nd.NDArray(target, device=device), device=device, dtype=dtype)
    return data, target
    ### END YOUR SOLUTION


def normalize(A):
    """
    Row-normalize matrix A.
    First, assume the symmetric adjacency matrix is A, the input matrix is A + I.
    Then, we calculate degree matrix of input, noted D.
    Since A + I is symmetric, we normalize the input as D^(-1) * (A + I).
    """
    degree = A.sum(axis=1)
    degree_inv = np.power(degree, -1).flatten()
    degree_inv[np.isinf(degree_inv)] = 0.
    D = np.diag(degree_inv)
    out = D.dot(A)
    return out

class CoraDataset(object):
    @staticmethod
    def load_data(path="data/cora/", dataset="cora", device=None):
        """Load Cora dataset
        Output:
        features - ndl.Tensor of shape (2708, 1433)
        labels - ndl.Tensor of shape (2708,)
        adj - ndl.SparseMatrix of shape (2708, 2708)
        """

        idx_features_labels = np.genfromtxt(f"{path}cora.content",
                                            dtype=np.dtype(str))
        features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
        classes = list(set(idx_features_labels[:, -1]))
        labels = idx_features_labels[:, -1].reshape((2708,)).tolist()
        labels = np.array([classes.index(i) for i in labels], dtype=np.int32).reshape((2708,))

        # adjacency matrix
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_idx = list(idx.reshape((idx.shape[0],)))
        edges_raw = np.genfromtxt("{}cora.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array([idx_idx.index(i) for i in edges_raw.reshape((edges_raw.shape[0] * 2,))],
                        dtype=np.int32).reshape(edges_raw.shape)
        adj = np.zeros((2708, 2708))
        adj[edges[:, 0], edges[:, 1]] = 1

        # symmetrify adjacency matrix
        adj = adj + np.multiply(adj.T, (adj.T > adj)) - np.multiply(adj, (adj.T > adj))

        features = normalize(features)
        adj = normalize(adj + np.eye(adj.shape[0]))

        #features = Tensor(nd.NDArray(features, device=device), device=device)
        #labels = Tensor(nd.NDArray(labels, device=device), device=device)
        #adj = nd.NDArray(adj, device=device)
        #adj = nd.to_sparse(adj)
        return adj, features, labels
