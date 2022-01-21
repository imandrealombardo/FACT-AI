# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import numpy as np

class DataSet(object):

    def __init__(self, x, labels):

        if len(x.shape) > 2:
            x = np.reshape(x, [x.shape[0], -1])

        assert(x.shape[0] == labels.shape[0])

        x = x.astype(np.float32)

        self._x = x
        self._x_batch = np.copy(x)
        self._labels = labels
        self._labels_batch = np.copy(labels)
        self._num_examples = x.shape[0]
        self._index_in_epoch = 0

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def reset_batch(self):
        self._index_in_epoch = 0        
        self._x_batch = np.copy(self._x)
        self._labels_batch = np.copy(self._labels)

    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        # ====== This is not used. idk if it can be useful in the future ======
        # if self._index_in_epoch > self._num_examples:
            # # Shuffle the data
            # perm = np.arange(self._num_examples)
            # np.random.shuffle(perm)
            # self._x_batch = self._x_batch[perm, :]
            # self._labels_batch = self._labels_batch[perm]

            # # Start next epoch
            # start = 0
            # self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._x_batch[start:end], self._labels_batch[start:end]