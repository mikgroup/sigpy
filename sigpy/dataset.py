import glob
import numpy as np
import pickle
from sigpy import util


class NpyFiles(object):

    def __init__(self, filepaths):
        self.filepaths = [str(f) for f in filepaths]
        self.ndim = None
        for f in self.filepaths:
            arr = np.load(f, mmap_mode='r')
            if self.ndim:
                if self.ndim != arr.ndim + 1:
                    raise ValueError('Datasets must have the same number of dimensions.')

                if self.dtype != arr.dtype:
                    raise ValueError('Datasets must have the same dtype.')
                
                self.shape = tuple([len(self.filepaths)] +
                                   [max(s1, s2) for s1, s2 in zip(self.shape[1:], arr.shape)])
            else:
                self.shape = (len(self.filepaths), ) + arr.shape
                self.ndim = arr.ndim + 1
                self.dtype = arr.dtype

    def __len__(self):
        return len(self.filepaths)

    def _get_dataset(self, i):
        return util.resize(np.load(self.filepaths[i]), self.shape[1:])

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_dataset(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self.filepaths))
            return np.stack(self._get_dataset(i) for i in range(start, stop, step))
        
        elif isinstance(index, tuple) or isinstance(index, list):
            if isinstance(index[0], int):
                return self._get_dataset(index[0])[index[1:]]
            elif isinstance(index[0], slice):
                start, stop, step = index[0].indices(len(self.filepaths))
                return np.stack(self._get_dataset(i)[index[1:]] for i in range(start, stop, step))
            
