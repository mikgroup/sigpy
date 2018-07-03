'''Index stream.
'''
import numpy as np


class ShuffleIndexStream(object):

    def __init__(self, n, batch_size=1):
        self.stream = np.arange(n)
        self.next_stream = np.arange(n)
        self.batch_size = batch_size

        np.random.shuffle(self.stream)
        np.random.shuffle(self.next_stream)
        self.count = -1

    def current(self):
        assert self.count >= 0
        return self.stream[self.count]

    def next(self):
        self.count += 1
        if self.count == len(self.stream):
            np.random.shuffle(self.stream)
            self.count = 0

        return self.current()

    def next_batch(self):
        ret = []

        for i in range(self.batch_size):

            ret.append(self.stream[self.count])

            self.count += 1
            if self.count == len(self.stream):
                self.stream[:] = self.next_stream
                np.random.shuffle(self.next_stream)
                self.count = 0

        return ret


class ZigzagIndexStream(object):

    def __init__(self, n, batch_size=1):
        self.stream = np.arange(n)
        self.batch_size = batch_size
        self.direction = 1
        self.count = -1

    def current(self):
        assert self.count >= 0
        return self.stream[self.count]

    def next(self):
        self.count += self.direction
        if self.count == -1:
            self.count = 0
            self.direction = 1
        elif self.count == len(self.stream):
            self.count = len(self.stream) - 1
            self.direction = -1

        return self.current()

    def next_batch(self):
        ret = []

        for i in range(self.batch_size):

            ret.append(self.stream[self.count])

            self.count += self.direction
            if self.count == -1:
                self.count = 0
                self.direction = 1
            elif self.count == len(self.stream):
                self.count = len(self.stream) - 1
                self.direction = -1

        return ret
