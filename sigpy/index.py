import numpy as np

            
class ShuffledIndex(object):
    """Produces indices that are shuffled between 0 and n-1.

    Args:
        n (int): Upper bound on indices.
    
    """

    def __init__(self, *args):
        self.stream = np.arange(*args)
        self.next_stream = np.arange(*args)
        
        np.random.shuffle(self.stream)
        self.idx = -1

    def current(self):
        if self.idx == -1:
            raise TypeError("ShuffledIndex.next() has not been called yet.")
        
        return self.stream[self.idx]
    
    def next(self):
        self.idx += 1
        if self.idx == len(self.stream):
            np.random.shuffle(self.stream)
            self.idx = 0

        return self.current()


class PingPongIndex(object):
    """Produces indices that goes back and forth between 0 and n-1.

    For example, if n = 3, it produces {0, 1, 2, 2, 1, 0}.

    Args:
        n (int): Upper bound on indices.
    
    """

    def __init__(self, n):
        self.n = n
        self.direction = 1
        self.idx = -1

    def current(self):
        if self.idx == -1:
            raise TypeError("PingPongIndex.next() has not been called yet.")
        
        return self.idx
    
    def next(self):
        self.idx += self.direction
        if self.idx == -1:
            self.idx = 0
            self.direction = 1
        elif self.idx == self.n:
            self.idx = self.n - 1
            self.direction = -1

        return self.current()
