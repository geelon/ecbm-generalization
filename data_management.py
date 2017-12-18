import numpy as np


class BatchGenerator:
    def __init__(self, X, Y, num_classes, batch_size, repeat=False, label_scheme='original', data_scheme='image'):
        self.X = X
        self.Y = Y
        self.num_classes  = num_classes
        self.batch_size   = batch_size
        self.repeat       = repeat
        self.max_index    = X.shape[0]
        self.label_scheme = label_scheme  # 'original', 'permute', or 'random'
        self.data_scheme  = data_scheme   # 'image', or 'random'
        self.index        = 0             # current index


        if label_scheme == 'permute':
            self.Y = np.random.shuffle(Y)
        if label_scheme == 'random':
            self.Y = np.random.randint(10, size=Y.shape[0])


        # Encode Y's as one-hot vectors
        Y_onehot = np.zeros([self.max_index,num_classes])
        for elem in range(self.max_index):
            Y_onehot[elem,Y[elem]] = 1
        self.Y = Y_onehot


        if data_scheme == 'random':
            self.X = np.random.randint(255,size=X.shape)

    def get_first_size(self,size):
        """
        Returns the first size number as a batch.
        """
        terminal = min(self.max_index, size)
        X_batch = self.X[:terminal]
        Y_batch = self.Y[:terminal]

        return X_batch, Y_batch

        
    def get_next_size(self, size):
        """
        Generates the next batch, with size number of elements.
        If repeat is False, throws exception if not enough left.
        Returns X_batch, Y_batch
        """
        num_classes = self.num_classes

        # compute index of head and tail of batch to return
        initial = self.index
        terminal = initial + size
        self.index = terminal

        # behavior for cycling through data
        if terminal >= self.max_index:
            if self.repeat:
                initial    = 0
                terminal   = size
                self.index = size
            else:
                raise NameError('Not enough training data.')

        # obtain batch
        X_batch = self.X[initial:terminal]
        Y_batch = self.Y[initial:terminal]

        return X_batch, Y_batch

    def get_next(self):
        return self.get_next_size(self.batch_size)


    def generate_random_size(self,size):
        X = np.random.randint(255, size=(size,3,32,32))
        Y = np.random.randint(self.num_classes, size=size)
        return X, Y

    def generate_random(self):
        return generate_random_size(self.batch_size)
