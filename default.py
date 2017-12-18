# Data properties
X_SHAPE     = [32,32] # image is 32 x 32 pixels
TRAIN_SIZE  = 50000   # number of training points to use, no more than 50000
NUM_CLASSES = 10      # number of categories
TEST_SIZE   = 10000   # at most 10000 (size of test set)


# Data management options:
REPEAT = True # cycle through data?


# Preprocessing options:
CENTERED  = False # zero-centers data
RESCALED  = True  # normalized to 1
GRAYSCALE = False # add grayscale channel
SHAPED    = True  # reshape to [32,32,3]


# Model options:
# conv >> pool >> fc1 >> dropout >> fc2
KERNEL_SIZE = 5    # convolution kernel has dimension KERNEL_SIZE x KERNEL_SIZE
NUM_FILTERS = 32   # number of convolution filters
POOLING     = 2    # reduction factor per dimension
FC_FEATURES = 1024 # number of features produced by fc1
KEEP_PROB   = 0.9  # 1 - dropout probatility


# Training options:
LEARNING_RATE   = 1e-3       # SGD learning rate
NUM_EPOCHS      = 50         # number of epochs
BATCH_SIZE      = 100        # at most TRAIN_SIZE
LABEL_SCHEME    = 'original' # 'original', 'permute', or 'random'
ACCURACY_SIZE   = 30         # number of points for training accuracy
VALIDATION_SIZE = 30         # number of points for testing accuracy at most TEST_SIZE


# Derived Statistics
# Number of Channels:
if GRAYSCALE:
    CHANNELS=4
else:
    CHANNELS=3
    
# learning problem
learning_kwargs = {
    'train_size'      : TRAIN_SIZE,
    'num_classes'     : NUM_CLASSES,
    'batch_size'      : BATCH_SIZE,
    'accuracy_size'   : ACCURACY_SIZE,
    'validation_size' : VALIDATION_SIZE,
    'centered'        : CENTERED,
    'rescaled'        : RESCALED,
    'grayscale'       : GRAYSCALE,
    'shaped'          : SHAPED,
    'repeat'          : REPEAT,
    'label_scheme'    : LABEL_SCHEME
}

# cnn
conv_kwargs = {
    'learning_rate' : LEARNING_RATE,
    'x_shape'       : X_SHAPE,
    'kernel_size'   : KERNEL_SIZE,
    'channels'      : CHANNELS,
    'pooling'       : POOLING,
    'filters'       : NUM_FILTERS,
    'fc_feat'       : FC_FEATURES,
    'keep_prob'     : KEEP_PROB,
    'num_classes'   : NUM_CLASSES
}
