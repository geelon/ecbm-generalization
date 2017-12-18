import math
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from data_management import BatchGenerator
from preprocess import Preprocessor
from default import learning_kwargs, conv_kwargs


class LearningProblem:
    def __init__(self,data,train_size,num_classes,batch_per_epoch,batch_size,accuracy_size,validation_size,centered=False,
                 rescaled=True,grayscale=True, shaped=True, repeat=False, label_scheme='original', data_scheme='image'):
        """
        data = [X_train, Y_train, X_test, Y_test]
        
        get_next_train() will return next batch of training data
        get_next_test() will return next batch of test data
        """
        X_train, Y_train, X_test, Y_test = data
        X_train = X_train[:train_size]
        Y_train = Y_train[:train_size]
        self.train_batch     = BatchGenerator(X_train,Y_train,num_classes,batch_size,repeat,label_scheme)
        self.test_batch      = BatchGenerator(X_test,Y_test,num_classes,validation_size,repeat)
        self.batch_per_epoch = batch_per_epoch  # number of batches per epoch
        self.batch_size      = batch_size       # size of batch for training
        self.accuracy_size   = accuracy_size    # size of batch to measure training accuracy
        self.validation_size = validation_size  # size of batch to measure validation accuracy
        self.centered        = centered
        self.rescaled        = rescaled
        self.grayscale       = grayscale
        self.shaped          = shaped
        


    def get_next(self,set):
        """
        Get next preprocessed batch
        """
        raw_data     = tf.placeholder(tf.float32, shape=[None, 3,32,32])
        preprocessor = Preprocessor(raw_data,centered=self.centered,rescaled=self.rescaled,
                                    grayscale=self.grayscale,shaped=self.shaped)
        
        if set == "train":
            batch = self.train_batch
            size = self.batch_size
            raw_x, y_batch = batch.get_next()
        elif set == "train_acc":
            batch = self.train_batch
            size = self.accuracy_size
            raw_x, y_batch = batch.get_first_size(self.accuracy_size)
        elif set == "test_acc":
            batch = self.test_batch
            size = self.validation_size
            raw_x, y_batch = batch.get_first_size(self.validation_size)

        with tf.Session() as sess:
            raw_x = raw_x.reshape(-1,3,32,32)
            x_batch = sess.run(preprocessor.apply(raw_x,size),
                               feed_dict={raw_data : raw_x})
        return x_batch, y_batch

    def get_next_train(self):
        return self.get_next("train")

    def get_train_acc(self):
        return self.get_next("train_acc")
    
    def get_test_acc(self):
        return self.get_next("test_acc")

    def get_acc(self):
        train_x, train_y = self.get_next("train_acc")
        test_x, test_y   = self.get_next("test_acc")
        return train_x, train_y, test_x, test_y


        
        

def get_params(name, shape):
    """
    Returns variable with specified name and shape.
    """
    return tf.get_variable(name, shape, 
                           initializer=tf.truncated_normal_initializer(),
                           dtype=tf.float32)



class BasicConv:
    """
    This model is:
    xs >> conv >> pool >> fc >> dropout >> fc
    """
    def __init__(self, learning_rate=1e-3, x_shape=[32,32], kernel_size=5, channels=3,
                 pooling=2, filters=32, fc_feat=1024, keep_prob=0.9, num_classes=10):
        self.learning_rate = learning_rate
        self.x_shape = x_shape
        self.kernel_size = kernel_size
        self.channels = channels
        self.pooling = pooling
        self.filters = filters
        self.fc_feat = fc_feat
        self.keep_prob = keep_prob

        fc1_in = int(x_shape[0] * x_shape[1] * filters // (pooling * pooling))
        self.fc1_in = fc1_in

        self.variable_names = ['conv_w', 'conv_b'
                               ,'fc1_w', 'fc1_b'
                               ,'fc2_w', 'fc2_b']
        self.variable_shape = [[kernel_size,kernel_size,channels,filters],[filters]
                               ,[fc1_in, fc_feat], [fc_feat]
                               ,[fc_feat,num_classes],[num_classes]]
        self.variables = {}

        for i in range(6):
            self.variables[self.variable_names[i]] = get_params(self.variable_names[i], 
                                                                self.variable_shape[i])

        self.train_x = tf.placeholder(tf.float32, shape=[None,x_shape[0],x_shape[1], channels])
        self.train_y = tf.placeholder(tf.float32, shape=[None,num_classes])
        
        self.test_x = tf.placeholder(tf.float32, shape=[None,x_shape[0],x_shape[1], channels])
        self.test_y = tf.placeholder(tf.float32, shape=[None,num_classes])

    def get_placeholders(self):
        return self.train_x, self.train_y, self.test_x, self.test_y

    def evaluate(self,xs):
        """
        Evaluating the CNN, returns output of CNN along with predictions.
        """
        variables = self.variables
        pooling = self.pooling
        # conv
        conv = tf.nn.conv2d(xs, filter=variables['conv_w'], strides=[1,1,1,1], padding='SAME')
        conv = tf.nn.relu(conv + variables['conv_b'], name='conv')
        # pool
        pool = tf.nn.max_pool(conv, ksize=[1,pooling,pooling,1],
                              strides=[1,pooling,pooling,1], padding='SAME')
        # fc1
        flatten = tf.reshape(pool, [-1,self.fc1_in])
        fc1 = tf.nn.relu(tf.matmul(flatten,variables['fc1_w']) + variables['fc1_b'])
        # dropout
        dropout = tf.nn.dropout(fc1, self.keep_prob)
        # fc2
        y_conv = tf.matmul(dropout, variables['fc2_w']) + variables['fc2_b']
    
        return y_conv

    def compute_accuracy(self,y_conv,labels):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels,1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction)

    def train(self,xs,ys):
        """
        Returns accuracy and cross_entropy loss
        """
        y_conv = self.evaluate(xs)
        accuracy = self.compute_accuracy(y_conv, ys)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=y_conv))
        trained = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(ce)
        return accuracy, ce, trained

    def training_pass(self,train_x, train_y, test_x, test_y):
        """
        Runs model on both training and testing data to compare generalization
        """
        test_pred = self.evaluate(test_x)
        test_accuracy = self.compute_accuracy(test_pred,test_y)
        train_accuracy, ce, trained = self.train(train_x, train_y)
        return train_accuracy, test_accuracy, ce, trained

    def accuracy(self, train_x, train_y, test_x, test_y):
        train_pred = self.evaluate(train_x)
        train_acc  = self.compute_accuracy(train_pred, train_y)
        test_pred  = self.evaluate(test_x)
        test_acc   = self.compute_accuracy(test_pred, test_y)
        return train_acc, test_acc


def compare_accuracy(data, num_epochs, learning_kwargs, conv_kwargs):
    # setup
    learning_kwargs['data'] = data
    batch_per_epoch = learning_kwargs['batch_per_epoch']
    batch_size = learning_kwargs['batch_size']
    problem = LearningProblem(**learning_kwargs)
    cnn = BasicConv(**conv_kwargs)
    train_x, train_y, test_x, test_y = cnn.get_placeholders()
    batch_accuracy = np.zeros((num_epochs,batch_per_epoch))
    train_accuracy = np.zeros(num_epochs)
    test_accuracy  = np.zeros(num_epochs)

    # description
    print("Training will consist of {} epochs".format(num_epochs))
    print("- there will be {} batches per epoch".format(batch_per_epoch))
    print("- each batch has {} samples".format(batch_size))
    print("- training accuracy requires {} samples".format(learning_kwargs['accuracy_size']))
    print("- testing accuracy requires {} samples".format(learning_kwargs['validation_size']))
    
    # run model
    print("Initializing model.")
    start = timer()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            for batch in range(batch_per_epoch):
                print("Batch {}".format(batch))
                feed_dict = dict(zip([train_x, train_y],problem.get_next_train()))
                batch_accuracy[epoch,batch], _, _ = sess.run(cnn.train(train_x, train_y), feed_dict=feed_dict)
            feed_dict = dict(zip(cnn.get_placeholders(), problem.get_acc()))
            train_accuracy[epoch], test_accuracy[epoch] = sess.run(cnn.accuracy(train_x, train_y, test_x, test_y), 
                                                                   feed_dict=feed_dict)
            print("Epoch {}, train acc: {}, valid acc: {}".format(epoch, train_accuracy[epoch], test_accuracy[epoch]))
    end = timer()
    print("Seconds elapsed: {}".format(end-start))
    return batch_accuracy, train_accuracy, test_accuracy

