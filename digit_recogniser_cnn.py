#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 20:51:46 2017

@author: nam
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Getting inputs, and outputs(in one-hot vector form) 
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Setting up hyperparameters
learning_rate = 0.0001
epochs = 10
batch_size = 50

# Setting up input and output vectors
x = tf.placeholder(tf.float32, [None, 784])

# Dynamically reshaping x into 2D [28x28 pixel image with one color channel(grayscale)]
x_shaped = tf.reshape(x, [-1, 28, 28, 1])

y = tf.placeholder(tf.float32, [None, 10])

# Handy function to create a new convolution layer

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, 
                          pool_shape, name):
    # Definining filter shape as a 4D vector for conv2d function
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                         num_filters]
    
    #Setting up weights and biases
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.3),
                          name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b') # number of filters essentially is the number of output channels
    
    # convolving
    out_layer = tf.nn.conv2d(input_data, weights, [1,1,1,1], padding='SAME')
    
    # Adding the bias
    out_layer += bias
    
    # Passing it through a non-linearity
    out_layer = tf.nn.relu(out_layer)
    
    # Max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1] # Max pooling filter size(ignoring first and last value as they are always set to 1)
    strides = [1, 2, 2, 1]
    out = tf.nn.max_pool(out_layer,ksize, strides, padding='SAME')
    
    return out
    
    
# Creating 2 convolution layers
conv_layer_1 = create_new_conv_layer(x_shaped, 1, 32, [5,5], [2,2], 'layer1')
conv_layer_2 = create_new_conv_layer(conv_layer_1, 32, 64, [5,5], [2,2], 'layer2')

# Flattening out the output of the final convolution layer
flattened = tf.reshape(conv_layer_2, [-1, 7*7*64])

# Intitalising the weight, and bias for fully connected layer
w1 = tf.Variable(tf.truncated_normal([7*7*64, 1000], stddev=0.1), name='full_layer1_W')
b1 = tf.Variable(tf.truncated_normal([1000], stddev=0.1), name='full_layer1_b')

# Calculating hypothesis
dense_layer_1 = tf.matmul(flattened, w1) + b1
dense_layer_1 = tf.nn.relu(dense_layer_1)

# Repeating for second output layer but using softmax as it connects to the output 
w2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1), name='full_layer2_W')
b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1), name='full_layer2_b')

# Calculating hypothesis
dense_layer_2 = tf.matmul(dense_layer_1, w2) + b2

hypothesis = tf.nn.softmax(dense_layer_2)

# Calculating the loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer_2, labels=y))

# Adding an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Defining an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(hypothesis, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Setting up the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialising the variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    #print total_batch
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cost], 
                            feed_dict={x: batch_x, y: batch_y})
#            if c<1.5:
#                avg_cost += c / total_batch
#                break
            avg_cost += c / total_batch
        test_acc = sess.run(accuracy, 
                       feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
