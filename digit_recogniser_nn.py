#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 22:57:34 2017

@author: nam
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Getting inputs, and outputs(in one-hot vector form) 
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Defining number of neurons in each layer, number of layers, batch size,
# and number of output classes
num_layers_1 = 500
num_layers_2 = 500
num_layers_3 = 500
num_output_classes = 10
batch_size = 100

# Input and output vectors
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


# Defining the model
def neural_network_model(data):
    # Intialising weights and biases for each layer
    input_layer = {'weights': tf.Variable(tf.random_normal([784, num_layers_1])),
               'biases': tf.Variable(tf.random_normal([num_layers_1]))}
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([num_layers_1, num_layers_2])),
               'biases': tf.Variable(tf.random_normal([num_layers_2]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([num_layers_2, num_layers_3])),
               'biases': tf.Variable(tf.random_normal([num_layers_3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([num_layers_3, num_output_classes])),
               'biases': tf.Variable(tf.random_normal([num_output_classes]))}
    
    #Calculating input to each neuron and then passing it through ReLU activation function
    input_layer_output = tf.add(tf.matmul(data, input_layer['weights']), input_layer['biases'])
    input_layer_output = tf.nn.relu(input_layer_output)
    
    hidden_layer_1_output = tf.add(tf.matmul(input_layer_output, hidden_layer_1['weights']), hidden_layer_1['biases'])
    hidden_layer_1_output = tf.nn.relu(hidden_layer_1_output)
    
    hidden_layer_2_output = tf.add(tf.matmul(hidden_layer_1_output, hidden_layer_2['weights']), hidden_layer_2['biases'])
    hidden_layer_2_output = tf.nn.relu(hidden_layer_2_output)
    
    final_output = tf.add(tf.matmul(hidden_layer_2_output, output_layer['weights']), output_layer['biases'])
    return final_output

# Training the network
def train_network(data):
    # Finding our prediction for input data
    hypothesis = neural_network_model(data)
    # Finding loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
    # Optimizing cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    # Creation of session, and defining number of epochs
    num_epochs = 18
    with tf.Session() as sess:
        # Initialising global variables
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                temp, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss=epoch_loss + c
            print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss)
            
        correct_predictions = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'))
        # Testing our model
        print('Accuracy on test data:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
train_network(x)
        
    
    
    

