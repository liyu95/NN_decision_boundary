import time
import numpy as np
import tflearn
from sklearn.cross_validation import train_test_split
import random
import cPickle
import copy
import math
import sys
import tensorflow as tf
import os

batch_size = 30
train_steps = 3000

#functions to generate variables, like weight and bias
def weight_variable(shape):
    import math
    if len(shape)>2:
        weight_std=math.sqrt(2.0/(shape[0]*shape[1]*shape[2]))
    else:
        weight_std=0.01
    initial=tf.truncated_normal(shape,stddev=weight_std)
    return tf.Variable(initial,name='weights')

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name='bias')

#functions to generate convolutional layer and pooling layer
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def conv1d(x,W):
	return tf.nn.conv1d(x,W,stride=1,padding='SAME')

def aver_pool2d(x,row,col):
    #Be careful about the dimensionality reduction of pooling and strides setting
    return tf.nn.avg_pool(x,ksize=[1,row,col,1],strides=[1,row,col,1],padding='SAME')

def max_pool2d(x,row,col):
    return tf.nn.max_pool(x,ksize=[1,row,col,1],strides=[1,row,col,1],padding='SAME')

def model_graph(x, y_):
	with tf.name_scope('fc_1'):
	    #Add the third densely connected layer
	    w_fc1=weight_variable([x.get_shape().as_list()[1],2000])
	    b_fc1=bias_variable([2000])
	    h_fc1=tflearn.softsign(tf.matmul(x,w_fc1)+b_fc1)

	# with tf.name_scope('conv_1d_1'):
	#     #Add the third densely connected layer
	#     h_fc1 = tf.expand_dims(h_fc1, -1)
	#     w_conv1=weight_variable([10, 1, 32])
	#     b_conv1=bias_variable([32])
	#     h_fc1=tflearn.relu(conv1d(h_fc1,w_conv1)+b_conv1)
	#     h_fc1=tf.reshape(h_fc1,[-1,
	#     	np.prod(h_fc1.get_shape().as_list()[1:])])

	with tf.name_scope('fc_2'):
	    #Add the third densely connected layer
	    w_fc2=weight_variable([
	    	h_fc1.get_shape().as_list()[1],2])
	    b_fc2=bias_variable([2])
	    h_fc1=tflearn.softsign(tf.matmul(h_fc1,w_fc2)+b_fc2)

	# with tf.name_scope('fc_3'):
	#     #Add the third densely connected layer
	#     w_fc3=weight_variable([2000,2000])
	#     b_fc3=bias_variable([2000])
	#     h_fc1=tflearn.relu(tf.matmul(h_fc1,w_fc3)+b_fc3)

	# with tf.name_scope('fc_4'):
	#     #Add the third densely connected layer
	#     w_fc4=weight_variable([2000,2000])
	#     b_fc4=bias_variable([2000])
	#     h_fc1=tflearn.relu(tf.matmul(h_fc1,w_fc4)+b_fc4)

	#ADD SOFTMAX LAYER
	with tf.name_scope('softmax_layer'):
	    w_s=weight_variable([h_fc1.get_shape().as_list()[1],
	    	y_.get_shape().as_list()[1]])
	    b_s=bias_variable([y_.get_shape().as_list()[1]])
	    y_conv_logit=tf.matmul(h_fc1,w_s)+b_s
	    y_conv=tf.nn.softmax(y_conv_logit)
	return y_conv_logit, y_conv, h_fc1

def model_graph_asym(x, y_):
	with tf.name_scope('fc_1'):
	    #Add the third densely connected layer
	    w_fc1=weight_variable([x.get_shape().as_list()[1],2000])
	    b_fc1=bias_variable([2000])
	    h_fc1=tflearn.relu(tf.matmul(x,w_fc1)+b_fc1)

	with tf.name_scope('fc_2'):
	    #Add the third densely connected layer
	    w_fc2=weight_variable([
	    	h_fc1.get_shape().as_list()[1],2000])
	    b_fc2=bias_variable([2000])
	    h_fc1=tflearn.relu(tf.matmul(h_fc1,w_fc2)+b_fc2)

	with tf.name_scope('softmax_layer'):
	    w_s1=weight_variable([1500,1])
	    w_s2=weight_variable([502,1])
	    b_s=bias_variable([y_.get_shape().as_list()[1]])
	    y_conv_logit1 = tf.matmul(h_fc1[:,:1500], w_s1)
	    y_conv_logit2 = tf.matmul(
	    	tf.concat([h_fc1[:,1500:], x], 1), w_s2)
	    y_conv_logit=tf.concat([y_conv_logit1,
	    	y_conv_logit2],1)+b_s
	    y_conv=tf.nn.softmax(y_conv_logit)
	return y_conv_logit, y_conv

