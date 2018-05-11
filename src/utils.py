#!/usr/bin/env python
from sklearn.datasets import make_classification, make_blobs
from sklearn.datasets import make_circles, make_moons
import sklearn
import numpy 
import pylab
import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
from itertools import cycle
import numpy as np
import tensorflow as tf


class batch_object(object):
        """docstring for batch_object"""
        def __init__(self, data_list, batch_size):
                super(batch_object, self).__init__()
                self.data_list = data_list
                self.batch_size = batch_size
                self.pool = cycle(data_list)
                
        def next_batch(self):
                data_batch = list()
                for i in xrange(self.batch_size):
                        data_batch.append(next(self.pool))
                data_batch = np.array(data_batch)
                return data_batch

def to_categorical(label_array,total_classes):
    from sklearn.preprocessing import OneHotEncoder
    enc=OneHotEncoder()
    label_list=[]
    for i in range(len(label_array)):
        label_list.append([label_array[i]])
    return enc.fit_transform(label_list).toarray()

def get_svm(feature, label):
	clf = SVC(kernel='linear')
	clf.fit(feature, label)
	return clf


# This binary classification should be linear separable
def play_iris():
	data = load_iris()
	feature = data.data
	label = data.target
	label[label==2]=1
	# clf = LinearSVC(random_state=0)
	clf = SVC(kernel='linear')
	clf.fit(feature, label)
	scores = cross_val_score(clf, feature, label, cv=5)


def generate_lin_sep_blobs(n_samples, random_state):
	samples = make_blobs(n_samples=n_samples, n_features=2,
		centers=[(0.2, 0.2), (0.8, 0.8)], cluster_std=0.1,
		random_state=random_state)
	# samples = make_blobs(n_samples=n_samples, n_features=2,
	# 	centers=[(20, 20), (80, 80)], cluster_std=10,
	# 	random_state=random_state)
	return samples[0], samples[1]

def draw_from_moons(n_samples, random_state):
	samples = make_moons(n_samples=n_samples, 
		random_state=random_state)
	return samples[0], samples[1]

def draw_from_sectors(size, inner, outer, random_state):
	np.random.seed(random_state)
	x1 = np.random.rand(size)*outer
	x2 = np.random.rand(size)*outer
	feature =  np.vstack([x1,x2]).transpose()
	z = np.linalg.norm(feature, ord=2, axis=1)
	feature = feature[(z<=outer) & (z>=inner)]
	return feature

def generate_difficult_sectors(n_samples, random_state):
	# feature0 = draw_from_sectors(n_samples, 0.1, 0.2, random_state)
	# feature1 = draw_from_sectors(n_samples*3, 0.283, 0.4, random_state)
	feature0 = draw_from_sectors(n_samples, 0.1, 200, random_state)
	feature1 = draw_from_sectors(n_samples*3, 300, 400, random_state)
	label0 = np.zeros(len(feature0))
	label1 = np.ones(len(feature1))
	feature = np.concatenate((feature0, feature1), axis=0)
	label = np.concatenate((label0, label1), axis=0)
	return feature, label



def generate_sectors_not_sep(n_samples, random_state):
	feature0 = draw_from_sectors(n_samples, 0.1, 0.2, random_state)
	feature1 = draw_from_sectors(n_samples*3, 0.25, 0.4, random_state)
	# feature0 = draw_from_sectors(n_samples, 0.1, 200, random_state)
	# feature1 = draw_from_sectors(n_samples*3, 240, 400, random_state)
	label0 = np.zeros(len(feature0))
	label1 = np.ones(len(feature1))
	feature = np.concatenate((feature0, feature1), axis=0)
	label = np.concatenate((label0, label1), axis=0)
	return feature, label

def plot_difficult_sectors(feature, label, name=None):
	pylab.figure()
	red = feature[label == 0]
	blue = feature[label == 1]
	pylab.plot(red[:, 0], red[:, 1], 'g.')
	pylab.plot(blue[:, 0], blue[:, 1], 'k.')
	pylab.plot([0,290],[290,0], linewidth=3.5)
	pylab.xticks(fontsize=17)
	pylab.yticks(fontsize=17)
	pylab.title('Traning_data', fontsize=17)
	if name==None:
		pylab.show()
	else:
		pylab.savefig(name)

def plot_blobs(feature, label, name=None):
	pylab.figure()
	red = feature[label == 0]
	blue = feature[label == 1]
	pylab.plot(red[:, 0], red[:, 1], 'g.')
	pylab.plot(blue[:, 0], blue[:, 1], 'k.')
	pylab.xticks(fontsize=17)
	pylab.yticks(fontsize=17)
	pylab.title(name.split('/')[-1].split('.')[0],
		fontsize=17)
	if name==None:
		pylab.show()
	else:
		pylab.savefig(name)

def plot_blobs_all_together(feature, label, feature_t, label_t, 
	name=None):
	pylab.figure()
	red = feature[label == 0]
	blue = feature[label == 1]
	green = feature_t[label_t == 0]
	black = feature_t[label_t == 1]
	pylab.plot(red[:, 0], red[:, 1], 'r.')
	pylab.plot(blue[:, 0], blue[:, 1], 'b.')
	pylab.plot(green[:, 0], green[:, 1], 'g.')
	pylab.plot(black[:, 0], black[:, 1], 'k.')
	pylab.xticks(fontsize=17)
	pylab.yticks(fontsize=17)
	pylab.title(name.split('/')[-1].split('.')[0],
		fontsize=17)
	if name==None:
		pylab.show()
	else:
		pylab.savefig(name)

def boundary_overlay(feature, label, feature_t, label_t, 
	name=None):
	pylab.figure()
	red = feature[label == 0]
	blue = feature[label == 1]
	green = feature_t[label_t == 0]
	black = feature_t[label_t == 1]
	pylab.plot(red[:, 0], red[:, 1], 'r.', alpha=0.1)
	pylab.plot(blue[:, 0], blue[:, 1], 'b.', alpha=0.1)
	pylab.plot(green[:, 0], green[:, 1], 'y.', alpha=0.05)
	pylab.plot(black[:, 0], black[:, 1], 'c.', alpha=0.05)
	pylab.xticks(fontsize=17)
	pylab.yticks(fontsize=17)
	pylab.title(name.split('/')[-1].split('.')[0],
		fontsize=17)
	if name==None:
		pylab.show()
	else:
		pylab.savefig(name)


def lin_sep_with_ground_truth(size, margin, random_state):
	np.random.seed(random_state)
	x1 = np.random.rand(size)
	x2 = np.random.rand(size)
	# x1 = np.random.rand(size)*100
	# x2 = np.random.rand(size)*100
	# margin = margin*100
	center = 1.0
	# center = center*100
	feature =  np.vstack([x1,x2]).transpose()
	z = np.sum(feature, 1)
	feature = feature[(z<=center-margin) | (z>=center+margin)]
	z = z[(z<=center-margin) | (z>=center+margin)]
	ind_1 = np.where(z>=center+margin)[0]
	ind_2 = np.where(z<=center-margin)[0]
	label = np.zeros(len(feature))
	label[ind_1] = 0
	label[ind_2] = 1
	return feature, label


def non_lin_sep(size, margin, random_state):
	np.random.seed(random_state)
	x1 = np.random.rand(size)
	x2 = np.random.rand(size)
	feature =  np.vstack([x1,x2]).transpose()
	z = np.linalg.norm(feature-[0.5,0.5], ord=2, axis=1)
	feature = feature[(z<=0.2-margin) | (z>=0.2+margin)]
	z = z[(z<=0.2-margin) | (z>=0.2+margin)]
	ind_1 = np.where(z>=0.2+margin)[0]
	ind_2 = np.where(z<=0.2-margin)[0]
	label = np.zeros(len(feature))
	label[ind_1] = 0
	label[ind_2] = 1
	return feature, label



def visual_separable_binary(feature,label,margin, name=None):
	pylab.figure()
	pylab.scatter(feature[label==0,0],feature[label==0,1])
	pylab.scatter(feature[label==1,0],feature[label==1,1])
	pylab.plot([0, 1.0+margin], [1.0+margin, 0])
	pylab.plot([0, 1.0-margin], [1.0-margin, 0])
	pylab.xlim(0,1)
	pylab.ylim(0,1)
	if name==None:
		pylab.show()
	else:
		pylab.savefig(name)

def random_points(start, end, size, random_state):
	np.random.seed(random_state)
	x1 = np.random.uniform(start, end, size)
	x2 = np.random.uniform(start, end, size)
	feature =  np.vstack([x1,x2]).transpose()
	label = np.random.choice(2,size)
	return feature, label
	
# help function for residual network
def weights_init(shape):
    '''
    Weights initialization helper function.
    
    Input(s): shape - Type: int list, Example: [5, 5, 32, 32], This parameter is used to define dimensions of weights tensor
    
    Output: tensor of weights in shape defined with the input to this function
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def bias_init(shape, bias_value=0.01):
    '''
    Bias initialization helper function.
    
    Input(s): shape - Type: int list, Example: [32], This parameter is used to define dimensions of bias tensor.
              bias_value - Type: float number, Example: 0.01, This parameter is set to be value of bias tensor.
    
    Output: tensor of biases in shape defined with the input to this function
    '''
    return tf.Variable(tf.constant(bias_value, shape=shape))

def conv2d_custom(input, filter_size, num_of_channels, num_of_filters, activation=tf.nn.relu, dropout=None,
                  padding='SAME', max_pool=True, strides=(1, 1)):  
    '''
    This function is used to define a convolutional layer for a network,
    
    Input(s): input - this is input into convolutional layer (Previous layer or an image)
              filter_size - also called kernel size, kernel is moved (convolved) across an image. Example: 3
              number_of_channels - how many channels the input tensor has
              number_of_filters - this is hyperparameter, and this will set one of dimensions of the output tensor from 
                                  this layer. Note: this number will be number_of_channels for the layer after this one
              max_pool - if this is True, output tensor will be 2x smaller in size. Max pool is there to decrease spartial 
                        dimensions of our output tensor, so computation is less expensive.
              padding - the way that we pad input tensor with zeros ("SAME" or "VALID")
              activation - the non-linear function used at this layer.
              
              
    Output: Convolutional layer with input parameters.
    '''
    weights = weights_init([filter_size, filter_size, num_of_channels, num_of_filters])
    bias = bias_init([num_of_filters])
    
    layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides[0], strides[1], 1], padding=padding) + bias
    
    if activation != None:
        layer = activation(layer)
    
    if max_pool:
        layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
        
    return layer

def flatten(layer):
    '''
    This method is used to convert convolutional output (4 dimensional tensor) into 2 dimensional tensor.
    
    Input(s): layer - the output from last conv layer in your network (4d tensor)
    
    Output(s): reshaped - reshaped layer, 2 dimensional matrix
               elements_num - number of features for this layer
    '''
    shape = layer.get_shape()
    
    num_elements_ = shape[1:4].num_elements()
    
    flattened_layer = tf.reshape(layer, [-1, num_elements_])
    return flattened_layer, num_elements_

def dense_custom(input, input_size, output_size, activation=tf.nn.relu, dropout=None):
    '''
    This function is used to define a fully connected layer for a network,
    
    Input(s): input - this is input into fully connected (Dense) layer (Previous layer or an image)
              input_size - how many neurons/features the input tensor has. Example: input.shape[1]
              output_shape - how many neurons this layer will have
              activation - the non-linear function used at this layer.    
              dropout - the regularization method used to prevent overfitting. The way it works, we randomly turn off
                        some neurons in this layer
              
    Output: fully connected layer with input parameters.
    '''
    weights = weights_init([input_size, output_size])
    bias = bias_init([output_size])
    
    layer = tf.matmul(input, weights) + bias
    
    if activation != None:
        layer = activation(layer)
    
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
        
    return layer

def residual_unit(layer, channels):
    '''
    Input(s): layer - conv layer before this res unit
    
    Output(s): ResUnit layer - implemented as described in the paper
    '''
    step1 = tf.layers.batch_normalization(layer)
    step2 = tf.nn.relu(step1)
    step3 = conv2d_custom(step2, 3, channels, channels, activation=None, max_pool=False) #32 number of feautres is hyperparam
    step4 = tf.layers.batch_normalization(step3)
    step5 = tf.nn.relu(step4)
    step6 = conv2d_custom(step5, 3, channels, channels, activation=None, max_pool=False)
    return layer + step6