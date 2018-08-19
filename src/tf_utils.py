#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np

BN_EPSILON = 0.001
weight_decay = 0.0002

config = tf.ConfigProto()
config.log_device_placement=False
config.allow_soft_placement=True
config.gpu_options.allow_growth=True



def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos 
    		if x.device_type == 'GPU']

def weight_variable(shape, name='weights', initializer=tf.contrib.layers.xavier_initializer(),
	is_fc_layer=False):
	'''
	Help function for defining the weight variables
	:param name: A string. The name of the new variable
	:param shape: A list of dimensions
	:param initializer: User Xavier as default.
	:param is_fc_layer: Bool, whether the weight is for the last layer
	:return: The created variable
	'''
	regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

	new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
	                                regularizer=regularizer)
	return new_variables

def bias_variable(shape):
	'''
	Help function for defining the bias variables
	:param shape: A list, shape of the bias variable
	:return: The created bias variable
	'''
	initial = tf.constant(0.01, shape=shape)
	return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, stride=1, with_bias=False):
	'''
	The 2d convolutional layer without activation
	:param input: A tensor, the input of this layer
	:param in_features: A scalar, the channels of the input
	:param out_features: A scalar, the channels of the output
	:param kernel_size: A scalar, filter size
	:param stride: A scale, the stride size
	:param with_bias: Boolean
	:return: The convolution result
	'''
	W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
	conv = tf.nn.conv2d(input, W, [ 1, stride, stride, 1 ], padding='SAME')
	if with_bias:
		return conv + bias_variable([ out_features ])
	return conv

def avg_pool(input, s):
	'''
	The average pooling layer
	:param input: 4d tensor
	:param s: A scalar, the stride 
	:return: A tensor after average pooling
	'''
	return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

def output_layer(input_layer, num_labels):
	'''
	The fully connected layers
	:param input_layer: 2D tensor
	:param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
	:return: output layer Y = WX + B
	'''
	input_dim = input_layer.get_shape().as_list()[-1]
	fc_w = weight_variable(shape=[input_dim, num_labels], name='fc_weights', is_fc_layer=True,
	                        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
	fc_b = bias_variable(shape=[num_labels])

	fc_h = tf.matmul(input_layer, fc_w) + fc_b
	return fc_h


def batch_normalization_layer(inputs, is_training, decay=0.999):
	'''
	Helper function to do batch normalization
	:param input_layer: 4D tensor
	:param is_training: boolean, do not update the mean and var during testing
	:return: the 4D tensor after being normalized
	'''
	n_out = inputs.get_shape().as_list()[-1]
	with tf.variable_scope('bn'):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
			name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
			name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(inputs, 
			range(len(inputs.get_shape().as_list())-1), name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(is_training,
			mean_var_with_update,
			lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
	return normed

	# return tf.contrib.layers.batch_norm(inputs, scale=True, is_training=is_training,
	# 	updates_collections=None)
	# scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	# beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	# pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	# pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

	# if is_training:
	# 	batch_mean, batch_var = tf.nn.moments(inputs,[0])
	# 	train_mean = tf.assign(pop_mean,
	# 	                       pop_mean * decay + batch_mean * (1 - decay))
	# 	train_var = tf.assign(pop_var,
	# 	                      pop_var * decay + batch_var * (1 - decay))
	# 	with tf.control_dependencies([train_mean, train_var]):
	# 	    return tf.nn.batch_normalization(inputs,
	# 	        batch_mean, batch_var, beta, scale, epsilon)
	# else:
	# 	return tf.nn.batch_normalization(inputs,
	# 	    pop_mean, pop_var, beta, scale, epsilon)

def bn_relu_conv_dropout(current, in_features, out_features, kernel_size, is_training, stride=1, keep_prob=1):
	'''
	A commonly used building block in Resnet and Densenet
	:param current: A tensor, the input tensor
	:param in_features: A scalar, the channels of the input tensor
	:param out_features: A scalar, the channels of the output tensor
	:param kernel size: A scalar, the kernel size
	:param stride: A scalar, the stride size
	:param keep_prob: A scalar, the dropout keep ratio
	:return: the output tensor
	'''
	current = batch_normalization_layer(current, is_training)
	current = tf.nn.relu(current)
	current = conv2d(current, in_features, out_features, kernel_size, stride)
	current = tf.nn.dropout(current, keep_prob)
	return current

def conv_bn_relu(input_layer, filter_shape, stride, is_training):
	'''
	A helper function to conv, batch normalize and relu the input tensor sequentially
	:param input_layer: 4D tensor
	:param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
	:param stride: stride size for conv
	:return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
	'''
	current = conv2d(input_layer, filter_shape[-2], filter_shape[-1], 
		filter_shape[0], stride)
	bn_layer = batch_normalization_layer(current, is_training)
	output = tf.nn.relu(bn_layer)
	return output

def conv_bn_relu_dropout(input_layer, filter_shape, stride, keep_prob, is_training):
	'''
	A helper function to conv, batch normalize and relu the input tensor sequentially
	:param input_layer: 4D tensor
	:param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
	:param stride: stride size for conv
	:return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
	'''
	output = conv_bn_relu(input_layer, filter_shape, stride, is_training)
	output = tf.nn.dropout(output, keep_prob)
	return output

def densenet_block(input, layers, in_features, growth, keep_prob, is_training):
	'''
	Helper function for the densenet block
	:param input: A tensor, the input tensor for the block
	:param layers: A scalar, the number of layers in the block
	:param in_features: A scalar, the number of input channels 
						input_layer.get_shape().as_list()[-1]
	:param growth: A scalar, the growth rate for the number of channels
	:param keep_prob: A scalar, the dropout keep ratio
	:return: the output of the block and the number of channels of the output
	'''
	current = input
	features = in_features
	for idx in xrange(layers):
		with tf.variable_scope('conv_%d' %idx):
			tmp = bn_relu_conv_dropout(current, features, growth, 3, is_training, keep_prob=keep_prob)
			current = tf.concat((current, tmp), axis=3)
			features += growth
	return current, features

def squeeze_excitation_layer(input_x, ratio):
	out_dim = input_x.get_shape().as_list()[-1]
	with tf.variable_scope('SE_layer'):
		squeeze = tf.reduce_mean(input_x, [1, 2])
		with tf.variable_scope('SE_fc_1'):
			excitation = output_layer(squeeze, out_dim / ratio)
			excitation = tf.nn.relu(excitation)
		with tf.variable_scope('SE_fc_2'):
			excitation = output_layer(excitation, out_dim)
			excitation = tf.nn.sigmoid(excitation)

		excitation = tf.reshape(excitation, [-1,1,1,out_dim])
		scale = input_x * excitation
	return scale

def residual_block(input_layer, output_channel, is_training, keep_prob=1 ,first_block=False, SE=False, ratio=16):
	'''
	Defines a residual block in ResNet
	:param input_layer: 4D tensor
	:param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
	:param first_block: if this is the first residual block of the whole network
	:return: 4D tensor.
	'''
	input_channel = input_layer.get_shape().as_list()[-1]

	# When it's time to "shrink" the image size, we use stride = 2
	if input_channel * 2 == output_channel:
		increase_dim = True
		stride = 2
	elif input_channel == output_channel:
		increase_dim = False
		stride = 1
	else:
		raise ValueError('Output and input channel does not match in residual blocks!!!')

	# The first conv layer of the first residual block does not need to be normalized and relu-ed.
	with tf.variable_scope('conv1_in_block'):
		if first_block:
			conv1 = conv2d(input_layer, input_channel, output_channel, 3)
		else:
			conv1 = bn_relu_conv_dropout(input_layer, input_channel, output_channel, 3, is_training,
				stride, keep_prob)

	with tf.variable_scope('conv2_in_block'):
		conv2 = bn_relu_conv_dropout(conv1, output_channel, output_channel, 3, is_training,1,
			keep_prob)

	if SE:
		with tf.variable_scope('se_in_block'):
			conv2 = squeeze_excitation_layer(conv2, ratio)

	# When the channels of input layer and conv2 does not match, we add zero pads to increase the
	#  depth of input layers
	if increase_dim is True:
		pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1], padding='VALID')
		padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
			input_channel // 2]])
	else:
		padded_input = input_layer

	output = conv2 + padded_input
	return output
