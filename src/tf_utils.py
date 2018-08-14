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


def batch_normalization_layer(input_layer):
	'''
	Helper function to do batch normalization
	:param input_layer: 4D tensor
	:return: the 4D tensor after being normalized
	'''
	dimension = input_layer.get_shape().as_list()[-1]
	mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
	beta = tf.get_variable('beta', dimension, tf.float32,
	                           initializer=tf.constant_initializer(0.0, tf.float32))
	gamma = tf.get_variable('gamma', dimension, tf.float32,
	                            initializer=tf.constant_initializer(1.0, tf.float32))
	bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
	return bn_layer

def bn_relu_conv_dropout(current, in_features, out_features, kernel_size, stride=1, keep_prob=1):
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
	current = batch_normalization_layer(current)
	current = tf.nn.relu(current)
	current = conv2d(current, in_features, out_features, kernel_size, stride)
	current = tf.nn.dropout(current, keep_prob)
	return current

def conv_bn_relu(input_layer, filter_shape, stride):
	'''
	A helper function to conv, batch normalize and relu the input tensor sequentially
	:param input_layer: 4D tensor
	:param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
	:param stride: stride size for conv
	:return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
	'''
	current = conv2d(input_layer, filter_shape[-2], filter_shape[-1], 
		filter_shape[0], stride)
	bn_layer = batch_normalization_layer(current)
	output = tf.nn.relu(bn_layer)
	return output

def conv_bn_relu_dropout(input_layer, filter_shape, stride, keep_prob):
	'''
	A helper function to conv, batch normalize and relu the input tensor sequentially
	:param input_layer: 4D tensor
	:param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
	:param stride: stride size for conv
	:return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
	'''
	output = conv_bn_relu(input_layer, filter_shape, stride)
	output = tf.nn.dropout(output, keep_prob)
	return output

def densenet_block(input, layers, in_features, growth, keep_prob):
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
			tmp = bn_relu_conv_dropout(current, features, growth, 3, keep_prob=keep_prob)
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

def residual_block(input_layer, output_channel, keep_prob=1 ,first_block=False, SE=False, ratio=16):
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
			conv1 = bn_relu_conv_dropout(input_layer, input_channel, output_channel, 3,
				stride, keep_prob)

	with tf.variable_scope('conv2_in_block'):
		conv2 = bn_relu_conv_dropout(conv1, output_channel, output_channel, 3, 1,
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



def mcf_regularizer(inputs, x_trans, output):
	'''
	mcf regularizer using dropout, scale up method
	using chain rule, only need to calculate the diagonal of the hesssian matrix
	or the last hidden layer and the derivative of the original input
	suppose the x_trans is the hidden representation
	we deal with all the batch samples together
	the variance
	:params inputs: the model input. It is used to calculate the variance
	:params x_trans: the hidden representation
	:params output: the logit output of the whole network.
	:return: the MCF regularizer
	'''
	q = 0.2
	x_flat = tf.reshape(inputs, [-1,np.prod(inputs.shape.as_list()[1:])])
	sigma = tf.square(x_flat)*q/(1-q)

	# The hessian approximation
	hidden_dim = x_trans.shape.as_list()[-1]
	# sample_id = 0
	z_der_list = list()

	# retrieve the prediction
	pred = tf.nn.softmax(output)
	sq_w_coeff = pred*(1-pred)

	# retrieve the last layer weight
	# w_sq_scaled: ?*hidden_dim
	w_last = [v for v in tf.trainable_variables() if 'last_layer/fc_weights:0' in v.name][0]
	w_last_sq = tf.square(w_last)
	w_sq_scaled = tf.matmul(sq_w_coeff, tf.transpose(w_last_sq))

	# w_sq_scaled[sample_id, :] should be the coefficient for each sample

	# change it to tf.while_loop() to accelerate
	for i in range(hidden_dim):
		der_tmp = tf.square(tf.gradients(x_trans[:, i], inputs)[0])
		# multiple the second derivate from loss to z coefficient
		z_der_list.append(der_tmp)

	z_der_sum = tf.convert_to_tensor(z_der_list)
	z_der_sum = tf.reshape(z_der_sum, [hidden_dim, 
		-1, np.prod(z_der_sum.shape.as_list()[2:])])*tf.expand_dims(
		tf.transpose(w_sq_scaled), -1)
	z_der_sum = tf.reduce_sum(z_der_sum, 0)

	# multiple the second derivative and first derivative
	mcf_reg = tf.reduce_sum(sigma*z_der_sum)
	return mcf_reg

# finish the idea of using a set of the hidden representation to accelerate
def mcf_reg_sto(inputs, x_trans, output, sel_dim_num = 2):
	'''
	mcf regularizer using dropout, scale up method with stochastic sampling
	:params inputs: the model input. It is used to calculate the variance
	:params x_trans: the hidden representation
	:params output: the logit output of the whole network.
	:return: the MCF regularizer
	'''
	q = 0.2
	x_flat = tf.reshape(inputs, [-1,np.prod(inputs.shape.as_list()[1:])])
	sigma = tf.square(x_flat)*q/(1-q)

	# The hessian approximation
	hidden_dim = x_trans.shape.as_list()[-1]
	dim_id_list = np.random.choice(hidden_dim, sel_dim_num, replace=False)
	dim_id_list = sorted(list(dim_id_list))

	z_der_list = list()

	# retrieve the prediction
	pred = tf.nn.softmax(output)
	sq_w_coeff = pred*(1-pred)

	# retrieve the last layer weight
	# w_sq_scaled: ?*hidden_dim
	w_last = [v for v in tf.trainable_variables() if 'last_layer/fc_weights:0' in v.name][0]
	w_last_sq = tf.square(w_last)
	w_sq_scaled = tf.matmul(sq_w_coeff, tf.transpose(w_last_sq))
	dim_id_list_gather = list()

	# w_sq_scaled[sample_id, :] should be the coefficient for each sample

	# change it to tf.while_loop() to accelerate
	for i in list(dim_id_list):
		der_tmp = tf.square(tf.gradients(x_trans[:, i], inputs)[0])
		# multiple the second derivate from loss to z coefficient
		z_der_list.append(der_tmp)
		dim_id_list_gather.append([i])

	z_der_sum = tf.convert_to_tensor(z_der_list)
	z_der_sum = tf.reshape(z_der_sum, [sel_dim_num, 
		-1, np.prod(z_der_sum.shape.as_list()[2:])])*tf.expand_dims(
		tf.gather_nd(tf.transpose(w_sq_scaled), dim_id_list_gather), -1)
	z_der_sum = tf.reduce_sum(z_der_sum, 0)

	# multiple the second derivative and first derivative
	mcf_reg = tf.reduce_sum(sigma*z_der_sum)

	mcf_reg = mcf_reg*np.square(hidden_dim*(1.0/sel_dim_num))
	return mcf_reg

def mcf_reg_internal(inputs, internal_layer, output, sel_dim_num = 8):
	'''
	mcf regularizer using dropout, scale up method
	using chain rule, only need to calculate the diagonal of the hesssian matrix
	or the last hidden layer and the derivative of the original input
	suppose the x_trans is the hidden representation
	we deal with all the batch samples together
	the variance
	:params inputs: the model input. It is used to calculate the variance
	:params x_trans: the hidden representation
	:params output: the logit output of the whole network.
	:return: the MCF regularizer
	'''
	x_trans = internal_layer[-1]
	mcf_layers = internal_layer[:-1]

	q = 0.2
	mcf_layers_flat = map(lambda x: 
			tf.reshape(x, [-1, np.prod(x.shape.as_list()[1:])]), mcf_layers)
	mcf_layers_flat = tf.concat(mcf_layers_flat, 1)
	sigma = tf.square(mcf_layers_flat)*q/(1-q)

	# The hessian approximation
	hidden_dim = x_trans.shape.as_list()[-1]
	dim_id_list = np.random.choice(hidden_dim, sel_dim_num, replace=False)
	dim_id_list = sorted(list(dim_id_list))
	# sample_id = 0
	z_der_list = list()

	# retrieve the prediction
	pred = tf.nn.softmax(output)
	sq_w_coeff = pred*(1-pred)

	# retrieve the last layer weight
	# w_sq_scaled: ?*hidden_dim
	w_last = [v for v in tf.trainable_variables() if 'last_layer/fc_weights:0' in v.name][0]
	w_last_sq = tf.square(w_last)
	w_sq_scaled = tf.matmul(sq_w_coeff, tf.transpose(w_last_sq))
	dim_id_list_gather = list()
	# w_sq_scaled[sample_id, :] should be the coefficient for each sample

	# change it to tf.while_loop() to accelerate
	for i in dim_id_list:
		der_tmp = tf.gradients(x_trans[:, i], mcf_layers)
		# multiple the second derivate from loss to z coefficient
		der_tmp = map(lambda x: tf.reshape(x, [-1, np.prod(x.shape.as_list()[1:])]), der_tmp)
		der_tmp = tf.concat(der_tmp, 1)
		der_tmp = tf.square(der_tmp)
		z_der_list.append(der_tmp)
		dim_id_list_gather.append([i])

	z_der_sum = tf.convert_to_tensor(z_der_list)
	z_der_sum = z_der_sum * tf.expand_dims(
		tf.gather_nd(tf.transpose(w_sq_scaled), dim_id_list_gather), -1)
	z_der_sum = tf.reduce_sum(z_der_sum, 0)

	# multiple the second derivative and first derivative
	mcf_reg = tf.reduce_sum(sigma*z_der_sum)

	mcf_reg = mcf_reg*np.square(hidden_dim*(1.0/sel_dim_num))
	return mcf_reg




