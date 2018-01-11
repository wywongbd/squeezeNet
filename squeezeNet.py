import tensorflow as tf
import numpy as np

import math

from matplotlib import pyplot as plt

class SqueezeNet:
	"""
	Implementation of squeezeNet model demonstrated in the paper: https://arxiv.org/abs/1602.07360
	
	"""
	def __init__(self, session, alpha, optimizer=tf.train.AdamOptimizer, squeeze_ratio = 0.125):
		self.sess = session
		self.target = tf.placeholder(tf.float32, [None, 1000])
		self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

		self.alpha = alpha
		self.sq_ratio  = squeeze_ratio
		self.optimizer = optimizer

		self.weights = {}
		self.layers = {}

		self.build_model()
		self.init_optimizer()

	def build_model(self):
		layers = {}

		self.mean = tf.constant([123.0, 117.0, 104.0], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
		# images = self.imgs - self.mean
		layers['input'] = self.imgs

		layers['conv1'] = tf.nn.conv2d(layers['input'], self.get_weight([3, 3, 3, 64], 'conv1'), strides = [1, 2, 2, 1], padding= 'SAME')
		layers['relu1'] = tf.nn.relu(tf.nn.bias_add(layers['conv1'], self.get_bias([64], 'relu1')))

		layers['pool1'] = tf.nn.max_pool(layers['relu1'], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

		layers['fire2'] = self.fire_module('fire2', layers['pool1'], self.sq_ratio * 128, 64, 64)
		layers['fire3'] = self.fire_module('fire3', layers['fire2'], self.sq_ratio * 128, 64, 64)

		layers['pool3'] = tf.nn.max_pool(layers['fire3'], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

		layers['fire4'] = self.fire_module('fire4', layers['pool3'], self.sq_ratio * 256, 128, 128)
		layers['fire5'] = self.fire_module('fire5', layers['fire4'], self.sq_ratio * 384, 192, 192)

		layers['pool5'] = tf.nn.max_pool(layers['fire5'], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

		layers['fire6'] = self.fire_module('fire6', layers['pool5'], self.sq_ratio * 384, 192, 192)
		layers['fire7'] = self.fire_module('fire7', layers['fire6'], self.sq_ratio * 384, 192, 192)
		layers['fire8'] = self.fire_module('fire8', layers['fire7'], self.sq_ratio * 512, 256, 256)
		layers['fire9'] = self.fire_module('fire9', layers['fire8'], self.sq_ratio * 512, 256, 256)

		layers['conv10'] = tf.nn.conv2d(layers['fire9'], self.get_weight([1, 1, 512, 1000], 'conv10'), strides = [1, 1, 1, 1], padding= 'VALID')
		layers['relu10'] = tf.nn.relu(tf.nn.bias_add(layers['conv10'], self.get_bias([1000], 'relu10')))

		layers['avgpool10'] = tf.nn.avg_pool(layers['relu10'], ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='VALID')

		avg_pool_shape = tf.shape(layers['avgpool10'])
		layers['pool_reshaped'] = tf.reshape(layers['avgpool10'], [avg_pool_shape[0],-1])

		self.logits = layers['pool_reshaped'] 
		self.probs = tf.nn.softmax(self.logits)

		self.layers = layers


	def init_optimizer(self):
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))
		self.optimize = self.optimizer(self.alpha).minimize(self.loss)

	def train_model(self, X_train, Y_train, epoch = 10):
		self.correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.target, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		self.sess.run(tf.global_variables_initializer())

		for step in range(1, epoch):
			batch_x, batch_y = X_train, Y_train
			# Run optimization op (backprop)
			self.sess.run(self.optimize, feed_dict={self.imgs: batch_x, self.target: batch_y})
			
			if step % 5 == 0 or step == 1:
				# Calculate batch loss and accuracy
				loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.imgs: batch_x, self.target: batch_y})
				print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

		print("Optimization Finished!")

		# for step in range(10):
		# 	return self.run_model(X_train, Y_train, epochs = epoch)
			# self.sess.run(self.optimize, feed_dict={self.imgs: X_train, self.target: Y_train})


	def get_weight(self, shape, name, initializer = 'truncated_normal'):
		if initializer == 'truncated_normal':
			weight = tf.Variable(tf.truncated_normal(shape), name = 'W_'+ name)
		else:
			weight = tf.Variable(tf.random_normal(shape), name = 'W_'+ name)

		self.weights['W_'+ name] = weight
		return self.weights['W_'+ name]

	def get_bias(self, shape, name):
		bias = tf.Variable(tf.truncated_normal(shape), name = 'b_' + name)
		self.weights['b_' + name] = bias
		return self.weights['b_' + name]

	def fire_module(self, name, input, s_1x1, e_1x1, e_3x3):
		"""
		Basic building block of Squeezelayers, tt is made up of two layers:
		Assuming input shape is (N x H x W x C), where N is size of training set, H is height, W is width, and C is number of channel

		Squeeze Layer: 1 x 1 convolution with s_1x1 number of filters
		Expand Layer: 1 x 1 convolution with  e_1x1 number of filters, 3 x 3 convolution with  e_3x3 number of filters

		"""
		N, H, W, C =  input.get_shape()
		s_1x1, e_1x1, e_3x3 = int(s_1x1), int(e_1x1), int(e_3x3)

		squeeze = tf.nn.conv2d(input, self.get_weight([1, 1, int(C), s_1x1], name + '_squeeze_1x1'), strides = [1, 1, 1, 1], padding= 'SAME') 
		squeeze = tf.nn.relu(tf.nn.bias_add(squeeze, self.get_bias([s_1x1], name + '_squeeze_1x1')))

		expand1 = tf.nn.conv2d(squeeze, self.get_weight([1, 1, s_1x1, e_1x1], name + '_expand_1x1'), strides = [1, 1, 1, 1], padding= 'SAME') 
		expand1 = tf.nn.relu(tf.nn.bias_add(expand1, self.get_bias([e_1x1], name + '_expand_1x1')))

		expand2 = tf.nn.conv2d(squeeze, self.get_weight([1, 1, s_1x1, e_3x3], name + '_expand_3x3'), strides = [1, 1, 1, 1], padding= 'SAME') 
		expand2 = tf.nn.relu(tf.nn.bias_add(expand2, self.get_bias([e_3x3], name + '_expand_3x3')))

		result = tf.concat([expand1, expand2], 3)
		result = tf.nn.relu(result)

		return result
