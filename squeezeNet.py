import tensorflow as tf

class squeezelayers:
	"""
	Implementation of squeezelayers model demonstrated in the paper: https://arxiv.org/abs/1602.07360
	
	"""
	def __init__(self, session, alpha, optimizer=tf.train.GradientDescentOptimizer, squeeze_ratio = 0.125):
		self.sess = session
        self.target    = tf.placeholder(tf.float32, [None, 1000])
        self.imgs      = tf.placeholder(tf.float32, [None, 224, 224, 3])

        self.alpha = alpha
        self.sq_ratio  = squeeze_ratio
        self.optimizer = optimizer

        self.weights = {}
        self.layers = {}

        self.build_model()
        self.init_optimizer()
        self.train_model()

    def build_model(self):
    	layers = {}

        # Caffe order is BGR, this model is RGB.
        # The mean values are from caffe protofile from DeepScale/SqueezeNet github repo.
        self.mean = tf.constant([123.0, 117.0, 104.0], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        images = self.imgs - self.mean

        layers['input'] = images

        layers['conv1'] = tf.nn.conv2d(layers['input'], self.get_weight([3, 3, 3, 64], 'conv1'), stride = [1, 2, 2, 1], padding= 'SAME')
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

        layers['conv10'] = tf.nn.conv2d(layers['fire9'], self.get_weight([1, 1, 512, 1000], 'conv10'), stride = [1, 1, 1, 1], padding= 'VALID')
        layers['relu10'] = tf.nn.relu(tf.nn.bias_add(layers['conv10'], self.get_bias([512], 'relu10')))

        layers['avgpool10'] = tf.nn.avg_pool(layers['relu10'], ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='VALID')

        avg_pool_shape = tf.shape(layers['pool10'])
        layers['pool_reshaped'] = tf.reshape(layers['pool10'], [avg_pool_shape[0],-1])

        self.logits = layers['pool_reshaped']
        self.probs = tf.nn.softmax(self.logits)

        self.layers = layers

    def init_optimizer(self):
    	self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))
        self.optimize = self.optimizer(self.alpha).minimize(self.loss)

    def train_model(self, X_train, Y_train):
    	self.sess.run(tf.global_variables_initializer())
    	self.sess.run(self.optimize, feed_dict={self.imgs: X_train, self.target: Y_train})

    def get_weight(self, shape, name, initializer = 'truncated_normal'):
    	if initializer == 'truncated_normal':
    		weight = tf.Variable(tf.truncated_normal(shape), name = 'W ' + name)
    	elif initializer == 'xavier':
        	weight = tf.get_variable(name = 'W ' + name, shape, initializer = tf.contrib.layers.xavier_initializer())
        else:
        	weight = tf.Variable(tf.random_normal(shape), name = 'W ' + name)

        self.weights['W ' + name] = weight
        return self.weights['W ' + name]

    def get_bias(self, shape, name):
    	bias = tf.Variable(tf.truncated_normal(shape), name = 'b ' + name)
    	self.weights['b ' + name] = bias
        return self.weights['b ' + name]

	def fire_module(input, name, s_1x1, e_1x1, e_3x3):
		"""
		Basic building block of Squeezelayers, tt is made up of two layers:
		Assuming input shape is (N x H x W x C), where N is size of training set, H is height, W is width, and C is number of channel

		Squeeze Layer: 1 x 1 convolution with s_1x1 number of filters
		Expand Layer: 1 x 1 convolution with  e_1x1 number of filters, 3 x 3 convolution with  e_3x3 number of filters

		"""
		N, H, W, C =  input.get_shape()

		squeeze = tf.nn.conv2d(input, self.get_weight([1, 1, C, s_1x1], name + '_squeeze_1x1'), stride = [1, 1, 1, 1], padding= 'SAME') 
		squeeze = tf.nn.relu(tf.nn.bias_add(squeeze, self.get_bias([s_1x1], name + '_squeeze_1x1')))

		expand1 = tf.nn.conv2d(squeeze, self.get_weight([1, 1, s_1x1, e_1x1], name + '_expand_1x1'), stride = [1, 1, 1, 1], padding= 'SAME') 
		expand1 = tf.nn.relu(tf.nn.bias_add(squeeze, self.get_bias([e_1x1], name + '_expand_1x1')))

		expand2 = tf.nn.conv2d(squeeze, self.get_weight([1, 1, s_1x1, e_3x3], name + '_expand_3x3'), stride = [1, 1, 1, 1], padding= 'SAME') 
		expand2 = tf.nn.relu(tf.nn.bias_add(expand2, self.get_bias([e_3x3], name + '_expand_3x3')))

		result = tf.concat([expand1, expand3], 3)
		result = tf.nn.relu(result)
	    
	    return result
