import tensorflow as tf
import layers

		
def multilayer_perceptron(input, units_list):
	output = input
	for l in range(len(units_list)):
		output = tf.layers.dense(output, units_list[l], activation=tf.nn.tanh)
		
	return input, output
	
	
def binary_multilayer_perceptron(input, units_list):
	output = input
	for l in range(len(units_list)-1):
		output = layers.binaryDense(output, units_list[l], activation=None, name='binarydense'+str(l))
		output = tf.layers.batch_normalization(output, training=True)
		output = tf.clip_by_value(output, -1, 1)
	output = layers.binaryDense(output, units_list[l+1], activation=None, name='binarydense'+str(len(units_list)-1))
	output = tf.contrib.layers.batch_norm(output)
	return input, output

	
def cifar10(input, training=True):
	out = tf.layers.conv2d(input, 128, [3,3], [1,1], padding='VALID', use_bias=False, name='c_conv2d_1')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 128, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_1')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 256, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_2')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 256, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_3')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 512, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_4')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.conv2d(out, 512, [3,3], [1,1], padding='SAME', use_bias=False, name='conv2d_5')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.flatten(out)
	out = tf.layers.dense(out, 1024, use_bias=False, name='dense_1')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.dense(out, 1024, use_bias=False, name='dense_2')
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.dense(out, 10, name='dense_3')
	output = tf.layers.batch_normalization(out, training=training)
	return input, output

	
def binary_cifar10(input, training=True):
	# This function is used only at the first layer of the model as we dont want to binarized the RGB images
	out = layers.binaryConv2d(input, 4, [3,3], [1,1], padding='VALID', use_bias=True, binarize_input=False, name='bc_conv2d_1_layer1')
	out = tf.layers.batch_normalization(out, training=training, name="batchNormalization1_layer1")
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_1_layer2')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.layers.batch_normalization(out, training=training, name="batchNormalization2_layer2")
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_2_layer3')
	out = tf.layers.batch_normalization(out, training=training, name="batchNormalization3_layer3")
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_3_layer4')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.layers.batch_normalization(out, training=training, name="batchNormalization4_layer4")
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_4_layer5')
	out = tf.layers.batch_normalization(out, training=training, name="batchNormalization5_layer5")
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_5_layer6')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = tf.layers.batch_normalization(out, training=training, name="batchNormalization6_layer6")
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 4, use_bias=True, name='binary_dense_1_layer7')
	out = tf.layers.batch_normalization(out, training=training, name="batchNormalization7_layer7")
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 4, use_bias=True, name='binary_dense_2_layer8')
	out = tf.layers.batch_normalization(out, training=training, name="batchNormalization8_layer8")
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 10, name='binary_dense_3_layer9')
	output = tf.layers.batch_normalization(out, training=training, name="batchNormalization9_layer9")
	
	return input, output
	

def binary_cifar10_sbn(input, training=True):
	out = layers.binaryConv2d(input, 4, [3,3], [1,1], padding='VALID', use_bias=True, binarize_input=False, name='bc_conv2d_1_layer1')
	out = layers.spatial_shift_batch_norm(out, training=training, name='batchNormalization1_layer1')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_1_layer2')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = layers.spatial_shift_batch_norm(out, training=training, name='batchNormalization2_layer2')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_2_layer3')
	out = layers.spatial_shift_batch_norm(out, training=training, name='batchNormalization3_layer3')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_3_layer4')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = layers.spatial_shift_batch_norm(out, training=training, name='batchNormalization4_layer4')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_4_layer5')
	out = layers.spatial_shift_batch_norm(out, training=training, name='batchNormalization5_layer5')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryConv2d(out, 4, [3,3], [1,1], padding='SAME', use_bias=True, name='bnn_conv2d_5_layer6')
	out = tf.layers.max_pooling2d(out, [2,2], [2,2])
	out = layers.spatial_shift_batch_norm(out, training=training, name='batchNormalization6_layer6')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 4, use_bias=True, name='binary_dense_1_layer7')
	out = layers.shift_batch_norm(out, training=training, name='batchNormalization7_layer7')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 4, use_bias=True, name='binary_dense_2_layer8')
	out = layers.shift_batch_norm(out, training=training, name='batchNormalization8_layer8')
	out = tf.clip_by_value(out, -1, 1)
	out = layers.binaryDense(out, 10, name='binary_dense_3_layer9')
	output = layers.shift_batch_norm(out, training=training, name='batchNormalization9_layer9')
	
	return input, output

	

def mnist(input, training=True):
	out = tf.layers.dense(input, 2048, activation=None)
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.dense(out, 2048, activation=None)
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.dense(out, 2048, activation=None)
	out = tf.layers.batch_normalization(out, training=training)
	out = tf.nn.relu(out)
	out = tf.layers.dense(out, 10, activation=None)
	output = tf.layers.batch_normalization(out, training=training)
	
	return input, output
	

def binary_mnist(input, training=True):
	fc1 = layers.binaryDense(input, 32, activation=None, name="binarydense1_layer1", binarize_input=False)
	bn1 = tf.layers.batch_normalization(fc1, training=training, name="batchNormalization1_layer1")
	ac1 = tf.clip_by_value(bn1, -1, 1)
	fc2 = layers.binaryDense(ac1, 32, activation=None, name="binarydense2_layer2")
	bn2 = tf.layers.batch_normalization(fc2, training=training, name="batchNormalization2_layer2")
	ac2 = tf.clip_by_value(bn2, -1, 1)
	fc3 = layers.binaryDense(ac2, 32, activation=None, name="binarydense3_layer3")
	bn3 = tf.layers.batch_normalization(fc3, training=training, name="batchNormalization3_layer3")
	ac3 = tf.clip_by_value(bn3, -1, 1)
	fc4 = layers.binaryDense(ac3, 10, activation=None, name="binarydense4_layer4")
	output =  tf.layers.batch_normalization(fc4, training=training, name="batchNormalization4_layer4")
	
	return input, output
	
	
def binary_mnist_sbn(input, training=True):
	fc1 = layers.binaryDense(input, 32, activation=None, name="binarydense1_layer1", binarize_input=False)
	bn1 = layers.shift_batch_norm(fc1, training=training, name="batchNormalization1_layer1")
	ac1 = tf.clip_by_value(bn1, -1, 1)
	fc2 = layers.binaryDense(ac1, 32, activation=None, name="binarydense2_layer2")
	bn2 = layers.shift_batch_norm(fc2, training=training, name="batchNormalization2_layer2")
	ac2 = tf.clip_by_value(bn2, -1, 1)
	fc3 = layers.binaryDense(ac2, 32, activation=None, name="binarydense3_layer3")
	bn3 = layers.shift_batch_norm(fc3, training=training, name="batchNormalization3_layer3")
	ac3 = tf.clip_by_value(bn3, -1, 1)
	fc4 = layers.binaryDense(ac3, 10, activation=None, name="binarydense4_layer4")
	output = layers.shift_batch_norm(fc4, training=training, name="batchNormalization4_layer4")
	
	return input, output
	
	

def get_network(type, dataset, *args, **kargs):

	if dataset == 'mnist':
		if type == 'standard':
			return mnist(*args, **kargs)
		if type == 'binary':
			return binary_mnist(*args, **kargs)
		if type == 'binary_sbn':
			return binary_mnist_sbn(*args, **kargs)
	
	if dataset == 'cifar10':
		if type == 'standard':
			return cifar10(*args, **kargs)
		if type == 'binary':
			return binary_cifar10(*args, **kargs)
		if type == 'binary_sbn':
			return binary_cifar10_sbn(*args, **kargs)
	
	return None
	

		