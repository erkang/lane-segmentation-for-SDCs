# -*- coding: utf-8 -*-
# @Author: Rishabh Thukral
# @Date:   2017-06-12 12:33:49
# @Last Modified by:   Rishabh Thukral
# @Last Modified time: 2018-06-14 04:48:24

### Basic python modules
import os.path
import numpy as np
import matplotlib.pyplot as plt
### end

### custom imports for DL
import tensorflow as tf
import helper
from helper import (
	weight_variable, 
	bias_variable, 
	conv2d, 
	conv2d_transpose_strided
)
import project_tests as tests
### end


# [START Function to load the selected layers of VGG Network]
def load_vgg(sess, vgg_path):

	"""
	Load Pretrained VGG Model into TensorFlow.
	:param sess: TensorFlow Session
	:param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
	:return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer3_out)
	""" 
	_ = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
	
	return (tf.get_default_graph().get_tensor_by_name( r + ":0") for r in ['image_input','keep_prob','layer3_out','layer4_out','layer7_out'])
tests.test_load_vgg(load_vgg, tf)
# [END]

# [START Function to implement the layers of the new network using skip layer architecture.]
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
	"""
	Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
	:param vgg_layer7_out: TF Tensor for VGG Layer 3 output
	:param vgg_layer4_out: TF Tensor for VGG Layer 4 output
	:param vgg_layer3_out: TF Tensor for VGG Layer 7 output
	:param num_classes: Number of classes to classify
	:return: The Tensor for the last layer of output
	"""
	W8 = weight_variable([1, 1, 4096, num_classes], name="W8_1")
	b8 = bias_variable([num_classes], name="b8_1")
	conv8 = conv2d(vgg_layer7_out, W8, b8)
		
	deconv_shape1 = vgg_layer4_out.get_shape()
	W_t1 = weight_variable([4, 4, deconv_shape1[3].value, num_classes], name="W_t1")
	b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
	conv_t1 = conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(vgg_layer4_out))
	fuse_1 = tf.add(conv_t1, vgg_layer4_out, name="fuse_1")
	
	deconv_shape2 = vgg_layer3_out.get_shape()
	W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
	b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
	conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(vgg_layer3_out))
	fuse_2 = tf.add(conv_t2, vgg_layer3_out, name="fuse_2")
	
	batch_size = tf.shape(fuse_2)[0]
	height, width = 160, 576
	deconv_shape3 = tf.stack([batch_size, height, width, num_classes])
	
	W_t3 = weight_variable([16, 16, num_classes, deconv_shape2[3].value], name="W_t3")
	b_t3 = bias_variable([num_classes], name="b_t3")
	conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
	return conv_t3
tests.test_layers(layers)
# [END]

# [START optimizer function to make the optimizations to the model. Function where magic happens :P]
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
	"""
	Build the TensorFLow loss and optimizer operations.
	:param nn_last_layer: TF Tensor of the last layer in the neural network
	:param correct_label: TF Placeholder for the correct label image
	:param learning_rate: TF Placeholder for the learning rate
	:param num_classes: Number of classes to classify
	:return: Tuple of (logits, train_op, cross_entropy_loss)
	"""
	logits = tf.reshape(nn_last_layer, (-1, num_classes))
	correct_label = tf.reshape(correct_label, (-1, num_classes))
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label, name="entropy"))

	optimizer = tf.train.AdamOptimizer(learning_rate)
	# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

	grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
	train_op = optimizer.apply_gradients(grads)
	
	return logits, train_op, loss
tests.test_optimize(optimize)
# [END]

# [START Function to train the network with given parameters.]
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
			 correct_label, keep_prob, learning_rate):
	"""
	Train neural network and print out the loss during training.
	:param sess: TF Session
	:param epochs: Number of epochs
	:param batch_size: Batch size
	:param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
	:param train_op: TF Operation to train the neural network
	:param cross_entropy_loss: TF Tensor for the amount of loss
	:param input_image: TF Placeholder for input images
	:param correct_label: TF Placeholder for label images
	:param keep_prob: TF Placeholder for dropout keep probability
	:param learning_rate: TF Placeholder for learning rate
	"""
	
	lossess = []
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for idx, (train_images, train_gt) in enumerate(get_batches_fn(batch_size)):
			feed_dict = {input_image: train_images, correct_label: train_gt, keep_prob: 0.85}
			sess.run(train_op, feed_dict=feed_dict)
			
			if idx%5 == 0:
				train_loss = sess.run([cross_entropy_loss], feed_dict=feed_dict)
				print("[{}/{}] Loss: {:.2f}".format(idx, epoch, train_loss[0]))
				lossess.append(train_loss[0])

	return lossess
tests.test_train_nn(train_nn)
# [END]


		
# [START Main function that does all the work]
if __name__ == '__main__':
	num_classes = 2
	image_shape = (160, 576)
	data_dir = './data'
	runs_dir = './runs'
	tests.test_for_kitti_dataset(data_dir)

	# Download pretrained vgg model
	helper.maybe_download_pretrained_vgg(data_dir)

	# OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
	# You'll need a GPU with at least 10 teraFLOPS to train on.
	# https://www.cityscapes-dataset.com/

	tf.reset_default_graph()
		
	vgg_path = './data/vgg'
	epochs, batch_size, learning_rate = 20, 8, 1e-4
	with tf.Session() as sess:
		# Create function to get batches
		get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
		
		# OPTIONAL: Augment Images for better results
		#  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
		
		correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
		
		# TODO: Build NN using load_vgg, layers, and optimize function
		input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
		layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
		logits, train_op, cross_entropy_loss = optimize(layers_output, correct_label, learning_rate, num_classes)
		
		# TODO: Train NN using the train_nn function
		lossess = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
		
		# TODO: Save inference data using helper.save_inference_samples
		helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
	
	plt.plot(lossess)
	plt.ylim([0, 2])
	plt.show()
# [END]
