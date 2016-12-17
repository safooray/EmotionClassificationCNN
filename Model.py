from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from DataIO import *

from sklearn.metrics import confusion_matrix
import gzip
import os
import sys
import time
from Util import augment_data

import numpy
import tensorflow as tf

IMAGE_SIZE = 48
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 7
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 10000
SEED = 1550
EVAL_FREQUENCY = 64 
SAVE_FREQUENCY = 2048
CULTURE = 'American'
PATH = 'TFD_results/new/trial3/'
FOLD=3
DEVICE = '/gpu:0'
CKPT = '329728'
LR = .05
def error_rate(predictions, labels):
	"""Return the error rate based on dense predictions and sparse labels."""
	return 100.0 - (
		  100.0 *
		  numpy.sum(numpy.argmax(predictions, 1) == labels) /
		  predictions.shape[0])
def eval_in_batches(data, sess):
	eval_prediction = sess.graph.get_tensor_by_name('Softmax_1:0')
	eval_data = sess.graph.get_tensor_by_name("val_images:0")
	"""Get all predictions for a dataset by running it in small batches."""
	size = data.shape[0]
	if size < EVAL_BATCH_SIZE:
		raise ValueError("batch size for evals larger than dataset: %d" % size)
	predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
	for begin in xrange(0, size, EVAL_BATCH_SIZE):
		end = begin + EVAL_BATCH_SIZE
		if end <= size:
			predictions[begin:end, :] = sess.run(
				eval_prediction,
				feed_dict={eval_data: data[begin:end, ...]})
		else:
			batch_predictions = sess.run(
				eval_prediction,
				feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
			predictions[begin:, :] = batch_predictions[begin - size:, :]
	return predictions


def train():
	with tf.device(DEVICE):
		#TFD DATA
		train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_tfd_data(split=FOLD)

		#NATURAL SINGLE_CULTURAL DATA
		#train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_natural_single(CULTURE, split=FOLD)

		#NATURAL BI_CULTURAL DATA
		
		train_size = train_labels.shape[0]
		train_err_list = []
		train_loss_list = []
		val_err_list = []


		train_data_node = tf.placeholder(tf.float32, shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='train_images')
		train_labels_node = tf.placeholder(tf.int64, shape = (BATCH_SIZE,), name='train_labels')

		eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),name = 'val_images')


		conv1_weights = tf.Variable(tf.truncated_normal([5,5,NUM_CHANNELS,64], stddev = 0.01, seed = SEED, dtype= tf.float32, name='conv1_W'))
		conv1_biases = tf.Variable(tf.zeros([64], dtype = tf.float32), name='conv1_b')

		conv2_weights = tf.Variable(tf.truncated_normal([5,5,64,128], stddev = .01, seed = SEED, dtype = tf.float32), name='conv2_W')
		conv2_biases = tf.Variable(tf.zeros([128], dtype = tf.float32), name = 'conv2_b')

		conv3_weights = tf.Variable(tf.truncated_normal([5,5,128,256], stddev = .01, seed = SEED, dtype = tf.float32), name = 'conv3_W')
		conv3_biases = tf.Variable(tf.zeros([256], dtype = tf.float32), name = 'conv3_b')


		fc1_weights = tf.Variable(tf.truncated_normal([int(IMAGE_SIZE/8 * IMAGE_SIZE/8 * 256), 300], stddev=0.01, seed=SEED, dtype=tf.float32), name='fc1_W')
		fc1_biases = tf.Variable(tf.constant(0.1, shape=[300], dtype=tf.float32), name = 'fc1_b')

		fc2_weights = tf.Variable(tf.truncated_normal([300, NUM_LABELS], stddev=0.01, seed=SEED, dtype=tf.float32), name='fc2_W')
		fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32), name = 'fc2_b')

		def model(data, train=False):
			conv = tf.nn.conv2d(data,
					conv1_weights,
					strides=[1, 1, 1, 1],
					padding='SAME')
			# Bias and rectified linear non-linearity.
			relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
			# Max pooling. The kernel size spec {ksize} also follows the layout of
			# the data. Here we have a pooling window of 2, and a stride of 2.
			pool = tf.nn.max_pool(relu,
					ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1],
					padding='SAME')
			conv = tf.nn.conv2d(pool,
					conv2_weights,
					strides=[1, 1, 1, 1],
					padding='SAME')
			relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
			pool = tf.nn.max_pool(relu,
					ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1],
					padding='SAME')
			conv = tf.nn.conv2d(pool,
					conv3_weights,
					strides=[1, 1, 1, 1],
					padding='SAME')
			relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
			pool = tf.nn.max_pool(relu,
					ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1],
					padding='SAME')
			# Reshape the feature map cuboid into a 2D matrix to feed it to the
			# fully connected layers.
			pool_shape = pool.get_shape().as_list()
			reshape = tf.reshape(
					pool,
					[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
			# Fully connected layer. Note that the '+' operation automatically
			# broadcasts the biases.
			hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
			# Add a 50% dropout during training only. Dropout also scales
			# activations such that no rescaling is needed at evaluation time.
			if train:
				hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
			return tf.matmul(hidden, fc2_weights) + fc2_biases

		# Training computation: logits + cross-entropy loss.
		logits = model(train_data_node, True)
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits, train_labels_node))

		# L2 regularization for the fully connected parameters.
		regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
				tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
		# Add the regularization term to the loss.
		loss += 5e-4 * regularizers

		# Optimizer: set up a variable that's incremented once per batch and
		# controls the learning rate decay.
		batch = tf.Variable(0, dtype=tf.float32, name='batch')
		# Decay once per epoch, using an exponential schedule starting at 0.01.
		learning_rate = tf.train.exponential_decay(
				LR,                # Base learning rate.
				batch * BATCH_SIZE,  # Current index into the dataset.
				train_size,          # Decay step.
				0.999,                # Decay rate.
				staircase=True)
		# Use simple momentum for the optimization.
		optimizer = tf.train.MomentumOptimizer(learning_rate,
				0.9).minimize(loss,
						global_step=batch)

				# Predictions for the current training minibatch.
		train_prediction = tf.nn.softmax(logits)

		# Predictions for the test and validation, which we'll compute less often.
		eval_prediction = tf.nn.softmax(model(eval_data))

	  # Small utility function to evaluate a dataset by feeding batches of data to
	  # {eval_data} and pulling the results from {eval_predictions}.
	  # Saves memory and enables this to run on smaller GPUs.
		# Create a local session to run the training.
	saver = tf.train.Saver()
	start_time = time.time()
	with tf.Session() as sess:
		# Run all the initializers to prepare the trainable parameters.
		tf.initialize_all_variables().run()
		print('Initialized!')
		# Loop through training steps.
		for step in xrange(int(NUM_EPOCHS * train_size / BATCH_SIZE)):
			# Compute the offset of the current minibatch in the data.
			# Note that we could use better randomization across epochs.
			offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
			batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
			batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
			# This dictionary maps the batch data (as a numpy array) to the
			# node in the graph it should be fed to.
			feed_dict = {train_data_node: augment_data(batch_data, flip=True, rotate=True),
					train_labels_node: batch_labels}
			# Run the graph and fetch some of the nodes.
			_, l, lr, predictions = sess.run(
					[optimizer, loss, learning_rate, train_prediction],
					feed_dict=feed_dict)
			#print (numpy.argmax(predictions, 1))
			if (step + 1) % EVAL_FREQUENCY == 0:
				elapsed_time = time.time() - start_time
				start_time = time.time()
		  		print('Step %d (epoch %.2f), %.1f ms' %
				  		(step, float(step) * BATCH_SIZE / train_size,
					  	1000 * elapsed_time / EVAL_FREQUENCY))
				print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
			  	train_loss_list.append(l)

			  	train_err = error_rate(predictions, batch_labels)
			  	print('Minibatch error: %.1f%%' %train_err)
			  	train_err_list.append(train_err)

			  	val_err = error_rate(eval_in_batches(validation_data, sess), validation_labels)
		  		print('Validation error: %.1f%%' %val_err)
		  		val_err_list.append(val_err)

		  		sys.stdout.flush()
			if step % SAVE_FREQUENCY == 0:
				numpy.save(os.path.join(os.getcwd(), PATH + 'loss.npy'), train_loss_list) 
				numpy.save(os.path.join(os.getcwd(), PATH + 'trn_err.npy'), train_err_list) 
				numpy.save(os.path.join(os.getcwd(), PATH + 'val_err.npy'), val_err_list) 
				save_path = saver.save(sess,  PATH + 'tmp/model'+str(step)+'.ckpt')
				print("Model saved in file: %s" % save_path)
		# Finally print the result!
		tst_predictions = eval_in_batches(test_data, sess)
		test_error = error_rate(tst_predictions, test_labels)
		preds = (numpy.argmax(tst_predictions, 1))
		print('Test error: %.1f%%' % test_error)
		print(confusion_matrix(test_labels, preds))
		print(test_labels)
		print(preds)


def load_and_train():
	#TFD DATA
	train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_tfd_data(split=FOLD)

	#train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_natural_single(CULTURE, split=FOLD)
	train_size = train_labels.shape[0]
	train_loss_list = numpy.load(os.path.join(os.getcwd(), PATH + 'loss.npy')).tolist()
	train_err_list = numpy.load(os.path.join(os.getcwd(), PATH + 'trn_err.npy')).tolist()
	val_err_list = numpy.load(os.path.join(os.getcwd(), PATH + 'val_err.npy')).tolist()
	start_time = time.time()
	with tf.Session() as sess:
		# Restore variables from disk.
		new_saver = tf.train.import_meta_graph(PATH + 'tmp/model' + CKPT + '.ckpt.meta')
		new_saver.restore(sess, PATH + 'tmp/model' + CKPT + '.ckpt')
		print("Model restored.")
		
		predictions = numpy.ndarray(shape=(EVAL_BATCH_SIZE, NUM_LABELS), dtype=numpy.float32)
		train_data_node = sess.graph.get_tensor_by_name("train_images:0")
		train_labels_node = sess.graph.get_tensor_by_name("train_labels:0")
		train_prediction = sess.graph.get_tensor_by_name('Softmax:0')
		#loss = sess.graph.get_tensor_by_name('SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0')
		loss = sess.graph.get_tensor_by_name('add_5:0')#('Mean:0')
		learning_rate = sess.graph.get_tensor_by_name('ExponentialDecay:0')
		print(learning_rate.__class__)
		batch = sess.graph.get_tensor_by_name('batch:0')
		optimizer = tf.get_collection('train_op')[0]


		for step in xrange(int(NUM_EPOCHS * train_size / BATCH_SIZE)):
			# Compute the offset of the current minibatch in the data.
			# Note that we could use better randomization across epochs.
			offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
			batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
			batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
			# This dictionary maps the batch data (as a numpy array) to the
			# node in the graph it should be fed to.
			feed_dict = {train_data_node: augment_data(batch_data, flip=True, rotate=True),
					train_labels_node: batch_labels}
			# Run the graph and fetch some of the nodes.
			_, l, lr, predictions = sess.run(
					[optimizer, loss, learning_rate, train_prediction],
					feed_dict=feed_dict)
			#print (numpy.argmax(predictions, 1))
			if (step + 1) % EVAL_FREQUENCY == 0:
				elapsed_time = time.time() - start_time
				start_time = time.time()
		  		print('Step %d (epoch %.2f), %.1f ms' %
				  		(step, float(step) * BATCH_SIZE / train_size,
					  	1000 * elapsed_time / EVAL_FREQUENCY))
				print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
			  	train_loss_list.append(l)

			  	train_err = error_rate(predictions, batch_labels)
			  	print('Minibatch error: %.1f%%' %train_err)
			  	train_err_list.append(train_err)

			  	val_err = error_rate(eval_in_batches(validation_data, sess), validation_labels)
		  		print('Validation error: %.1f%%' %val_err)
		  		val_err_list.append(val_err)

		  		sys.stdout.flush()
			if step % SAVE_FREQUENCY == 0:
				numpy.save(os.path.join(os.getcwd(), PATH + 'loss.npy'), train_loss_list) 
				numpy.save(os.path.join(os.getcwd(), PATH + 'trn_err.npy'), train_err_list) 
				numpy.save(os.path.join(os.getcwd(), PATH + 'val_err.npy'), val_err_list) 
				save_path = new_saver.save(sess,  PATH + 'tmp/model'+str(step)+'.ckpt')
				print("Model saved in file: %s" % save_path)
		# Finally print the result!
		tst_predictions = eval_in_batches(test_data, sess)
		test_error = error_rate(tst_predictions, test_labels)
		preds = (numpy.argmax(tst_predictions, 1))
		print(confusion_matrix(test_labels, preds))

		print('Test error: %.1f%%' % test_error)


if __name__ == '__main__':
	train()
