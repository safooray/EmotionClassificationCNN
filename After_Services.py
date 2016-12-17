import tensorflow as tf
from sklearn.metrics import confusion_matrix
import os
import numpy
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from DataIO import *
from Model import eval_in_batches
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
	return tf.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

IMAGE_SIZE = 96
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 5
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 10
SEED = 1550
EVAL_FREQUENCY = 16 
SAVE_FREQUENCY = 512
#0 Anger, 1 Disgust, 2 Fear, 3 Happy, 4 sad, 5 surprise, 6 neutral 
CHANNEL = 3
FEAT_I = 5
FEAT_J = 5
CULTURE = 'American'
FOLD = 0
#PATH = "American0_10K/"
PATH = "TFD_results/new/trial1/"
CKPT = '456704'

def modify_emotion():
	train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_tfd_data(split=FOLD)
#	train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_natural_single(culture=Culture, split=FOLD)

	with tf.Session() as sess:
		# Restore variables from disk.
		new_saver = tf.train.import_meta_graph(PATH + 'tmp/model' + CKPT + '.ckpt.meta')
		new_saver.restore(sess, PATH + 'tmp/model' + CKPT + '.ckpt')
		layer = 'Conv2D_2'
		layer = 'Softmax'
		#layer = 'MatMul_1'
		#layer = 'SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits'
		t_layer= sess.graph.get_tensor_by_name("%s:0"%layer)
		print t_layer
	
		input_data_node = sess.graph.get_tensor_by_name("train_images:0")
		#tmp = t_layer[:,FEAT_I,FEAT_J,CHANNEL]
		#print tmp
		#mask = numpy.zeros(t_layer.get_shape())
		#mask [:, FEAT_I, FEAT_J, CHANNEL] +=1
		#mask = tf.convert_to_tensor(mask, dtype=tf.float32)
		#t_obj = tf.mul(t_layer, mask)
		print t_layer
		t_obj = t_layer[:,CHANNEL]
		t_score = tf.reduce_mean(t_obj)
		t_grad = tf.gradients(t_score, input_data_node)[0]
		step = 0.01
		#summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)
		print("Model restored.")
		g = tf.get_default_graph()
		with g.gradient_override_map({'Relu': 'GuidedRelu'}):
			img = train_data[:EVAL_BATCH_SIZE,...].copy()
			labels = train_labels[:EVAL_BATCH_SIZE]
			print labels
			for i in range(NUM_EPOCHS):
				print('iteration = ', i)
				g, score = sess.run([t_grad, t_obj], {input_data_node:img})
				# normalizing the gradient, so the same step size should work 
				g /= g.std()+1e-8         # for different layers and networks
				img += g*step
			inds = numpy.where(labels == CHANNEL)
			print inds
			numpy.save(os.path.join(os.getcwd(), 'final_img.npy'), img)
			numpy.save(os.path.join(os.getcwd(), 'grad_img.npy'), g)
			numpy.save(os.path.join(os.getcwd(), 'initial_img.npy'), train_data[:EVAL_BATCH_SIZE,...])
			numpy.save(os.path.join(os.getcwd(), 'class_imgs.npy'), inds)

def visualize(feat_i, feat_j, channel, layer):
	train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_tfd_data(split=FOLD)
	#train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_natural_single(culture=CULTURE, split=FOLD)
	train_size = train_labels.shape[0]
	with tf.Session() as sess:
		# Restore variables from disk.
		new_saver = tf.train.import_meta_graph(PATH + 'tmp/model' + CKPT + '.ckpt.meta')
		new_saver.restore(sess, PATH + 'tmp/model' + CKPT + '.ckpt')
		"""
		all_vars = tf.trainable_variables()
		ops = sess.graph.get_operations()
		for op in ops:
			print op.name
			print "values: ", op.values()
		for v in all_vars:
		    print(v.name)
		"""
		t_layer= sess.graph.get_tensor_by_name("%s:0"%layer)
		print t_layer
	
		input_data_node = sess.graph.get_tensor_by_name("train_images:0")
		#tmp = t_layer[:,FEAT_I,FEAT_J,CHANNEL]
		#print tmp
		mask = numpy.zeros(t_layer.get_shape())
		mask [:, feat_i, feat_j, channel] +=1
		mask = tf.convert_to_tensor(mask, dtype=tf.float32)
		t_obj = tf.mul(t_layer, mask)
		print t_layer
		#t_obj = t_layer[:,CHANNEL]
		t_score = tf.reduce_mean(t_obj)
		t_grad = tf.gradients(t_score, input_data_node)[0]
		step = 0.01
		#summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)
		print("Model restored.")
		#find images maximally activating neurons
		#eval_in_batches(train_data, sess)
		#max_i = numpy.zeros(train_size, 1, 1, 256)
		#for i in train_size:
		#	for j in range(256):
		#		max_i[i,:,:,j] = numpy.unravel_index(argmax(max_i[i,...,j]), max_i[i,...,j].shape)
		g = tf.get_default_graph()
		with g.gradient_override_map({'Relu': 'GuidedRelu'}):
			img = train_data[:EVAL_BATCH_SIZE,:,:,:].copy()
			#labels = train_labels[:EVAL_BATCH_SIZE]
			#print labels
			for i in range(NUM_EPOCHS):
				print('iteration = ', i)
				g, score = sess.run([t_grad, t_obj], {input_data_node:img})
				# normalizing the gradient, so the same step size should work 
				g /= g.std()+1e-8         # for different layers and networks
				img += g*step
			return g, img
def save_all_vis():
	train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_tfd_data(split=FOLD)
	numpy.save(os.path.join(os.getcwd(), 'initial_img.npy'), train_data[:EVAL_BATCH_SIZE,:,:,:])
	#layer = 'Conv2D_2'
	layer = 'MaxPool_2'
	#layer = 'MatMul_1'
	#layer = 'SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits'
	for feat_i in range(6):
		for feat_j in range(6):
			for channel in range(8):
				grad_img, final_img = visualize(feat_i, feat_j, channel, layer)
				numpy.save(os.path.join(os.getcwd(), 'visuals_conv2/'+str(feat_i) +'-'+ str(feat_j)+'-'+str(channel) + 'final_img.npy'), final_img)
				numpy.save(os.path.join(os.getcwd(), 'visuals_conv2/'+str(feat_i) +'-'+ str(feat_j)+'-'+str(channel) + 'grad_img.npy'), grad_img)


def validate():
	train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_natural_single(CULTURE, split=4)
	TEST_SIZE = test_data.shape[0]
	with tf.Session() as sess:
		# Restore variables from disk.
		new_saver = tf.train.import_meta_graph(PATH + 'tmp/model' + CKPT + '.ckpt.meta')
		new_saver.restore(sess, PATH + 'tmp/model' + CKPT + '.ckpt')
		print("Model restored.")
		predictions = numpy.ndarray(shape=(TEST_SIZE, NUM_LABELS), dtype=numpy.float32)
		eval_prediction = sess.graph.get_tensor_by_name('Softmax_1:0')
		eval_data = sess.graph.get_tensor_by_name("val_images:0")
		predictions = eval_in_batches(test_data, sess)
		preds = (numpy.argmax(predictions, 1))
		print(confusion_matrix(test_labels, preds))
if __name__ == '__main__':
	modify_emotion()


