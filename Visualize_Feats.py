import tensorflow as tf
import os
import numpy
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
	return tf.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

IMAGE_SIZE = 48
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 7
BATCH_SIZE = 10
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 1
SEED = 1550
EVAL_FREQUENCY = 16 
SAVE_FREQUENCY = 512
# 1 Disgust, 3 Happy, 4 sad, 6 neutral 
CHANNEL = 2

def visualize():
	path_in =  os.path.join(os.getcwd(), 'TFD_HERE/npy_files/TFD_48/')
	split = 'split_2'

	path_inds =  os.path.join(path_in, split)
	trn_ind = numpy.load(os.path.join(path_inds, 'trn_ind.npy'))
	labeled_data = numpy.load(os.path.join(path_in, 'X.npy'))
	labeled_data = (labeled_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
	labels =  numpy.load(os.path.join(path_in, 'y.npy')) - 1
	train_data = labeled_data[trn_ind, ...]
	train_labels = labels[trn_ind, ...] 
	train_size = train_labels.shape[0]
	perm = range(train_size)
	numpy.random.shuffle(perm)
	train_data = train_data[perm,...]
	train_labels = train_labels[perm,...]
	
	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph('tmp3/model30720.ckpt.meta')
		new_saver.restore(sess, "tmp3/model30720.ckpt")
		all_vars = tf.trainable_variables()
		ops = sess.graph.get_operations()
		"""
		for op in ops:
			print op.name
			print "values: ", op.values()
		for v in all_vars:
		    print(v.name)
		"""
		#new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		#layer = 'Conv2D_2'
		layer = 'MatMul_1'
		#layer = 'SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits'
		t_layer= sess.graph.get_tensor_by_name("%s:0"%layer)
		print t_layer
		input_data_node = sess.graph.get_tensor_by_name("train_images:0")
		#t_obj = t_layer[:,:,:,CHANNEL]
		t_obj = t_layer[:,CHANNEL]
		t_score = tf.reduce_mean(t_obj)
		t_grad = tf.gradients(t_score, input_data_node)[0]
		step = 0.01
		# Restore variables from disk.
		summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)
		print("Model restored.")
		g = tf.get_default_graph()
		with g.gradient_override_map({'Relu': 'GuidedRelu'}):
			img = train_data.copy()
			img = img[:64,:,:,:]
			labels = train_labels[:64]
			print labels
			for i in range(NUM_EPOCHS):
				print('iteration = ', i)
				g, score = sess.run([t_grad, t_obj], {input_data_node:img})
				# normalizing the gradient, so the same step size should work 
				g /= g.std()+1e-8         # for different layers and networks
				img += g*step
			train_sub = train_data[:64,:,:,:]
			inds = numpy.where(labels == CHANNEL)
			print inds
			numpy.save(os.path.join(os.getcwd(), 'final_img.npy'), img)
			numpy.save(os.path.join(os.getcwd(), 'grad_img.npy'), g)
			numpy.save(os.path.join(os.getcwd(), 'initial_img.npy'), train_sub)
			numpy.save(os.path.join(os.getcwd(), 'class_imgs.npy'), inds)
if __name__ == '__main__':
	visualize()

