import numpy
import scipy.io as sio
import os
PIXEL_DEPTH = 255
def read_tfd_data(split=0):
	path_in =  os.path.join(os.getcwd(), 'TFD_HERE/npy_files/TFD_48/')
	split = 'split_' + str(4)

	path_inds =  os.path.join(path_in, split)
	trn_ind = numpy.load(os.path.join(path_inds, 'trn_ind.npy'))
	tst_ind = numpy.load(os.path.join(path_inds, 'tst_ind.npy'))
	val_ind = numpy.load(os.path.join(path_inds, 'val_ind.npy'))

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
	test_data = labeled_data[tst_ind, ...]
	test_labels = labels [tst_ind, ...]
	validation_data = labeled_data[val_ind, ...]
	validation_labels = labels[val_ind, ...]
	return train_data, train_labels, test_data, test_labels, validation_data, validation_labels

def read_natural_single(culture='Korean', split=0):
	numpy.random.seed(split)
	path_in = 'Natural_48x48/' + culture + '_Faces.mat'
	D = sio.loadmat(path_in)
	labeled_data = D['images'];
	labels = D['labels'];
	sample_size = labels.shape[0]
	perm = range(sample_size)
	numpy.random.shuffle(perm)
	labeled_data = labeled_data[perm,...]
	labeled_data = labeled_data[:, :, :, numpy.newaxis]
	labels = labels[perm].reshape((sample_size,))
	
	#Normalize 0-1
	labeled_data = (labeled_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
	
	#Split data into train, test, validate
	test_i = sample_size*1/10
	val_i = sample_size*2/10
	train_data = labeled_data[val_i:,...]
	train_labels = labels[val_i:] 
	test_data = labeled_data[:test_i,...]
	test_labels = labels[:test_i] 
	validation_data = labeled_data[test_i:val_i,...]
	validation_labels = labels[test_i:val_i]	
	return train_data, train_labels, test_data, test_labels, validation_data, validation_labels


def read_natural_bi(cultureA='Korean', cultureB='American'):
	return 

if __name__=="__main__":
	train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_natural_single('American')
	print train_data.shape, train_labels.shape
	print test_data.shape, test_labels.shape
	print validation_data.shape, validation_labels.shape
