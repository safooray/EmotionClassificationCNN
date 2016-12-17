import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.transform as skimg
# Do data augmentation (crops, flips, rotations, scales, intensity)
def augment_data(X, crop_shape=None, flip=False, rotate=False, scale=False):

	batch_size, width, height, channels = X.shape
	out_batch = np.zeros(X.shape, dtype=X.dtype)

	for ind in range(batch_size):
		output = np.empty((width, height, channels), dtype=np.float32)
		transform = skimg.AffineTransform(scale=(1.0, 1.0))
		sample = X[ind, ...]
		if flip == True: # flip
			flip_rv = np.random.randint(0,2)
			if flip_rv == 1:
				sample = sample[:, ::-1, :]

		if rotate == True: #rotate
			angle = (np.random.rand()-.5)*10
			transform += skimg.AffineTransform(
					rotation=np.deg2rad(angle))
		if scale == True: #scale
			scale_p = np.random.rand() * 0.7 + 0.7
			transform += skimg.AffineTransform(
					scale=(scale_p, scale_p))

		for channel in range(channels):
			output[:, :, channel] = skimg._warps_cy._warp_fast(
					image=sample[:, :, channel],
					H=transform.params,
					output_shape=sample[:, :, channel].shape)
			out_batch[ind, ...] = output
	return out_batch

"""	def gray_augment_image(data):
		image = data.transpose(1, 2, 0)
		v_factor1 = numpy.random.uniform(0.25, 4)
		v_factor2 = numpy.random.uniform(0.7, 1.4)
		v_factor3 = numpy.random.uniform(-0.1, 0.1)

		# print '(v1, v2, v3) = (%f, %f, %f)' % (v_factor1, v_factor2, v_factor3)

		image = (image ** v_factor1) * v_factor2 + v_factor3

		# Rescale to [0, 1] range
		image_min = image.min()
		image -= image_min
		image_max = image.max()
		image /= image_max

		data_out = image.transpose(2, 0, 1)
		return data_out """
def test():
	path_in =  os.path.join(os.getcwd(), 'TFD_HERE/npy_files/TFD_48/')
	X = np.load(os.path.join(path_in, 'X.npy'))
	X = X[0:3,...]
	X_aug = augment_data(X, flip=False, scale=False,rotate=True)
	print X.shape, X_aug.shape
	plt.subplot(3,2,1)
	plt.imshow(X[0,:,:,0], cmap='gray')
	plt.subplot(3,2,2)
	plt.imshow(X_aug[0,:,:,0], cmap='gray')
	plt.subplot(3,2,3)
	plt.imshow(X[1,:,:,0], cmap='gray')
	plt.subplot(3,2,4)
	plt.imshow(X_aug[1,:,:,0], cmap='gray')
	plt.subplot(3,2,5)
	plt.imshow(X[2,:,:,0], cmap='gray')
	plt.subplot(3,2,6)
	plt.imshow(X_aug[2,:,:,0], cmap='gray')
	plt.show()

if __name__=='__main__':
	test()



