import matplotlib.pyplot as plt
import numpy as np

def plot_feats(k, l):
	i = np.load('initial_img.npy')
	g = np.load('grad_img.npy')
	f = np.load('final_img.npy')
	inds = np.load('class_imgs.npy')
	inds = inds.reshape((inds.size,))
	plt.subplot(2,3,1)
	plt.imshow(i[inds[k],:,:,0], cmap='gray')
	plt.subplot(2,3,2)
	plt.imshow(g[inds[k],:,:,0], cmap='gray')
	plt.subplot(2,3,3)
	plt.imshow(f[inds[k],:,:,0], cmap='gray')
	plt.subplot(2,3,4)
	plt.imshow(i[inds[l],:,:,0], cmap='gray')
	plt.subplot(2,3,5)
	plt.imshow(g[inds[l],:,:,0], cmap='gray')
	plt.subplot(2,3,6)
	plt.imshow(f[inds[l],:,:,0], cmap='gray')
	return plt

if __name__=='__main__':
	plt = plot_feats(4, 5)
	plt.show()
