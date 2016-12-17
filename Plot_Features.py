import matplotlib.pyplot as plt
import numpy as np

def plot_feats():
	m = 8
	n = 16
	OFFSET = 16
	count = 1
	Orig = np.load('initial_img.npy')
	print Orig.shape
	for i in range(m):
		g = np.load('visuals_conv2/1-3-'+str(i)+'grad_img.npy')
		for j in range(n/2):
			plt.subplot(m,n,count)
			plt.imshow(g[j+OFFSET,:,:,0], cmap='gray')
			count = count+1
			plt.subplot(m,n,count)
			count = count+1
			plt.imshow(Orig[j+OFFSET,:,:,0], cmap='gray')
	return plt
def plot_results(k, l, m, n, o, p):
	i = np.load('initial_img.npy')
	plt.subplot(2,3,1)
	plt.imshow(i[k,:,:,0], cmap='gray')
	plt.subplot(2,3,2)
	plt.imshow(i[l,:,:,0], cmap='gray')
	plt.subplot(2,3,3)
	plt.imshow(i[m,:,:,0], cmap='gray')
	plt.subplot(2,3,4)
	plt.imshow(i[n,:,:,0], cmap='gray')
	plt.subplot(2,3,5)
	plt.imshow(i[o,:,:,0], cmap='gray')
	plt.subplot(2,3,6)
	plt.imshow(i[p,:,:,0], cmap='gray')
	return plt

def plot_emotion_modified(k, l):
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
	#plt = plot_feats()
	#plot_results(0, 1, 2, 3, 4, 5)
	plot_emotion_modified(0,1)
	plt.show()
