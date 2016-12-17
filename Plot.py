import matplotlib.pyplot as plt
import numpy as np
import os

def plotresults(path = '', msr = 'ci', colori = 0):
	markers = ['o', '*', '^', 'v', 'x', 's', '<', '>','d', 'p', 'h', '+']
	colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', 'w', '#56B4E9', '#A60628', '#8C0900', '#7A68A6']
	loss = np.load(os.path.join(os.getcwd(), path+'1/loss.npy'))
	print loss.shape
	arrs = np.empty([4, len(loss)])
	for i in range(4):
		curpath = os.path.join(os.getcwd(), path+str(i+1)+'/'+msr+'.npy')
		print curpath
		arr = np.load(curpath)
		print arr.shape
		arrs[i] = arr

	mean = np.mean(arrs, 0)
	std = np.std(arrs, 0)

	plt.plot(range(len(mean)), mean, color=colors[colori], marker=markers[colori], lw=2, ms=5, mfc = colors[colori], markevery = 5)
	plt.fill_between(range(len(mean)), mean-std, mean+std,color = colors[colori], alpha = .3)
	return plt

if __name__ == '__main__':
	msr = 'loss'
	path='American'
	plt = plotresults(path=path, msr=msr, colori=2)
#	plt = plotresults(msr=msr, colori=0)

#	plt.plot(range(2816), np.ones([2816]) * 19.8, color='c', marker='*', lw=2, ms=5, mfc = 'c', markevery = 15)
#	plt.plot(range(2816), np.ones([2816]) * 20.6, color='b', marker='+', lw=2, ms=5, mfc = 'b', markevery = 15)
#	plt.plot(range(2816), np.ones([2816]) * 10.1, color='b', marker='p', lw=2, ms=5, mfc = 'b', markevery = 15)
    
	
	plt.title('Misclassification Rate on Test Set- Korean faces')
	plt.ylabel('Misclassification rate on test set')
	plt.xlabel('training steps')
	#plt.legend(['our training', 'our validation', 'our testing', 'SOTA test', 'SOTA+augmentation test'])
	plt.xlim([0,160])
	plt.show()
