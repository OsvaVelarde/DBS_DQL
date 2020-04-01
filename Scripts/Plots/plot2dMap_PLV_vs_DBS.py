"""
Plot 2d-colour plot.
"""

import numpy as np
import matplotlib.pyplot as plt

I_PATH = '/mnt/BTE2b/DBS/2020/v2/Results/'
O_PATH = '/mnt/BTE2b/DBS/2020/v2/Results/Plots/'

# Load data ---------------------------------------------------------------------
gg1, gg2, amplitude, lfpw, plv, rew= np.loadtxt(I_PATH + 'BioMk&Rew_FMap.dat',unpack=True)

# Configuration X - Y Lims
lims = [min(gg1), max(gg1), min(gg2), max(gg2)]

for aa in np.linspace(0,10, num = 21):
	z1 = lfpw[amplitude==aa]
	z2 = plv[amplitude==aa]
	z3 = rew[amplitude==aa]
	
	Z1 = z1.reshape(11,11,order='F')
	Z2 = z2.reshape(11,11,order='F')
	Z3 = z3.reshape(11,11,order='F')

	# Plot LPFW ----------------------------------------------------------------
	plt.figure()
	plt.imshow(Z1, cmap = 'jet',interpolation = 'bilinear', vmin=0, vmax=np.max(lfpw), extent = lims ,origin='lower')
	plt.colorbar()
	
	figname = O_PATH + 'Lfpw_' + 'Amp_' + str(aa) + '.eps'
	plt.savefig(figname,format='eps')

	plt.close()

	# Plot PLV -----------------------------------------------------------------
	plt.figure()
	plt.imshow(Z2, cmap = 'jet',interpolation = 'bilinear',vmin = 0, vmax=1, extent = lims ,origin='lower')
	plt.colorbar()

	figname = O_PATH + 'PLV_' + 'Amp_' + str(aa) + '.eps'
	plt.savefig(figname,format='eps')

	plt.close()

	# Plot Reward --------------------------------------------------------------
	plt.figure()
	plt.imshow(Z3, cmap = 'jet',interpolation = 'bilinear',vmin = 0, vmax=1, extent = lims ,origin='lower')
	plt.colorbar()
	
	figname = O_PATH + 'Reward_' + 'Amp_' + str(aa) + '.eps'
	plt.savefig(figname,format='eps')

	plt.close()
	#plt.show()