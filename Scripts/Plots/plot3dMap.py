"""
Plot 3d-colour plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

I_PATH = '/mnt/BTE2b/DBS/2020/v2/Results/'
O_PATH = '/mnt/BTE2b/DBS/2020/v2/Results/Plots/'


# Load data ---------------------------------------------------------------------
gg1, gg2, amplitude, lfpw, plv, _= np.loadtxt(I_PATH + 'BioMk&Rew_FMap.dat',unpack=True)
_, _, _, lfpwSH, plvSH, _= np.loadtxt(I_PATH + 'BioMk&Rew_SH.dat',unpack=True)

# Deseamos construir una funcion de recompensa de forma que A, PLV, Beta contribuyan inicialmente igual.
# exp (a*A + b* Beta + c* PLV) donde los productos queremos sean de orden 1.
# A \in [0,10]
# beta \in [0,0.05]
# plv \in [0,1]

# Analizamos el efecto de a,b,c. Fijado "a", veremos el efecto de "b/a" y "c/a"

# # =================================================================================
# # Reward function ---------------------------------------------------------------
# Ncoeff_amp = 1
# Ncoeff_lfp = 100
# Ncoeff_plv = 100

# MAXcoeff_amp = 0.1
# MAXcoeff_lfp = 200 * MAXcoeff_amp
# MAXcoeff_plv = 10 * MAXcoeff_amp

# a = np.linspace(MAXcoeff_amp,MAXcoeff_amp,Ncoeff_amp)
# b = np.linspace(0,MAXcoeff_lfp,Ncoeff_lfp)
# c = np.linspace(0,MAXcoeff_plv,Ncoeff_plv)

# # -------------------------------------------------------------------------------
# MaxDeltaR = np.zeros((Ncoeff_lfp*Ncoeff_plv,3)) 

# j=0
# for coeff_a, coeff_beta, coeff_plv in itertools.product(a,b,c):
# 	maxsA_reward = np.zeros(121)

# 	reward = coeff_a * amplitude + coeff_beta * lfpw + coeff_plv * plv
# 	reward = np.exp(-reward)

# 	for i in range(121):
# 		reward_local = reward[21*i:21*(i+1)]
# 		maxsA_reward[i] = max(reward_local)

# 	MaxDeltaR[j,0] = coeff_beta
# 	MaxDeltaR[j,1] = coeff_plv
# 	MaxDeltaR[j,2] = max(maxsA_reward) - min(maxsA_reward)

# 	j=j+1

# # Plot DeltaReward-----------------------------------------------------------
# Z = MaxDeltaR[:,2].reshape(Ncoeff_lfp,Ncoeff_plv,order='F')

# fig, ax= plt.subplots(1,1)
# im = ax.imshow(Z, cmap = 'jet',interpolation = 'bilinear', vmin=0, vmax=0.5,origin='lower')

# ax.set_xticks([0,25,50,75,100])
# ax.set_xticklabels(np.linspace(0,MAXcoeff_plv,5))

# ax.set_yticks([0,25,50,75,100])
# ax.set_yticklabels(np.linspace(0,MAXcoeff_lfp,5))

# fig.colorbar(im)

# ## GUARDAR IMAGEN

# # =================================================================================

# =================================================================================
# Five states: Steady state - HFOscillation - SHopf - LFOscillation - PEI  ------------
states = [[0.4, 0.4],
		  [0.4, 0.8],
		  [1.6, 0.4],
		  [1.6, 1.4]]

name_states = ['Steady State', 'HF Oscillation', 'LF Oscillation', 'PAC PEI']

amplDBS = np.linspace(0,10,101)
z0 = np.linspace(0,10,21) # DBS amplitude

# Reward function - Parameters ---------------------------------------------------------
coeff_amp = -0.1
coeff_lfp = -50
coeff_plv = -2.5

# PLOT 1: Rewards vs DBS in States: LF - HF - PEI - Steady state. ----------------------
# Plot options -----------
fig, axs = plt.subplots(2,2)

for ax in axs.flat:
	ax.set(xlabel='DBS amplitude', ylabel='Reward')

for ax in axs.flat:
	ax.label_outer()


for ss in range(4):

	# Plot options. ------
	m = int(ss/2)
	n = ss%2
	axs[m,n].text(5,0.9,name_states[ss])
	axs[m,n].set_xlim([0,10])
	axs[m,n].set_ylim([0,1])

	# Data. --------------
	z1 = plv[np.logical_and(gg1 == states[ss][0], gg2 == states[ss][1])]
	z2 = lfpw[np.logical_and(gg1 == states[ss][0], gg2 == states[ss][1])]

	# Plot. --------------
	i = 0
	
	reward = coeff_amp * z0 + coeff_plv * z1 + coeff_lfp * z2
	reward = np.exp(reward)

	axs[m,n].plot(z0,reward)
	i=i+1

# Save fig
#figname = '/mnt/BTE2b/DBS/Agosto-Diciembre-2019/DQLearning/RewardFunctionAnalysis/Rewards_States_CoeffPLV=' + str(MAXcoeff_plv) + '_CoeffLFP=' + str(MAXcoeff_lfp) + '.eps'
#fig.savefig(figname,format='eps')

# --------------------------------------------------------------------------------------
# PLOT 2: Reward function in SH state --------------------------------------------------

# Plot options. -------
fig2, ax2 = plt.subplots() 
ax2.set_xlim([0,10])
ax2.set_xlabel('DBS amplitude')
ax2.set_ylim([0,1])
ax2.set_ylabel('Reward')

# Plot ----------------
i=0

# Reward function
reward = coeff_amp * amplDBS + coeff_plv * plvSH + coeff_lfp * lfpwSH
reward = np.exp(reward)
	
ax2.plot(amplDBS,reward) # Plot Reward Function
i=i+1

ax2.legend()

# Save fig
#figname = '/mnt/BTE2b/DBS/Agosto-Diciembre-2019/DQLearning/RewardFunctionAnalysis/Rewards_SH_CoeffPLV=' + str(MAXcoeff_plv) + '_CoeffLFP=' + str(MAXcoeff_lfp) + '.eps'
#fig2.savefig(figname,format='eps')
plt.show()



