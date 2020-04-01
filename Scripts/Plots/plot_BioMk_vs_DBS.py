"""
For particular state (G12,G13), PLV vs Amplitude DBS
"""

import numpy as np
import matplotlib.pyplot as plt

I_PATH = '/mnt/BTE2b/DBS/2020/v2/Results/'

# Five states: Steady state - HFOscillation - SHopf - LFOscillation - PEI  ------------
states = [[0.4, 0.4],
		  [0.4, 0.8],
		  [0.4, 1.6],
		  [1.6, 0.4],
		  [1.6, 1.4]]

color_states = ['blue','orange','green','red','purple']
name_states = ['Steady State', 'HF Oscillation', 'PAC SH', 'LF Oscillation', 'PAC PEI']

# Load data ----------------------------------------------------------------------------
g1, g2, lfpower, plv = np.loadtxt(I_PATH + 'BioMk&Rew_FMap.dat',usecols=(0,1,3,4),unpack=True) 
SH_lfp, SH_plv = np.loadtxt(I_PATH + 'BioMk&Rew_SH.dat',usecols=(3,4),unpack=True)

# Amplitudes: For SH state, plot fives signals with differents DBS amplitude. ----------
amplitude = [0, 0.5, 2, 6]
position = [[0.22,0.22,0.17,0.15],
			[0.35,0.7,0.17,0.15],
			[0.41,0.4,0.17,0.15],
			[0.6,0.22,0.17,0.15]]

path_files = I_PATH + 'Simulations_SH/'

# Reward configuration ----------------------------------------------------------------
# ======================================================================================

# Figure: States vs DBS ----------------------------------------------------------------
fig1, ax1 = plt.subplots()

ax1.set_xlim([0,10])
ax1.set_xlabel('DBS amplitude')

ax12 = ax1.twinx()

for ss in range(len(states)):
 	z1 = plv[np.logical_and(g1 == states[ss][0], g2 == states[ss][1])]
 	z2 = lfpower[np.logical_and(g1 == states[ss][0], g2 == states[ss][1])]

 	ax1.plot(np.linspace(0,10,21),z1,color=color_states[ss],label=name_states[ss]) 
 	ax12.plot(np.linspace(0,10,21),z2,color=color_states[ss],linestyle='dashed')

ax1.set_ylim([0,1])
ax1.set_ylabel('|PLV|')

ax1.legend()

figname = I_PATH + 'Plots_States/BiomarkvsDBS_Node3_States.eps'
fig1.savefig(figname,format='eps')

# ======================================================================================

# Figure 2: SH state vs DBS. Insets: Signals ----------------------------------------------
fig2, ax2 = plt.subplots()
ax2.set_xlim([0,10])
ax2.set_xlabel('DBS amplitude')

ax22 = ax2.twinx()

ax2.set_ylim([0,1])
ax2.set_ylabel('|PLV|')
ax22.set_ylim([0,0.05])
ax22.set_ylabel('Power')

lns1 = ax2.plot(np.linspace(0,10,101),SH_plv,color='black',label='PLV')
lns2 = ax22.plot(np.linspace(0,10,101),SH_lfp,color='black',linestyle='dashed',label='LFPower')

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax2.legend(lns,labs,loc=0)


for aa in range(len(amplitude)):
	inset = fig2.add_axes(position[aa])
	
	signal = np.loadtxt(path_files + 'Inputs_0.5_0.4_1.6_' + str(amplitude[aa]) + '.dat')
	inset.plot(signal)

	inset.set_xlim([4000,6000])
	inset.set_xticklabels([])
	inset.set_yticklabels([])

figname = figname = I_PATH + 'Plots_States/BiomarkvsDBS_Node3_SH.eps' 
fig2.savefig(figname,format='eps')

# ======================================================================================
plt.show()

