# -*- coding: utf-8 -*-
"""
Created on October 2019

Description:
In this script the Phase Locking Value (PLV), Phase Clustering (PC),
Time Locked Index (TLI) and Harmonic Index (HI) are computed for:  
- Model BGTC Network. Ref: Velarde 2019

Steps:
1) From the whole TIME SERIES, we select an EPOCH to be processed.
2) The EPOCH is Z-score normalized in order to have zero mean and unit variance.
3) The whole EPOCH is Band-Pass Filtered.
4) The BPF signals are then subdivided in sliding SEGMENTS.
5) Each SEGMENT is Z-score normalized in order to have zero mean and unit variance.
6) The feature (phase/amplitude/frequency) is computed for each of the sliding SEGMENTS via the Hilbert transform. 
7) The PLV is computed for each of the sliding SEGMENTS.

@authors: Osvaldo M Velarde - Damián Dellavale - Javier Velez.
@affiliation: Dpto Física Médica - Centro Atómico Bariloche
@title: Quantification and characterization of Cross Frequency Coupling.

"""

import sys
import os
import numpy as np
#import matplotlib.pyplot as plt

import itertools
from sklearn.preprocessing import scale

sys.path.append('/mnt/BTE2b/DBS/2020/v2/Scripts/Modules/')
import bgtc_network_v2 as bgnet
import rewards as rew
from downsampling import downsampling

__author__ = "Velarde, O M"
__email__ = "osva.m.velarde@gmail.com"


# LOAD or GENERATE =========================================================
# Set the configuration for the data sets - MODEL BGTC Network. ------------

# # Option 1 : Read files --------------------------------------------------
#I_PATH = '/mnt/BTE2b/DBS/2020/v2/Results/Simulations_FullMaps/'
#prefName = 'Inputs_0.5_'
#fs = 2000
#dt = 1000/fs
# --------------------------------------------------------------------------

# Option 2 : Generate signals ----------------------------------------------
I_PATH = '/mnt/BTE2b/DBS/2020/v2/TrainingFiles/'
prefName = 'Inputs_0.5_'
Default_parameters = np.loadtxt(I_PATH + 'Conexiones.dat') # BGTC parameters.
dt = 0.5         # ms
tf=10000    # ms 
dt_transitory = 7000    # ms
fs = 1000/dt     # Hz  
network = bgnet.network(dt=dt,FILES_PATH=I_PATH)

# FULL MAP or SPECIFIC STATE ===============================================
# For Full Maps ------------------------------------------------------------
#Ng1 = 11
#Ng2 = 11
#Namp = 21

#synapt_G1 = np.linspace(0,2,num = Ng1)
#synapt_G2 = np.linspace(0,2,num = Ng2)    
#ampDBS = np.linspace(0,10, num = Namp)  
# --------------------------------------------------------------------------

# For Particular state------------------------------------------------------
Ng1 = 1
Ng2 = 1
Namp = 101

synapt_G1 = np.linspace(0.4,2,num = Ng1)
synapt_G2 = np.linspace(1.6,2,num = Ng2)
ampDBS = np.linspace(0,10, num = Namp)  

# ==========================================================================
# Set configuration for output path/file. ----------------------------------               
O_PATH = '/mnt/BTE2b/DBS/2020/v2/Results/'

if not os.path.exists(O_PATH+'Simulations/'):
    os.makedirs(O_PATH+'Simulations/')

# Configuration Output Path -----------------------------------------------
output_filename = O_PATH + 'BioMk&Rew.dat'
reward_file = open(output_filename,'w')

# ==========================================================================
# Preprocessing: Downsampling parameters. ---------------------------------- 
F3dB = 900                                     # Cutoff frequency for the anti-aliasing LPF [Hz]
LPF_WINDOW_PARAM = {'name':'tukeywide','r':0.1} # Tukey window with f2 configured as Cutoff frequency at 0dB.
LPFcfg = {'f2':F3dB, 'zeropadding':0,
          'freqWindowParam':LPF_WINDOW_PARAM, 'timeWindowParam':{'name':'box'} ,
          'function':'function_FDF'}            # Configure the anti-aliasing LPF.
decFactor = int(fs/(2*F3dB))                    # Downsampling - Decimation factor
fs = fs / decFactor                             # [samples/s] Nominal sampling rate.

# ==========================================================================
# Biomarkers: Frequency bands. ---------------------------------------------
psd_bands = [20, 200]          # [Hz]. Frequency range for Power calculation. (verificar que no cambia usar np.array)
plv_bands = [[1, 19], [20, 200]]           # [Hz]. Frequency bands of interest for  PLV computation.

# ==========================================================================
# Reward function. ---------------------------------------------------------
method = 'LfPlvAmpExp'
constants = [1, -50, -3, -0.25]
reward_cfg = {'fs':fs, 'psd_bands':psd_bands, 'plv_bands':plv_bands,
              'constants':constants, 'method':method}

# ==========================================================================
i=0
# Calculation: Loop in signals
for gg1, gg2, aa in itertools.product(synapt_G1,synapt_G2,ampDBS):

    print(i)
    
    # Load/Generate data. -------------------------------------------------- 
    
    # Option 1 -------------------------------------------------------------
    #filename = prefName + str('%g' % gg1) + '_' + str('%g' % gg2) + '_' + str('%g' % aa) + '.dat'
    #state = np.loadtxt(I_PATH + filename, usecols=(2))

    # Option 2 -------------------------------------------------------------
    network.Connections[1].G = gg1  
    network.Connections[2].G = gg2   
    dbs_config = [2, aa, 0.13, 0.5]                               # Amplitude DBS = 0 in N2.
    state = network.solving_system(dbs_config,tf,dt_transitory)[:,5] # Recording from N3

    filename = prefName + str('%g' % gg1) + '_' + str('%g' % gg2) + '_'+ str('%g' % aa)+'.dat'
    np.savetxt(O_PATH + 'Simulations/' + filename,state,fmt='%.6f')
    
    # Pre-processing -------------------------------------------------------
    state = scale(state)
    LPFcfg['freqWindowParam']['name']='tukeywide'
    LPFcfg['freqWindowParam']['r'] = 0.1
    state = downsampling(state,1000/dt,decFactor,LPFcfg)

    state= state + np.random.normal(scale = 0.001,size=np.shape(state))
    
    # Compute Biomarkers & Rewards   
    lfpower, plv ,reward = rew.reward_function(state, aa, state, reward_cfg)

    # Option 2: Reset 
    network.reset() # Time = 0, Initial Conditions = 0

    print(gg1, gg2, aa, lfpower, plv[0][0], reward[0][0], file = reward_file)
    i=i+1
#Fijarse que los G's lo guarde con precision .2f
reward_file.close()