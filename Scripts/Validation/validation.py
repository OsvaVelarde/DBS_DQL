# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:22:15 2019

Application of Deep Q-learning algotithm (DQL) as scheme of Closed-Loop DBS.

About DQL:
Environment -> State (s) ->

Environment:
The model of BGTC network is composed by two nodes. 
The state of model is controlled by synaptic efficacies (g1->2,g2->1) between nodes.
The bifurcation parameter is G = g1*g2 (total efficacy). For |G| < 2, the network converges to fixed point.
In other case, the dynamics is oscillatory with frequency 13 Hz. 
It is possible to apply a stimulation to the network (Deep Brain Stimulation - DBS)

The configuration of system is discussed in "Velarde et al 2017".
The implementation of BGTC network is in the module BGTCNetwork.

State space (s):
We consider the signal I_{1} (input of node 1) in BGTC network as 'state of system' (s).

Action space (a):
We consider the amplitude of DBS as 'action on the system' (a). After the stimulation a, the state s changes to s'. 

Reward (R):
Our goal is to minimize the pathological oscillations ('beta power') with the least possible stimulation power ('gamma power').
We compute 'beta/gamma power' in the new state s'.

The proposed reward function is:
R(s,s',a) = A * (B-beta)^2 * (C-gamma)

Episode:
Para cada episodio, configuramos el sistema con valores de ganancias aleatorios (g1,g2).
En cada uno, el agente probará diferentes estimualciones (a) en la misma configuración.

@author: Osvaldo M Velarde

@title: Deep Q-learning Algorithm in Closed-Loop DBS
"""

import sys
import numpy as np
import os

import sklearn.preprocessing

from datetime import date
import json

sys.path.append('/mnt/BTE2b/DBS/2020/v2/Scripts/Modules/')
import bgtc_network_v2 as bgnet
import biomarkers as biomk
import rewards as rew
import configuration as cfg
from dqnagent import DQNAgent
from downsampling import downsampling

__author__ = "Velarde, O M"
__email__ = "osva.m.velarde@gmail.com"

FILES_PATH = '/mnt/BTE2b/DBS/2020/v2/TrainingFiles/'
CFG_PATH = '/mnt/BTE2b/DBS/2020/v2/Trainings/'
TRAINING_DATE =  '2020-03-13'
OUTPUT_PATH = '/mnt/BTE2b/DBS/2020/v2/Results/Validation/'

# ===============================================================================

# Configuration of BGTC network and Validation samples.
initial_conf_network = np.loadtxt(FILES_PATH + 'Conexiones.dat')
g12_lims = [[0.25,1.5]]
g13_lims = [[0.1,1.6]]
num_samples = 3000 
g12,g13,_ = cfg.config_episode(g12_lims,g13_lims,num_samples)

# Parameters for simulations of BGTC network.
dt = 0.5                # [ms] - Time resolution
fs = 1000/dt            # [Hz] - Sampling frequency
tf = 3500               # [ms] - Generate the simulation in time (t0,t0 + tf)
dt_transitory = 1000    # [ms] - Delete the transitory interval (t0,t0 + dt_transitory)

# Downsampling parameters. 
F3dB = 900                                     # Cutoff frequency for the anti-aliasing LPF [Hz]
LPF_WINDOW_PARAM = {'name':'tukeywide','r':0.1} # Tukey window with f2 configured as Cutoff frequency at 0dB.
LPFcfg = {'f2':F3dB, 'zeropadding':0, 
          'freqWindowParam':LPF_WINDOW_PARAM, 'timeWindowParam':{'name':'box'} ,
          'function':'function_FDF'}            # Configure the anti-aliasing LPF.
decFactor = int(fs/(2*F3dB))                    # Downsampling - Decimation factor
fs = fs / decFactor                             # [samples/s] Nominal sampling rate.

# ============================
dim_state = int(fs*tf/1000) - int(fs*dt_transitory/1000) # dimension of state space (length of simulation).

# Load Action Space
action_space = np.loadtxt(FILES_PATH + 'Action_space.dat')
num_action = action_space.shape[0]

# Load agent
step = 149500
cfg_filename = CFG_PATH + 'Cfg_' + TRAINING_DATE +'.json'
path_training = CFG_PATH + TRAINING_DATE +'/'

agent = cfg.load_agent(cfg_filename,path_training,dim_state, num_action, step)

# Reward function parameters
psd_bands = [20, 200]          # [Hz]. Frequency range for Power calculation. (verificar que no cambia usar np.array)
plv_bands = [[1, 19], [20, 200]]           # [Hz]. Frequency bands of interest for  PLV computation.

# Output files
output_filename = OUTPUT_PATH + TRAINING_DATE + '.dat' 
output_file = open(output_filename,'w')

# Initialization of BGTC network
network = bgnet.network(dt=dt,FILES_PATH=FILES_PATH)

for ep in np.arange(num_samples):

    # Configuration of BGTC in the episode
    network.Connections[1].G = g12[ep]  
    network.Connections[2].G = g13[ep]

    # Initial state: BGTC dynamic without stimulation
    dbs_config = [2, 0, 1, 0.5]                               # Amplitude DBS = 0 in N2.
    state = network.solving_system(dbs_config,tf,dt_transitory)[:,5] # Recording from N3
    state = sklearn.preprocessing.scale(state)

    LPFcfg['freqWindowParam']['name']='tukeywide'
    LPFcfg['freqWindowParam']['r'] = 0.1
    state = downsampling(state,1000/dt,decFactor,LPFcfg)

    state = state + np.random.normal(scale = 0.001,size=np.shape(state))
    state = sklearn.preprocessing.scale(state)

    # Beta Power / PAC in initial state ===============================

     # Lf power
    mean_power = biomk.PSD_features(state,fs,psd_bands)
    initial_Lf_power = mean_power[0]

    # PLV
    initial_PLV = biomk.PLV_features(state,fs,plv_bands)

    # Selection of action. ===========================================
    index_action = agent.select_action(state)
    action = action_space[index_action]
        
    # Execution of action and detection of new state. ================

    t1 = network.time + tf
    next_state = network.solving_system(action,t1,dt_transitory)[:,2]
    next_state = sklearn.preprocessing.scale(next_state)    
        
    LPFcfg['freqWindowParam']['name']='tukeywide'
    LPFcfg['freqWindowParam']['r'] = 0.1
    next_state = downsampling(next_state,1000/dt,decFactor,LPFcfg)

    next_state = next_state + np.random.normal(scale = 0.001,size=np.shape(next_state))
    next_state=sklearn.preprocessing.scale(next_state)

    # Beta Power / PAC in initial state ===============================

     # Lf power
    mean_power = biomk.PSD_features(next_state,fs,psd_bands)
    final_Lf_power = mean_power[0]

    # PLV
    final_PLV = biomk.PLV_features(next_state,fs,plv_bands)
    
    # ================================================================
    print('Episode = ', ep)
    print(g12[ep], g13[ep], action[1], initial_Lf_power, final_Lf_power, np.abs(initial_PLV[0][0]), np.abs(final_PLV[0][0]), file = output_file)

    network.reset() # Time = 0, Initial Conditions = 0

output_file.close()
