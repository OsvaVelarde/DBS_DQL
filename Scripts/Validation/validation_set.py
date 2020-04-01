# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:22:15 2019

@author: Osvaldo M Velarde

@title: Generation of validation set.
"""


import numpy as np
from scipy import signal

import bgtc_network as bgnet
import sklearn.preprocessing

FILES_PATH = '/mnt/BTE2b/DBS/Enero-Julio-2019/DQLearning/Validation/ValidationSet/Files/'
OUTPUT_PATH = '/mnt/BTE2b/DBS/Enero-Julio-2019/DQLearning/Validation/ValidationSet/'

if __name__ == "__main__":
    
    # Configuration of BGTC network and Episodes.
    initial_conf_network = np.loadtxt(FILES_PATH + 'Conexiones.dat')
    g1 = [1,2.5,2.5,3.5,0.5,0.5,3.5,1]
    g2 = [-3,-1.5,-2.5,-3.5,-0.5,-3.5,-0.5,-1.5]
    num_episodes=len(g1)

    # Parameters for simulations of BGTC network.
    dt = 0.5                # [ms] - Time resolution
    fs = 1000/dt            # [Hz] - Sampling frequency
    tf = 5000               # [ms] - Generate the simulation in time (t0,t0 + tf)
    dt_transitory = 2500    # [ms] - Delete the transitory interval (t0,t0 + dt_transitory)
    dim_state = int(tf/dt) - int(dt_transitory/dt) # dimension of state space (length of simulation).

    for ep in np.arange(num_episodes):

        # Temporal limit for simulations of BGTC dynamics.
        t0 = 0
        t1 = tf    

        # Configuration of BGTC in the episode    
        initial_conf_network[0,0]=g1[ep]
        initial_conf_network[1,0]=g2[ep]
        np.savetxt(FILES_PATH + 'Conexiones.dat',initial_conf_network,fmt='%.2f')

        # Initial state: BGTC dynamic without stimulation
        network, populations, num_connections = bgnet.initialization_network(dt=dt,FILES_PATH=FILES_PATH)
        dbs_config = np.array([2, 0, 1, 0.5])                               # Amplitude DBS = 0.
        state = bgnet.solving_system(network,populations,dbs_config,t0,t1,dt_transitory,dt)[:,0]
        state = state + np.random.normal(0,0.001,dim_state)
        state = sklearn.preprocessing.scale(state)

        # Output files
        output_filename = OUTPUT_PATH + 'Signal_' + str(g1[ep]) + '_' + str(g2[ep]) + '.dat' 
        np.savetxt(output_filename,state)