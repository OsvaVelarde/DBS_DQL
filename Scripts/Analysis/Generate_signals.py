# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 08:40 2019

@author: Osvaldo M Velarde

@title: PLV vs DBS.
"""

import numpy as np
from scipy import signal
import Biomarkers
import BGTCNetwork as BGNet
import sklearn.preprocessing
import itertools
import matplotlib.pyplot as plt

FILES_PATH = '/mnt/BTE2b/DBS/Enero-Julio-2019/DQLearning/Files/'

if __name__ == "__main__":
    
    ##Environment parameters.
    Default_parameters = np.loadtxt(FILES_PATH + 'Conexiones.dat')

    nG1 = 10
    nG2 = 10
    nAmp = 20

    G1 = np.linspace(0,3,num = nG1)
    G2 = np.linspace(-3,0,num = nG2)       
    AmpDBS = np.linspace(0,10, num = nAmp)  
    DATA = np.zeros((nG1*nG2*nAmp,5))

    dt=0.5
    t_initial=0
    t_final=6000    
    DTrans = 2500

    i=0
    
    for g_1, g_2, amp in itertools.product(G1,G2,AmpDBS):
    
        print(i)

        Default_parameters[0,0] = g_1
        Default_parameters[1,0] = g_2
        
        np.savetxt(FILES_PATH + 'Conexiones.dat',Default_parameters,fmt='%.2f')

        Network, Populations, N_con = BGNet.initialization_network(dt,t_initial)
        DBS_Parameters = np.array([2, amp, 0.13, 0.08])   
        state=BGNet.solving_system(Network,Populations,DBS_Parameters,t_initial,t_final,DTrans,dt)[:,0]


    np.savetxt('Reward_vs_G1_G2_Amp.dat',DATA,fmt='%f')