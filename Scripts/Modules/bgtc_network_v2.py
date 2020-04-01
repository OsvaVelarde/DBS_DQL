#ª/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# BG-ThalamoCortical Network.
(See reference: Velarde et al 2017). 

Equations : ...
Estructura: ... 

Summary:
This module allows to obtain signals that are solutions of BGTCNetwork. 

"""
__author__ = "Velarde, O M"
__email__ = "osva.m.velarde@gmail.com"

import numpy as np

class node:

    """ 'node' represents to a population in BGTCNetwork.
    It is characterized by:
    - threshold (thr)
    - external input (ext_input)
    - noise level (noise)
    - total input (I)
    """

    def __init__(self, threshold, ext_input, noise_level, input_value):
        self.thr = threshold
        self.ext_i = ext_input
        self.noise = noise_level
        self.I = input_value

    def external_input(self):
        """ Update the total input I as external input ext_i."""

        self.I = self.ext_i

class connection:
    """ 'connection' represents the directed synaptic transmission between two nodes.
    It is characterized by:
    - synaptic efficacy (G)
    - time constant (tau)
    - synaptic delay (d)
    - synaptic current between 'node initial' -> 'node final' in [t-d,t] (m)
    - initial node (Ni)
    - final node (Nf)
     """

    def __init__(self, gain, delay, time_constant, m_value, initial_node,final_node):
        self.G = gain
        self.d = delay
        self.tau = time_constant
        self.m = m_value
        self.Ni = int(initial_node-1)
        self.Nf = int(final_node-1)

class network:

    """ Creation of BGTC Network using files data: 
        'Ganglios.dat', 'Interaction.dat', 'Conexiones.dat'.

        Input:
        - dt: numeric value temporal resolution.
        - t_0: numeric value. Default value = 0.
        - m_0: numeric array . Default value = None.

        Output:
        - Network
        - Population
        - N_con 
    """


    def __init__(self,dt,FILES_PATH=''):

        self.path = FILES_PATH
        self.nnodes = 0
        self.nconn = 0
        self.dt = dt
        self.time = 0
        self.Populations = self.add_nodes()
        self.Connections = self.add_connections()
        

    def add_nodes(self):
        BG_parameters = np.loadtxt(self.path + 'Ganglios.dat')
        self.nnodes = BG_parameters.shape[0]    # Number of nodes in Network.

        Populations = [node(BG_parameters[i][0],BG_parameters[i][1], 0, 0) for i in range(self.nnodes)]
        return Populations

    def add_connections(self):
        Nodes_in_connections  = np.loadtxt(self.path + 'Interaction.dat')
        Connection_parameters = np.loadtxt(self.path + 'Conexiones.dat')
        self.nconn = Connection_parameters.shape[0]    # Number of connections in network

        # Default inital conditions are : t_0=0 and m_0([-d,0])=0 (for all connections)
        len_delay=[int(x) for x in np.floor(Connection_parameters[:,1]/self.dt)]

        Connections = [connection(Connection_parameters[i][0],
                                        Connection_parameters[i][1], 
                                        Connection_parameters[i][2], 
                                        np.zeros(len_delay[i]), 
                                        Nodes_in_connections[i][0],
                                        Nodes_in_connections[i][1]) for i in range(self.nconn)]
        return Connections

    def update_inputs(self):
        """ Update Inputs values xxxxxxx
        """
        for CON in self.Connections:
            value = CON.m[0]
            final_node = CON.Nf
        
            self.Populations[final_node].I += CON.G * value
            CON.m=CON.m[1:]

    def solving_system(self,DBS_Parameters,t_final,DTrans):
        
        if t_final > self.time:
            N_step= int((t_final - self.time)/ self.dt)

        if t_final - self.time > DTrans:
            N_save = int(DTrans/self.dt)

        # DBS Parameters
        Node_DBS  = int(DBS_Parameters[0]-1)
        A_DBS     = DBS_Parameters[1]
        freq_DBS  = DBS_Parameters[2]
        width_DBS = DBS_Parameters[3]
    
        Signals = np.zeros((N_step-N_save,self.nconn))
       
        for i in range(N_step):

            for NODE in self.Populations:
                NODE.external_input()

            self.Populations[Node_DBS].I += DBS(self.time,A_DBS,freq_DBS,width_DBS)
            
            self.update_inputs()

            for j in range(self.nconn):
                CON = self.Connections[j]
                initial_node = CON.Ni
            
                act = self.Populations[initial_node].I - self.Populations[initial_node].thr 
                act = act if act > 0 else 0 
            
                m_value = CON.m[-1]
                dm = (act-m_value)*self.dt/CON.tau
                m_value += dm
            
                CON.m = np.concatenate((CON.m,[m_value]))

                if  N_save <= i:
                    #Signals[i-N_save,0]= time
                    Signals[i-N_save,j] = self.Populations[initial_node].I   
        
            self.time += self.dt
    
        return Signals

    def reset(self):
        self.time=0
        for CON in self.Connections:
            CON.m = 0 * CON.m 


def DBS(time,amplitude,frequency,width):
    """ Deep Brain Stimulation (DBS): 
    Square pulse train with monopolar amplitude.  

    Input:
        -time: numeric value.
        -amplitude: numeric value +
        -frequency: numeric value + 
        -width: numeric value (0,1)

    Output:
        -stimulus: num value
    """

    u = np.mod(time,1/frequency)
    return amplitude if u < width else 0