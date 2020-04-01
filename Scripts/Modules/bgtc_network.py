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
from scipy import signal

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
        self.Ni=int(initial_node-1)
        self.Nf=int(final_node-1)

def initialization_network(dt,t_0=0,m_0=None,FILES_PATH=''):   
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
 
    # Read files of parameters.
    BG_parameters=np.loadtxt(FILES_PATH + 'Ganglios.dat')
    Nodes_in_connections=np.loadtxt(FILES_PATH + 'Interaction.dat')
    Connection_parameters=np.loadtxt(FILES_PATH + 'Conexiones.dat')
    
    # Initialization Network and Populations.
    Populations=[]
    Network=[]
    
    # Number of nodes and connections in Network.
    N_nodes=BG_parameters.shape[0]
    N_con=Connection_parameters.shape[0]
 
    # Creation of nodes. Populations is 'node' array.
    #for i in np.arange(N_nodes):
    for i in range(N_nodes):
        new_node = node(BG_parameters[i][0],BG_parameters[i][1], 0, 0)
        Populations.append(new_node)

    # Default inital conditions are : t_0=0 and m_0([-d,0])=0 (for all connections)
    if t_0==0:
        len_delay=[int(x) for x in np.floor(Connection_parameters[:,1]/dt)]

        #for i in np.arange(N_con):
        for i in range(N_con):
            new_connection = connection(Connection_parameters[i][0], 
                                        Connection_parameters[i][1], 
                                        Connection_parameters[i][2], 
                                        np.zeros(len_delay[i]), 
                                        Nodes_in_connections[i][0],
                                        Nodes_in_connections[i][1])
            Network.append(new_connection)

    else:

        #for i in np.arange(N_con):
        for i in range(N_con):
            new_connection = connection(Connection_parameters[i][0], 
                                        Connection_parameters[i][1], 
                                        Connection_parameters[i][2], 
                                        m_0[i], 
                                        Nodes_in_connections[i][0],
                                        Nodes_in_connections[i][1])
            Network.append(new_connection)        
        
    
    return Network, Populations, N_con    

def update_input(Network,Populations):    
    """ Update Inputs values xxxxxxx
    """

    for CON in Network:
        value = CON.m[0]
        final_node = CON.Nf
        
        #Version 2019
        #new=np.delete(CON.m,0)
        #Populations[final_node].I += CON.G * value
        #CON.m=new                

        Populations[final_node].I += CON.G * value
        CON.m=CON.m[1:]

def activity(node_input,threshold):
    """ Activity of a node: Semilinear function of input of the node. 

    Input:
        -node_input: numeric value.
        -threshold: numeric value.

    Output:
        -activity: numeric value
    """
    # Version 2019: return np.maximum(0,node_input-threshold)

    return node_input-threshold if node_input-threshold > 0 else 0

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

    #Version 2019
    #omega=2*np.pi*frequency    #[frequency] debe ser igual a [time]^-1
    #stimulus = 0.5* amplitude * (1 + signal.square(omega*time,duty=width))
    #return stimulus

    u = np.mod(time,1/frequency)
    return amplitude if u < width else 0

def solving_system(Network,Populations,DBS_Parameters,t_initial,t_final,DTrans,dt):
    
    N_step= int((t_final-t_initial)/dt)
    N_save= int(DTrans/dt)

    Node_DBS=int(DBS_Parameters[0]-1)
    A_DBS=DBS_Parameters[1]
    freq_DBS=DBS_Parameters[2]
    width_DBS=DBS_Parameters[3]
    
    time=t_initial
    Signals=np.zeros((N_step-N_save,len(Network)))
       
    #for i in np.arange(N_step):
    for i in range(N_step):

        for NODE in Populations:
            NODE.external_input()

        Populations[Node_DBS].I += DBS(time,A_DBS,freq_DBS,width_DBS)

        update_input(Network,Populations)

        #for j in np.arange(len(Network)):
        for j in range(len(Network)):
            
            CON=Network[j]
            initial_node=CON.Ni
            
            act = activity(Populations[initial_node].I,
                           Populations[initial_node].thr)
            m_value = CON.m[-1]
            dm=(act-m_value)*dt/CON.tau
            
            m_value += dm
            
            #Version 2019: CON.m=np.append(CON.m,m_value)
            CON.m=np.concatenate((CON.m,[m_value]))

            if  N_save <= i:
                #Signals[i-N_save,0]= time
                Signals[i-N_save,j] = Populations[initial_node].I	
        
        time += dt
    
    return Signals
