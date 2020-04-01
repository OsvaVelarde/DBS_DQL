# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:25:10 2019

@author: Osvaldo M Velarde

@title: Save / Load Configurations / Agents
"""
#%--------------------------------------------------------------------------

import json
import numpy as np
from dqnagent import DQNAgent

def save_configurations(filename, bgtc_cfg, algorithm_cfg, agent_cfg, critic, reward_cfg):
    """
    Save configuration of script.

    Inputs:
        - bgtc_cfg: dictionary with information about 
                    BGTC network simulation (state space).
        - algorithm_cfg: dictionary with information about
                    DQL algorithm.
        - agent_cfg: dictionary with information about 
                    agent.
        - critic: Deep Q Network for approximation of Q-value

    Output: 
        - a file called 'filename' with input info.
    """

    CFGfile = open(filename,'w')
    ## Ver como agregar comas, y corchetes al inicio y final.    
    CFGfile.write(json.dumps(bgtc_cfg,indent=2))
    CFGfile.write("\n")
    CFGfile.write(json.dumps(algorithm_cfg,indent=2))
    CFGfile.write("\n")
    CFGfile.write(json.dumps(agent_cfg,indent=2))
    CFGfile.write("\n")
    CFGfile.write(critic.to_json(indent=4))
    CFGfile.write("\n")
    CFGfile.write(json.dumps(reward_cfg,indent=2))

#%--------------------------------------------------------------------------

def load_agent(filename,path_training,dim_state,num_action,step):

    # Load configuration.
    with open(filename,'r') as cfg:
        agent_cfg = json.load(cfg)[2] 
 
    # Delete "epsilon greedy" for validation.  
    agent_cfg['epsilon']=0

    # Create agent
    agent = DQNAgent(dim_state, num_action, agent_cfg)
    weights_agent = agent.critic.get_weights()
    num_weights = len(weights_agent)

    # Load weights
    list_w = []
    for k in range(num_weights):
        W = np.loadtxt(path_training + 'Weight_' + str(k) + '_Step_' + str(step) +'.dat')
        list_w.append(W)

    agent.critic.set_weights(list_w)

    return agent

#%--------------------------------------------------------------------------

def config_episode(g1_lims,g2_lims,num_episodes):
    """
    Generation of episodes. In each episode, synaptic efficacies
    are fixed. See map 
    '/mnt/BTE2b/DBS/Agosto-Diciembre-2019/CFC_vs_DBS/Results/[1, 19]_vs_[20, 200]/PLV_Node3_Amp_0.0.eps'

    Inputs:
        - g1_lims: Numeric array (2d: num_regions x 2).
        - g2_lims: Numeric array (2d: num_regions x 2).                                      
        - num_episodes: Int value.

    Outputs:
        - g1: Numeric array (num_episodes x 1)
        - g2: Numeric array (num_episodes x 1)

    Description:
        In the parameter space (g12,g13), "num_regions" states was identified.
        Each state is associated with parameter region. For each region, 
        g1, g2 are generated with uniform distribution (num_episodes/num_regions samples).
        The output is a random permutation of all generated samples. 
    """

    num_regions = len(g1_lims)
    num_samples = int(num_episodes/num_regions)

    g1 = np.array([])
    g2 = np.array([])
    index_region = []

    for rr in range(num_regions):
        aux1=np.random.uniform(g1_lims[rr][0],g1_lims[rr][1],num_samples)
        aux2=np.random.uniform(g2_lims[rr][0],g2_lims[rr][1],num_samples)
        g1 = np.concatenate((g1,aux1))
        g2 = np.concatenate((g2,aux2))
        index_region += [rr] * num_samples 

    i = np.random.permutation(num_episodes)
    g1 = g1[i]
    g2 = g2[i]

    return g1,g2,np.array(index_region)[i]

#%--------------------------------------------------------------------------

def evolution_PD(g1_lims,g2_lims,num_episodes,direction = 'anti'):
    """
    Generation of curve of episodes. In each episode, synaptic efficacies
    are fixed. See map 
    '/mnt/BTE2b/DBS/Agosto-Diciembre-2019/CFC_vs_DBS/Results/[1, 19]_vs_[20, 200]/PLV_Node3_Amp_0.0.eps'
    The curve represents the evolution of PD.
    
    Inputs:
        - g1_lims: Numeric array (1d: 1 x 2).
        - g2_lims: Numeric array (1d: 1 x 2).                                      
        - num_episodes: Int value.
        - direction: 'anti' or 'hor' (antihorario u horario) 

    Outputs:
        - g1: Numeric array (num_episodes x 1)
        - g2: Numeric array (num_episodes x 1)

    Description:
        In the parameter space (g12,g13), we generate a curve.
        The curve is a square with initial point (g1_lims[0],g2_lims[0]).
        'anti' direction: 
        (g1_lims[0],g2_lims[0]) ---> (g1_lims[1],g2_lims[0]) ---> (g1_lims[1],g2_lims[1]) ---> (g1_lims[0],g2_lims[1]) 
        
        'hor' direction: 
        (g1_lims[0],g2_lims[0]) ---> (g1_lims[0],g2_lims[1]) ---> (g1_lims[1],g2_lims[1]) ---> (g1_lims[1],g2_lims[0]) 
    """

    num_sides = 4
    num_samples = int(num_episodes/num_sides)

    vertices = [[g1_lims[0],g2_lims[0]], [g1_lims[1],g2_lims[0]], [g1_lims[1],g2_lims[1]], [g1_lims[0],g2_lims[1]]]

    if (direction != 'anti'):
      vertices = [[g1_lims[0],g2_lims[0]], [g1_lims[0],g2_lims[1]], [g1_lims[1],g2_lims[1]], [g1_lims[1],g2_lims[0]]]
                      
    g1 = np.array([])
    g2 = np.array([])

    for initial_v in range(num_sides):
        
        next_v = (initial_v+1)%num_sides

        aux1=np.linspace(vertices[initial_v][0],vertices[next_v][0],num_samples)
        aux2=np.linspace(vertices[initial_v][1],vertices[next_v][1],num_samples)

        g1 = np.concatenate((g1,aux1))
        g2 = np.concatenate((g2,aux2))

    return g1,g2