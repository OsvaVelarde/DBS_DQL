# -*- coding: utf-8 -*-
"""
Created on October 22 - 2019
Version January 2020

Application of Deep Q-learning algotithm (DQL) as scheme of Closed-Loop DBS.

About DQL:
Environment -> State (s) ->

Environment:
The model of BGTC network is composed by three nodes. 
The state of model is controlled by synaptic efficacies (g1->2,g1->3) between nodes.
The bifurcation structure of BGTCNetwork: -Steady state -LF oscillation -HF oscillation -SH bifurcation -PEI mechanism
It is possible to apply a stimulation to the network (Deep Brain Stimulation - DBS)

The configuration of system is discussed in "Velarde et al 2019".
The implementation of BGTC network is in the module BGTCNetwork.

State space (s):
We consider the signal I_{3} (input of node 3) in BGTC network as 'state of system' (s).

Action space (a):
We consider the amplitude of DBS as 'action on the system' (a). After the stimulation a, the state s changes to s'. 

Reward (R):
Our goal is to minimize the pathological features: ('LF power' and 'PLV') with the least possible stimulation power.
We compute 'LF power and PLV' in the new state s'.

The proposed reward function is:
................

Episode:
Para cada episodio, configuramos el sistema con valores de ganancias aleatorios (g12,g13).
En cada uno, el agente probará diferentes estimualciones (a) en la misma configuración.


@author: Osvaldo M Velarde

@title: Deep Q-learning Algorithm in Closed-Loop DBS
"""

import numpy as np
import os
import sys

import sklearn.preprocessing

from datetime import date
import json

import bgtc_network_v2 as bgnet
import rewards as rew
from dqnagent import DQNAgent
from downsampling import downsampling

__author__ = "Velarde, O M"
__email__ = "osva.m.velarde@gmail.com"

FILES_PATH = '/mnt/BTE2b/DBS/2020/DQLearning/TrainingFiles/'
OUTPUT_PATH = '/mnt/BTE2b/DBS/2020/DQLearning/Trainings/'
DATE = str(date.today())
CFG_FILENAME =  OUTPUT_PATH + 'Cfg_' + DATE +'.json'
EPI_FILENAME =  OUTPUT_PATH + 'Episodes_' + DATE +'.dat'

def config_episode(g1_lims,g2_lims,num_episodes,filename):
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

    for rr in range(num_regions):
        aux1=np.random.uniform(g1_lims[rr][0],g1_lims[rr][1],num_samples)
        aux2=np.random.uniform(g2_lims[rr][0],g2_lims[rr][1],num_samples)
        g1 = np.concatenate((g1,aux1))
        g2 = np.concatenate((g2,aux2))

    i = np.random.permutation(num_episodes)
    g1 = g1[i]
    g2 = g2[i]

    np.savetxt(filename,np.concatenate((g1,g2)).reshape(2,num_episodes).T)

    return g1,g2

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


def load_agent(dim_state,num_action,step):

    # Load configuration.
    Cfg_filename = CFG_PATH + 'Cfg_' + TRAINING_DATE +'.json'
    with open(Cfg_filename,'r') as cfg:
        agent_cfg = json.load(cfg)[2] 
 
    # Delete "epsilon greedy" for validation.  
    agent_cfg['epsilon']=0

    # Create agent
    agent = DQNAgent(dim_state, num_action, agent_cfg)
    weights_agent = agent.critic.get_weights()
    num_weights = len(weights_agent)

    # Load weights
    Train_folder = CFG_PATH + TRAINING_DATE +'/'
    list_w = []
    for k in range(num_weights):
        W = np.loadtxt(Train_folder + 'Weight_' + str(k) + '_Step_' + str(step) +'.dat')
        list_w.append(W)

    agent.critic.set_weights(list_w)

    return agent
    
if __name__ == "__main__":
    
    # 1. Parameters of DQL algorithm.  -------------------------------------
    num_episodes = 3000              # Number of episodes 3000
    num_steps = 50                   # Number of actions to execute 50
    batch_size = 50                  
    num_copy = 1000                  # Cada num_copy steps se copiará critic -> actor
    num_save = 500                   # Cada num_save salvará el estado del aprendizaje
    training_start = 200

    algorithm_cfg = {'num_episodes':num_episodes,
                     'num_steps':num_steps,
                     'batch_size':batch_size,
                     'num_copy':num_copy,
                     'num_save':num_save,
                     'training_start':training_start}
    # ----------------------------------------------------------------------

    # 2. Configuration of BGTC network and Episodes. -----------------------
    initial_conf_network = np.loadtxt(FILES_PATH + 'Conexiones.dat')
    g12_lims = [[0.25,0.5], [1,2], [0,0.25], [0.5,1], [1.25,2]]
    g13_lims = [[0,0.25], [0,0.5], [0.5,1], [1.3,2], [1,2]]

    episode_file = FILES_PATH + DATE
    g12,g13 = config_episode(g12_lims,g13_lims,num_episodes,EPI_FILENAME)

    # Parameters for simulations of BGTC network. --------------------------
    dt = 0.05               # [ms] - Time resolution
    fs = 1000/dt            # [Hz] - Sampling frequency
    tf = 3500 #10000              # [ms] - Generate the simulation in time (t0,t0 + tf)
    dt_transitory = 1000#7500    # [ms] - Delete the transitory interval (t0,t0 + dt_transitory)

    bgtc_cfg = {'g12_lims':g12_lims,
                 'g13_lims':g13_lims,
                 'dt':dt,
                 'tf':tf,
                 'dt_transitory':dt_transitory}

    # Downsampling parameters. 
    F3dB = 1000                                     # Cutoff frequency for the anti-aliasing LPF [Hz]
    LPF_WINDOW_PARAM = {'name':'tukeywide','r':0.1} # Tukey window with f2 configured as Cutoff frequency at 0dB.
    LPFcfg = {'f2':F3dB, 'zeropadding':0, 
              'freqWindowParam':LPF_WINDOW_PARAM, 'timeWindowParam':{'name':'box'} ,
              'function':'function_FDF'}            # Configure the anti-aliasing LPF.
    decFactor = int(fs/(2*F3dB))                    # Downsampling - Decimation factor
    fs = fs / decFactor                             # [samples/s] Nominal sampling rate.

    # ----------------------------------------------------------------------   
    dim_state = int(fs*tf/1000) - int(fs*dt_transitory/1000) # dimension of state space (length of simulation).
    # ----------------------------------------------------------------------
    
    # 3. Load Action Space -------------------------------------------------
    action_space = np.loadtxt(FILES_PATH + 'Action_space.dat')
    num_action = action_space.shape[0]

    # ----------------------------------------------------------------------

    # 4. Configuration of agent in DQL. ------------------------------------
    max_memory = 500        
    gamma = 0
    per_step_eps = 0.5   
    epsilon = 1  
    epsilon_min = 0.1 
    epsilon_decay = epsilon_min ** (1/(per_step_eps*num_episodes*num_steps))    
    learning_rate = 0.0001    

    agent_cfg = {'max_memory':max_memory, 'gamma':gamma, 'epsilon': epsilon,
                    'epsilon_min':epsilon_min, 'epsilon_decay':epsilon_decay, 
                    'learning_rate':learning_rate}

    # Initialization of Agent
    agent = DQNAgent(dim_state, num_action, agent_cfg)
    # ----------------------------------------------------------------------

    # 5. Reward function Parameters ----------------------------------------
    psd_bands = [4, 10, 20, 100, 200]          # [Hz]. Frequency range for Power calculation. (verificar que no cambia usar np.array)
    plv_bands = [[1, 19], [20, 200]]           # [Hz]. Frequency bands of interest for  PLV computation.

    method = 'LfPlvAmpExp'
    constants = [1, -50, -3, -0.25]
    reward_cfg = {'fs':fs, 'psd_bands':psd_bands, 'plv_bands':plv_bands,'constants':constants, 'method':method}

    # Save all configurations 
    save_configurations(CFG_FILENAME, bgtc_cfg, algorithm_cfg, agent_cfg, agent.critic, reward_cfg)

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    # Output directory and files --------------------------------------------
    dir_path = OUTPUT_PATH + DATE + '/'
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass

    # Output files
    output_filename = dir_path + 'Training.dat' 
    output_file = open(output_filename,'w')
    output_filename = dir_path + 'Rewards_Episode.dat'
    reward_file = open(output_filename,'w')

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    # Initialization of BGTC network
    network = bgnet.network(dt=dt,FILES_PATH=FILES_PATH)

    #for ep in np.arange(num_episodes):
    for ep in range(num_episodes):

        # Configuration of BGTC in the episode ----------------------------- 
        network.Connections[1].G = g12[ep]  
        network.Connections[2].G = g12[ep]
        print('Episode = ', ep,'G12 = ', g12[ep],'G13 =',g13[ep])

        # Initial state: BGTC dynamic without stimulation ------------------
        dbs_config = [2, 0, 1, 0.5]                               # Amplitude DBS = 0 in N2.
        state = network.solving_system(dbs_config,tf,dt_transitory)[:,2] # Recording from N3
        state = sklearn.preprocessing.scale(state)

        LPFcfg['freqWindowParam']['name']='tukeywide'
        LPFcfg['freqWindowParam']['r'] = 0.1
        state = downsampling(state,1000/dt,decFactor,LPFcfg)

        state = state + np.random.normal(scale = 0.001,size=np.shape(state))

        # Reward in episode ------------------------------------------------
        total_reward = 0

        # Loop in training step (same episode) -----------------------------
        for step in range(num_steps):

            # Training step 
            training_step = num_steps * ep + step

            # Selection of action.
            index_action = agent.select_action(state)
            action = action_space[index_action]
            
            # Execution of action and detection of new state.
            t1 = network.time + tf
            next_state = network.solving_system(action,t1,dt_transitory)[:,2]
            next_state = sklearn.preprocessing.scale(next_state)    
            
            LPFcfg['freqWindowParam']['name']='tukeywide'
            LPFcfg['freqWindowParam']['r'] = 0.1
            next_state = downsampling(next_state,1000/dt,decFactor,LPFcfg)

            next_state = next_state + np.random.normal(scale = 0.001,size=np.shape(next_state))

            # Compute Reward.
            reward = rew.reward_function(state, action[1], next_state, reward_cfg)
            total_reward = total_reward + reward

            # Variable 'done' indicates if this step is the last in the current episode.
            if step == num_steps-1:
                done=0
            else:
                done=1

            # Remember the experience (s,a,s',R,done)
            agent.remember(state, index_action, reward, next_state, done)
            
            # Training
            if training_step > training_start:
                q_history, q_evaluate = agent.replay(batch_size)

                if training_step%num_copy == 0:
                    agent.actor = agent.copy_to_actor()

                print('Step:',step,'Epsilon', '%.2f' % agent.epsilon,
                      'Action:',action[1],'Reward =', '%.4f' % reward, 'q_history =', '%.4f' % q_history,
                      'q_evaluate =', '%.4f' % q_evaluate)

            else:

                print('Step:',step,'Epsilon', '%.2f' % agent.epsilon,
                      'Action:',action[1],'Reward =', '%.4f' % reward, 'No training time')

            # Saving Training State and information Q-network 
            if training_step%num_save == 0 and training_step > training_start:

                print(training_step,'%.2f' % agent.epsilon,'%.4f' % reward, '%.4f' % q_history, 
                      '%.4f' % q_evaluate, file = output_file)    

                weights_critic = agent.critic.get_weights()

                #for k in np.arange(len(weights_critic)):
                for k in range(len(weights_critic)):
                    output_filename = dir_path + 'Weight_' + str(k) + '_Step_' + str(training_step) + '.dat'
                    np.savetxt(output_filename,weights_critic[k],fmt='%.3f')

            # New state is the initial state for next step.
            state = next_state

        network.reset() # Time = 0, Initial Conditions = 0

        print('Total Reward: ',total_reward)
        print(ep, g12[ep], g13[ep], total_reward, file = reward_file)

    output_file.close()