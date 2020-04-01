# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:25:10 2019

@author: Osvaldo M Velarde

@title: Biomarkers and Reward Signal
"""
#%--------------------------------------------------------------------------
import biomarkers as biomk
import numpy as np

def reward_BetaGamma(state,action,next_state,reward_cfg):

    """
    Reward function: Polynomial function of Beta and Gamma Power 
    of next state. A * (B - beta)^p * (C-gamma)^q

    Input:
        -state: Numeric array (1 x N_samples)
        -action: Numeric value
        -next_state: Numeric array (1 x N_samples)
        -reward_cfg: Dictionary
                - fs: Numeric value (sampling frequency)
                - bands: Numeric array (M limits of frequency bands)
                - constants: [A,B,p,C,q]
    Output:
        -R: Numeric array

    Description:
        We compute the beta/gamma power for 'next_state'. See 'help'
        of module 'Biomarkers'.
        'R' is a polynomial function of Gamma and Beta.
    """

    fs = reward_cfg['fs']
    freq_bands = reward_cfg['bands']
    K_poly = reward_cfg['constants']

    mean_power = biomk.PSD_features(next_state,fs,freq_bands)
    beta_power = mean_power[2]
    gamma_power = mean_power[4]
    
    R = K_poly[0] * (K_poly[1] - beta_power)**K_poly[2] * (K_poly[3] - gamma_power)**K_poly[4]
    return R

def reward_BetaAmpPoly(state,action,next_state,reward_cfg):

    """
    Reward function: Polynomial function of Beta Power of next state
    and Amplitude of action. A * (B - beta)^p * (C- action)^q

    Input:
        -state: Numeric array (1 x N_samples)
        -action: Numeric value
        -next_state: Numeric array (1 x N_samples)
        -reward_cfg: Dictionary
                - fs: Numeric value (sampling frequency)
                - bands: Numeric array (M limits of frequency bands)
                - constants: [A,B,p,C,q]
    Output:
        -R: Numeric array

    Description:
        We compute the beta power for 'next_state'. See 'help'
        of module 'Biomarkers'.
        'R' is a polynomial function of Beta and Amplitude.
    """

    fs = reward_cfg['fs']
    freq_bands = reward_cfg['bands']
    K_poly = reward_cfg['constants']

    mean_power = biomk.PSD_features(next_state,fs,freq_bands)
    beta_power = mean_power[2]
    
    R = K_poly[0] * (K_poly[1] - beta_power)**K_poly[2] * (K_poly[3]-action)**K_poly[4]
    return R

def reward_BetaAmpExp(state,action,next_state,reward_cfg):

    """
    Reward function. Product between a exponential function of Amplitude 
    and a polynomial function of beta power.
    A * (B - beta)^p * exp(-amp/C)

    Input:
        -state: Numeric array (1 x N_samples)
        -action: Numeric value
        -next_state: Numeric array (1 x N_samples)
        -reward_cfg: Dictionary
                - fs: Numeric value (sampling frequency)
                - bands: Numeric array (M limits of frequency bands)
                - constants: [A,B,p,C,q]
    Output:
        -R: Numeric array

    Description:
        We compute the beta power for 'next_state'. See 'help'
        of module 'Biomarkers'.
        'R' is a product between a exponential function of amplitude
        and polynomial function of beta power.
    """

    fs = reward_cfg['fs']
    freq_bands = reward_cfg['bands']
    K_poly = reward_cfg['constants']

    mean_power = biomk.PSD_features(next_state,fs,freq_bands)
    beta_power = mean_power[2]
    
    R = K_poly[0] * (K_poly[1] - beta_power)**K_poly[2] * np.exp(-action/K_poly[3])
    return R

def reward_LfPLVAmpExp(state,action,next_state,reward_cfg):
    """
    Reward function. Exponential function of linear combination.
    A * exp( B * Lf_power + C * PLV + D * amp)

    Input:
        -state: Numeric array (1 x N_samples)
        -action: Numeric value
        -next_state: Numeric array (1 x N_samples)
        -reward_cfg: Dictionary
                - fs: Numeric value (sampling frequency)
                - bands: Numeric array (M limits of frequency bands)
                - constants: [A,B,C,D,E]
    Output:
        -R: Numeric array

    Description:
        We compute LF power and PLV for 'next_state'. See 'help'
        of module 'Biomarkers'.
        'R' is exponential function of linear combination of amplitude, LF power and PLV.
    """

    fs = reward_cfg['fs']
    psd_bands = reward_cfg['psd_bands']
    plv_bands = reward_cfg['plv_bands']
    const = reward_cfg['constants']

    # Lf power
    mean_power = biomk.PSD_features(next_state,fs,psd_bands)
    #Lf_power = mean_power[2]
    Lf_power = mean_power[0]

    # PLV
    PLV = biomk.PLV_features(next_state,fs,plv_bands)

    LCombination = const[1] * Lf_power + const[2] * np.abs(PLV) + const[3] * action
    R = const[0] * np.exp(LCombination)

    return Lf_power, np.abs(PLV), R

LIST_REWARDS = {'BetaGamma':reward_BetaGamma,
                'BetaAmpPoly':reward_BetaAmpPoly,
                'BetaAmpExp':reward_BetaAmpExp,
                'LfPlvAmpExp':reward_LfPLVAmpExp}
                
def reward_function(initial_state,action,final_state,reward_cfg):
    method = reward_cfg['method']
    function = LIST_REWARDS.get(method, lambda:"Invalid method")
    return function(initial_state,action,final_state,reward_cfg)