# -*- coding: utf-8 -*-
"""
Created on September 2019

Description:
In this script we synthesize the weighting function for windowing in time domain.
%Author: Dami�n Dellavale.

%Refs:
%Matlab -> Help -> window
%http://www.mathworks.com/help/signal/ref/window.html
%https://en.wikipedia.org/wiki/Window_function

@author: Osvaldo M Velarde

@title: Window.
"""

import scipy.signal

# Observation - Relations (sigma-alpha-XdB):
# sigma = (n-1)/(2*alpha)
# alpha = sqrt(-2*XdB/(10 log10 e)) // Power attenuation
# alpha = sqrt(-XdB/(10 log10 e)) // Amplitude attenuation (usual)
# En general, XdB=-30.


def function_window(windowParam, windowLength):
    """
    Inputs:
        - windowParam: Structure. Parameters of the window function.
                       - 'name':  String {'gausswin','hamming','hann','tukey','rectwin'}. 
                                  Name defining the window type. 
                       - 'sigma': Numeric value. 
                                  Parameter for the gausswin:standard deviation of a 
                                  Gaussian random variable.
                       - 'sflag': {False ('periodic'), True('symmetric')(default)}. 
                                  Sampling parameter for hamming and hann windows.
                       - 'r':     Numeric value.
                                  *If r in (0,1): A Tukey window is a rectangular window with the 
                                  first and last 100*r/2 percent of the samples equal to parts of a cosine.
                                  *If you input r ≤ 0, you obtain a rectwin window. 
                                  *If you input r ≥ 1, you obtain a hann window.

        - windowLength: Int value. Length of the window function.

    Outputs:
        - windowFunction. Numeric array (). Synthesized window function.

    % Info sflag:
    The 'periodic' flag is useful for DFT/FFT purposes, such as in spectral analysis.
    The DFT/FFT contains an implicit periodic extension and the periodic flag enables a signal windowed
    with a periodic window to have perfect periodic extension. When 'periodic' is specified,
    hamming/hann computes a length "windowLength+1" window and returns the first "windowLength" points.

    When using windows for filter design, the 'symmetric' flag should be used. 
    """

# %Argument completion ------------------------------------------------------
# if (nargin < 2)||isempty(windowParam)||isempty(windowLength),...
#    error('MATLAB:function_window','Input argument error.');
# end
# %--------------------------------------------------------------------------

# %Check the input arguments ------------------------------------------------
# assert(isstruct(windowParam), 'Input argument error in function "function_window": windowParam must be a structure array.');
# %--------------------------------------------------------------------------


    LIST_WINDOWS = {'gausswin':scipy.signal.gaussian,
                    'hamming':scipy.signal.hamming,
                    'hann':scipy.signal.hann,
                    'tukey':scipy.signal.tukey}

    LIST_PARAMETERS = {'gausswin':'sigma',
                       'hamming':'sflag',
                       'hann':'sflag',
                       'tukey':'r'}

    name = windowParam['name']

    if name in LIST_WINDOWS.keys():
        functwin = LIST_WINDOWS.get(name, lambda:"Invalid method")
        parameter = windowParam[LIST_PARAMETERS[name]]
        return functwin(windowLength,parameter)
    else:
        return scipy.signal.boxcar(windowLength)

