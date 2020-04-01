# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:25:10 2019

@author: Osvaldo M Velarde

@title: Biomarkers
"""
#%--------------------------------------------------------------------------

import numpy as np
from scipy import signal

from sklearn.preprocessing import scale

import comodulogram
import filtering 
from segmentation import function_segmentation

# --------------------------------------------------------------------------

def nextpow2(i):
    """
    It computes the lowest power of 2 greater than the number i.

    Input:
        - i: Int value
    
    Output:
        - n: Int value

    Description: 
        'n' is the lowest power of 2 greater than 'i'.
    """

    n = 1
    while n < i: n *= 2
    return n

# ----------------------------------------------------------------------------------

def ModifiedPeriodogram(input_signal,method_parameters,window_parameters):
    """
    Return the Power Spectral Density (PSD) of a signal using Modified Periodogram
    Method.

    Inputs:
        -input_signal: Numeric array (1 x Ns)
        -method_parameters: structure
                            - 'fs': Numeric value (sampling frequency)
                            - 'Pad': Numeric value (number of zero-padding).
        -window_parameters: structure
                            - 'Name': String (name of window)
                            - 'Alpha': Numeric value (window parameter)

    Outputs:
        -f: Numeric array (1 x nfft)
        -PSD: Numeric array (1 x nfft)

    Description:
        'PSD' is Power Spectral Density Estimator of 'input_signal'.
        It is computed using Modified Periodogram with:
        - window characterized by 'window_parameters'
        - sampling frequency 'fs'
        - zero-padding. 
    """

    Ns = input_signal.shape[0]

    fs = method_parameters['fs']
    Pad = method_parameters['Pad']   
    nfft = 2^(nextpow2(Ns) + Pad)
    
    name_window = window_parameters['Name']
    alpha = window_parameters['Alpha']
    sigma=(Ns-1)/(2*alpha)
    window = signal.get_window((name_window, sigma), Ns)
   
    f, PSD = signal.periodogram(input_signal,fs,window,nfft)

    return f,PSD   

# ----------------------------------------------------------------------------------

def power_in_bands(f,PSD,max_range):
    k=len(max_range)
    len_band = [max_range[0]]
    
    #for i in np.arange(1,k):
    for i in range(1,k):
        len_band.append(max_range[i]-max_range[i-1])

    Integral=np.zeros(k)
    
    #for i in np.arange(k):   
    for i in range(k):
        Index=f<max_range[i]
        Dominio=f[Index]
        Imagen=PSD[Index]
        
        Integral[i]=np.trapz(Imagen,Dominio)/len_band[i]

        N_band=len(Dominio)
        PSD = PSD[N_band:]
        f = f[N_band:]

    return Integral       

def PSD_features(input_signal,fs,max_range):

    Pad = 2
    method_parameters = {'Pad':Pad,'fs':fs}

    name_window='gaussian'
    XdB = -30
    alpha=np.sqrt(-2*XdB*np.log(10)/10)
    window_parameters = {'Name':'gaussian', 'Alpha':alpha}

    f, PSD = ModifiedPeriodogram(input_signal,method_parameters,window_parameters)
    PSD_mean_bands = power_in_bands(f,PSD,max_range)

    return PSD_mean_bands

# ----------------------------------------------------------------------------------

def PLV_features(input_signal,fs,freq_bands):

    """
    Inputs: 
      - input_signal
      - fs
      - freq_bands

    Outputs:
      - PLV

    """

    # Parameters for processing the time series ------------------------------
    NOISE_LEVEL = 0.1   # Define the noise level. Fraction of the signal's standard deviation.

    SEGMENT_LENGTH = float('Inf')   # [sec] Length of the sliding segment.
    SEGMENT_OVERLAP  = 0    # Percentual overlaping between successive segments.

    # Checking filter flags.
    CHEKING_FILTER_FLAG = 1
    plotFlag = 0

    # Parameters: HF-Band / LF-Band / Comodulogram
    fXres = 1
    fYres = 1
    CFC_windowParam = {'name':'hann', 'sflag':True}
    # -------------------------------------------------------------------------

    # Processing between frequency bands --------------------------------------  
    for indLF in range(len(freq_bands)):    # Loop across the LF band.
        # Input parameters for the High-frequency band.
        fXmin = (freq_bands[indLF][1]+freq_bands[indLF][0])/2
        fXmax = fXmin + fXres / 2           # Set "fXmax<fXmin+fXres" so just one LF-band is computed.
        fXBw  = freq_bands[indLF][1]-freq_bands[indLF][0]

        for indHF in range(indLF+1,len(freq_bands)):    # Loop across the HF band.
            # Input parameters for the High-frequency band. 
            fYmin = (freq_bands[indHF][1]+freq_bands[indHF][0])/2
            fYmax = fYmin + fYres / 2 # Set "fYmax<fYmin+fYres" so just one HF-band is computed.
            fYBw  = freq_bands[indHF][1]-freq_bands[indHF][0]

            # Input parameters to compute the comodulogram. -------------------

            # Frequency Domain Filter (FDF). 
            # For the description of the parameters see function "function_FDF".
            
            BPFXcfg = {'Bw':fXBw,
                       'zeropadding':0,
                       'freqWindowParam': CFC_windowParam,
                       'timeWindowParam': CFC_windowParam,
                       'conv':'circular',
                       'causal':0,
                       'Nf':0,
                       'function':'function_FDF'}

            BPFYcfg = {'Bw':fYBw,
                       'zeropadding':0,
                       'freqWindowParam': CFC_windowParam,
                       'timeWindowParam': CFC_windowParam,
                       'conv':'circular',
                       'causal':0,
                       'Nf':0,
                       'function':'function_FDF'}

            # Comodulogram. For the description of the parameters see function "function_CFCcfg".            
            CFCcfg = {'fXmin':fXmin, 'fXmax':fXmax, 'fXres':fXres,
                      'fYmin':fYmin, 'fYmax':fYmax, 'fYres':fYres,
                      'fXlookAt':'PHASE', 'fYlookAt':'PHASEofAMPLITUDE',
                      'nX':1, 'nY':1,
                      'BPFXcfg':BPFXcfg, 'BPFYcfg':BPFYcfg,
                      'saveBPFsignal':1,                 
                      'Nbins':[], 'sameNumberOfCycles':0,
                      'CFCmethod':'plv', 'verbose':1,
                      'perMethod':'FFTphaseShuffling',
                      'Nper':100, 'Nrep':1, 'Pvalue':0.05,
                      'corrMultComp':'Bonferroni',
                      'fs':fs}

            CFCcfg = comodulogram.function_setCFCcfg(CFCcfg) # Set the filter and comodulogram parmeters.
            # ----------------------------------------------------------------
                   
            rawSignal = input_signal                          
            # ---------------------------------------------------------              

            # Pre-processing (BGTC Network) ----------------------------
            rawSignal = scale(rawSignal) # Z-score
            
            # ---------------------------------------------------------

            # Additive white Gaussian noise. --------------------------
            # Ref: Eq. (5) "Introduction to the Theory of error, by Yardley Beers"  
            ddof = 1 # 1: divide por N-ddof 
            stdDevSignal = np.std(rawSignal, ddof = ddof ,axis=0) # Compute the standard deviation normalized with "N-ddof".

            # Include additive white Gaussian noise! Ojo
            noise = NOISE_LEVEL * np.random.normal(scale=stdDevSignal,size=np.shape(rawSignal))
            rawSignal = rawSignal + noise
            # ---------------------------------------------------------

            # Check the filters and  ----------------------------------
            # compute the settling time (percLevel). ------------------

            indSettling = 0 # Initialize the index.             
            Nraw = np.shape(rawSignal)[0] # Compute the number of samples.

            #  Check the filters just once.
            if CHEKING_FILTER_FLAG == 1:

                CHEKING_FILTER_FLAG = 0    
        
                # BPFs for the "x" axis of the comodulogram.
                indSettlingLF = filtering.function_checkFilter(CFCcfg['fXcfg']['BPFcfg'], CFCcfg['fs'], Nraw, plotFlag)

                # BPFs for the "y" axis of the comodulogram.
                indSettlingHF = filtering.function_checkFilter(CFCcfg['fYcfg']['BPFcfg'], CFCcfg['fs'], Nraw, plotFlag);
                
                # Compute the maximum settling time.
                indSettling = np.amax(np.concatenate((indSettlingHF,indSettlingLF.T)))

            # Compute the maximum settling time.
            indSettling = max(indSettling, Nraw+1)
            #-----------------------------------------------------------

            # Z-score normalization of the input signal -----------------
            rawSignal = scale(rawSignal)
            #-----------------------------------------------------------

            # Reflect the time series to minimize edge artifacts ----------------- 
            # due to the transient response of the BPFs. -------------------------

            rawSignal = np.concatenate((rawSignal[::-1],rawSignal,rawSignal[::-1]))
            rawSignal = rawSignal.reshape(rawSignal.shape[0],-1)    

            # Compute the Band-Pass Filtering. -----------------------------------
            LFSIGNAL, _ = comodulogram.function_comodulogramBPF(rawSignal, CFCcfg['fXcfg']['BPFcfg'], CFCcfg['fs'], indSettling)
            HFSIGNAL, _ = comodulogram.function_comodulogramBPF(rawSignal, CFCcfg['fYcfg']['BPFcfg'], CFCcfg['fs'], indSettling)

            # Compute the length of "LFSIGNAL" to compute the number of segments.
            N_LFSIGNAL = np.shape(LFSIGNAL)[0]

            # Restore the length of the raw signal.
            rawSignal = rawSignal[indSettling-1:rawSignal.shape[0]-(indSettling-1),:]

            # ---------------------------------------------------------------------

            # Segmentation of the Band-Pass Filtered time series ------------------
            indSegment = function_segmentation(SEGMENT_LENGTH, SEGMENT_OVERLAP, N_LFSIGNAL, CFCcfg['fs'])

            for ii in range(len(indSegment)): # Loop across the segments.

                #Compute the Raw segment --------------------------------------------------
                segRawSignal = rawSignal[indSegment[ii,0]:indSegment[ii,1]]

                #Compute the LF segment ---------------------------------------------------
                segLFSIGNAL = LFSIGNAL[indSegment[ii,0]:indSegment[ii,1]]

                #Compute the HF segment ---------------------------------------------------
                segHFSIGNAL = HFSIGNAL[indSegment[ii,0]:indSegment[ii,1]]

                # -----------------------------------------------------------------------
                Namples_per_segment = len(segLFSIGNAL) # Compute the number of samples of the segment.
                # -----------------------------------------------------------------------

                # Compute the phase/amplitude/frequency for the LF segment ----------------             
                for jj in range(segLFSIGNAL.shape[2]):
                    segLFSIGNAL[:,:,jj] = scale(segLFSIGNAL[:,:,jj],axis=0) #%Z-score normalization of the segment

                # Reflect the segment to minimize edge artifacts due to the transient response of the Hilbert transform.
                segLFSIGNAL = np.concatenate((segLFSIGNAL[::-1,:,:],segLFSIGNAL,segLFSIGNAL[::-1,:,:]))

                # Compute the phase/amplitude/frequency segment.
                segX, _ = comodulogram.function_comodulogramFeature(segLFSIGNAL, CFCcfg['fXcfg'], CFCcfg['fs'], Namples_per_segment+1)
                segX = np.squeeze(segX,axis=2)

                # --------------------------------------------------------------------------

                #%Compute the phase/amplitude/frequency for the HF segment -----------------
                for jj in range(segLFSIGNAL.shape[2]): #%Loop for Bandwidths.
                    segHFSIGNAL[:,:,jj] = scale(segHFSIGNAL[:,:,jj],axis=0)

                # Reflect the segment to minimize edge artifacts due to the transient response of the Hilbert transform.
                segHFSIGNAL = np.concatenate((segHFSIGNAL[::-1,:,:],segHFSIGNAL,segHFSIGNAL[::-1,:,:]))

                # Compute the phase/amplitude/frequency segment.
                segY, _ = comodulogram.function_comodulogramFeature(segHFSIGNAL, CFCcfg['fYcfg'], CFCcfg['fs'], Namples_per_segment+1)

                # --------------------------------------------------------------------------

                # Compute the Phase Locking Value ------------------------------------------
                PLV, _, _ = comodulogram.function_PLV(segX, segY, 0, 0, CFCcfg)
                
    return PLV