# -*- coding: utf-8 -*-
"""
Created on September 2019
@authors: Osvaldo M Velarde - Damián Dellavale - Javier Velez
@title: Package - "filtering"
"""
import numpy as np
import spectrum

# Optional windows. ----------------------------------------------------------------
# ----------------------------------------------------------------------------------
def function_gausswinW1(f1,f2,*args):
    sigma = args[0]
    n = args[1]
    name = 'gausswin'
    f1aux = ( f1 + f2 - (f2-f1) * (n-1)/(2*sigma*np.sqrt(np.log(2))) ) / 2  # Cutoff frequency at -3dB.
    f2aux = ( f1 + f2 + (f2-f1) * (n-1)/(2*sigma*np.sqrt(np.log(2))) ) / 2 # Cutoff frequency at -3dB.
    return name, f1aux, f2aux

def function_gausswinW2(f1,f2,*args):
    n = kwargs['Ns']
    sigma = kwargs['sigma']
    name = 'gausswin'
    f1aux = ( f1 + f2 - (f2-f1) * (n-1)/(2*sigma) ) / 2  # Cutoff frequency at the inflection point.
    f2aux = ( f1 + f2 + (f2-f1) * (n-1)/(2*sigma) ) / 2  # Cutoff frequency at the inflection point.
    return name, f1aux, f2aux

def function_hannW1(f1,f2,*args):
    name = 'hann'
    f1aux = f1 - (f2-f1)/(np.pi-2)     # Cutoff frequency at -3dB.
    f2aux = f2 + (f2-f1)/(np.pi-2)     # Cutoff frequency at -3dB.
    return name,f1aux,f2aux

def function_hannW2(f1,f2,*args):
    name = 'hann'
    f1aux = (3*f1 - f2)/2              # Cutoff frequency at the inflection point.
    f2aux = (3*f2 - f1)/2              # Cutoff frequency at the inflection point.
    return name, f1aux, f2aux

def function_tukeyW(f1,f2,*args):
    name = 'tukey'
    r = args[0]
    f1aux = ( f1*(r-2) + r*f2 ) / (2*(r-1)) # Cutoff frequency at 0dB.
    f2aux = ( f2*(r-2) + r*f1 ) / (2*(r-1)) # Cutoff frequency at 0dB.
    return name, f1aux, f2aux

list_windowParam = {'gausswinwide1':'sigma',
                    'gausswinwide2':'sigma',
                    'hannwide1':'sflag',
                    'hannwide2':'sflag',
                    'tukeywide':'r'}

list_optwin = {'gausswinwide1':function_gausswinW1,
               'gausswinwide2':function_gausswinW2,
               'hannwide1':function_hannW1,
               'hannwide2':function_hannW2,
               'tukeywide':function_tukeyW}

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# Filter types: Limits -------------------------------------------------------------
# ----------------------------------------------------------------------------------
def function_limsBPF(f1aux, f2aux,*args):
    if f1aux>=0 and f2aux<=args[2]/2:
        return f1aux, f2aux
    else:
        print('Frequencies out of range for ' + windowName + ' window. \n' + \
              'f1aux=' + str(f1aux) + ' -- f2aux=' + str(f2aux) + '\n' + \
              FDFcfg['windowParam']['name'] + ' implemented instead with: \n' + \
              'f1=' + str(f1) + ' -- f2=' + str(f2))

def function_limsHPF(f1aux, f2aux,*args):
    if f1aux>=0:
        return f1aux, args[1]
    else:
        print('Frequencies out of range for ' + windowName + ' window. \n' + \
              'f1aux=' + str(f1aux) + '\n' + \
              FDFcfg['windowParam']['name'] + ' implemented instead with: \n' + \
              'f1=' + str(f1))

def function_limsLPF(f1aux,f2aux,*args):
    if f2aux<=args[2]/2:
        return args[0], f2aux
    else:
        print('Frequencies out of range for ' + windowName + ' window. \n' + \
              'f2aux=' + str(f2aux) + '\n' + \
              FDFcfg['windowParam']['name'] + ' implemented instead with: \n' + \
              'f2=' + str(f2))

list_limsftype = {'bpf': function_limsBPF,
                  'hpf': function_limsHPF,
                  'lpf': function_limsLPF}
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# Filter types: Windows ------------------------------------------------------------
# ----------------------------------------------------------------------------------
def function_winBPF(windowParam,windowLength):
    window = spectrum.function_window(windowParam, windowLength)
    return window

def function_winHPF(windowParam,windowLength):
    # Scaling the window prameters.
    if 'sigma' in windowParam.keys(): #gausswin
        windowParam['sigma'] = 2*windowParam['sigma']

    if 'r' in windowParam.keys():   #tukey
        windowParam['r'] = windowParam['r']/2 

    # Compute the window doubling the number of samples.
    window = spectrum.function_window(windowParam, 2*windowLength)
    window = window[0:windowLength]

    return window

def function_winLPF(windowParam,windowLength):
    # Scaling the window prameters.
    if 'sigma' in windowParam.keys(): #gausswin
        windowParam['sigma'] = 2*windowParam['sigma']

    if 'r' in windowParam.keys():   #tukey
        windowParam['r'] = windowParam['r']/2 

    # Compute the window doubling the number of samples.
    window = spectrum.function_window(windowParam, 2*windowLength)
    window = window[windowLength:]

    return window

list_winftype = {'bpf': function_winBPF,
                 'hpf': function_winHPF,
                 'lpf': function_winLPF}
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

def function_FDF(signal, FDFcfg, fs=1):

    """
    Description:
    In this script the Frequency Domain Filtering (FDF) is implemented.
    If f0 and Bw or f1 and f2 are specified, a Band-Pass filter is implemented.
    If just f2 (or f1=NaN) is specified, a Low-Pass filter is implemented.
    If just f1 (or f2=NaN) is specified, a High-Pass filter is implemented.

    Inputs:
        -signal: Numeric array (samples x channels). Data.
                 If "signal=NaN", the "indSettling" and "FilterMag" are computed,
                 but the filter is not applied to the signal.
        -FDFcfg: Structure. Frequency Domain Filtering configuration.
                    - 'f0': Numeric value. Center frequency of the BPF [Hz].
                    - 'Bw': Numeric value. Bandwidth of the Band-Pass Filter [Hz].
                    - 'f1': Numeric value. Lower cutoff frequency (-Inf dB) [Hz].
                    - 'f2': Numeric value. Higher cutoff frequency (-Inf dB) [Hz].
                    - 'zeropadding': Numeric value. Padding flag.
                                *If pad>=0, is the "(pad+1)-th" next power of 2 greater 
                                 than Ns (length of "signal").
                                *If pad<0, is the "pad-th" previous power of 2 lesser 
                                 than Ns (length of "signal").
                    - 'conv': String. Convolution flag.
                            If conv='linear', zero-padding is implemented so the product of 
                            the FFTs results to the LINEAR convolution in the time domain.
                            Otherwise, no zero-padding is implemented so the product of the 
                            FFTs results to the CIRCULAR convolution in the time domain. 
                    - 'causal': Boolean. Causal filtering flag.
                            1: Causal filtering is implemented. That is, the filter kernel (h) is shifted to the rigth so h=0 for all t<0.
                            0: Non-causal filtering. The filter kernel (h) is centered at t=0.
                            IMPORTANT: The 'causal' flag applies only in the case of linear convolution (conv='linear').
                    - 'freqWindowParam': Structure. Parameters of the window function in the frequency domain.
                    - 'timeWindowParam': Structure. Parameters of the window function in the time domain.
                            IMPORTANT: It only applies in the case of linear convolution (conv='linear').
                            - 'name':  String. Name defining the window type.
                            - 'sigma': Numeric value. Parameter for the gausswin: STD of gaussian variable.
                            - 'sflag': Boolean value. (True = symmetric - default, False = periodic).
                    -'Nf': Int value. Number of frequencies to evaluate the BPF's frequency response.  
        -fs: Numeric value. Sampling rate [Hz].

    Outputs:
        -filteredSignal: Numeric array (samples x channels). Filtered signal.
        -indSettling:    Numeric value. Index corresponding to the output settling time.
        -FilterMag:      Numeric array. Magnitude of the Filter's frequency response.
        -fmag:           Numeric array. Frequency vector corresponding to the BPF's frequency response.

    """

    # Argument completion ------------------------------------------------------
    if 'f0' in FDFcfg.keys() and 'Bw' in FDFcfg.keys():
        if np.size(FDFcfg['f0'])==1 and np.size(FDFcfg['Bw'])==1:
            f1 = FDFcfg['f0'] - FDFcfg['Bw']/2
            f2 = FDFcfg['f0'] + FDFcfg['Bw']/2
            FilterType = 'bpf'
        else:
            print('Error en las dimensiones de f0 y Bw')
    elif 'f1' in FDFcfg.keys() and 'f2' in FDFcfg.keys():
        if len(FDFcfg['f1'])==1 and len(FDFcfg['f2'])==1:
            f1 = FDFcfg['f1']
            f2 = FDFcfg['f2'] 
            FilterType = 'bpf'
        else:
            print('Error en las dimensiones de f1,f2')
    elif 'f1' in FDFcfg.keys():
        if len(FDFcfg['f1'])==1:
            f1 = FDFcfg['f1']
            f2 = fs/2 
            FilterType = 'hpf'
        else:
            print('Error en la dimension de f1')
    elif 'f2' in FDFcfg.keys():
        if np.size(FDFcfg['f2'])==1:
            f1 = 0
            f2 = FDFcfg['f2'] 
            FilterType = 'lpf'
        else:
            print('Error en la dimension de f2')
    else:
        print('No estan definidos los limites o el centro')

    if not 'Nf' in FDFcfg.keys(): 
        FDFcfg['Nf'] = 2 ** 10      #Default value for the number of frequencies to evaluate the BPF's frequency response.

    if not 'conv' in FDFcfg.keys():
        FDFcfg['conv'] = 'circular'

    if not 'causal' in FDFcfg.keys():
        FDFcfg['causal'] = 0
    # --------------------------------------------------------------------------

    # Parameters ---------------------------------------------------------------
    f1aux = f1
    f2aux = f2

    Ns = signal.shape[0]  # Number of samples.
    Nch = signal.shape[1] # Number of channels.
    Nf = FDFcfg['Nf']     # Number of frequencies to evaluate the BPF's frequency response.
    
    #nextpow2(x): int(ceil.log2(x)))
    nfft = 2 ** (int(np.ceil(np.log2(Ns))) + FDFcfg['zeropadding']) # [samples]. Length of the FFT.
    onesidedLength = int((nfft - nfft%2)/2)  # + 1 [samples]. Length of the onsided PSD depend on the nfft.
    # --------------------------------------------------------------------------
    # Change the length of the signal to the corresponding power of 2, ---------
    # in order to use the fft() function. --------------------------------------
    if FDFcfg['zeropadding'] < 0:
        signal = signal[1:nfft,:] # If nfft < Ns. %Truncation.  
    else:
        signal = np.concatenate((signal,np.zeros((nfft-Ns, Nch)))) # If nfft >= Ns. % Zero Padding
    # --------------------------------------------------------------------------

    # Compute the parameters for customized windows ----------------------------
    windowName = FDFcfg['freqWindowParam']['name']

    if windowName in list_windowParam.keys():
        optwinCfg = list_optwin.get(windowName, lambda:"Invalid method")
        auxparam  = FDFcfg['freqWindowParam'][list_windowParam[windowName]]
        FDFcfg['freqWindowParam']['name'], f1aux, f2aux = optwinCfg(f1,f2,auxparam,Ns)

    # --------------------------------------------------------------------------
    
    # Check the values of f1aux and f2aux. - Switch for filter's type.
    if FilterType in list_limsftype.keys():
        functlimsFType = list_limsftype.get(FilterType, lambda:"Invalid method")
        f1, f2 = functlimsFType(f1aux,f2aux,f1,f2,fs)
    # --------------------------------------------------------------------------

    # Compute the Frequency response of the filter -----------------------------
    fmag = np.linspace(f1,f2,Nf)
    FilterMag = spectrum.function_window(FDFcfg['freqWindowParam'], Nf)

    # Compute the transient response of the filter. 
    # Devolver un error si la señal no está normalizada.
    indSettling = round(10*Ns/100)  # Ten percent of the samples.
    # --------------------------------------------------------------------------

    if np.isnan(signal[0,0]):
        return np.nan, indSettling, FilterMag, fmag

    # --------------------------------------------------------------------------

    # Compute the frequency vector
    f = np.linspace(0,fs/2,onesidedLength)
    faux = -f[1:len(f)-1]                       
    f = np.concatenate((f,faux[::-1])) #verificar f = [f(1:end); -f(end-1:-1:2)] 
    # --------------------------------------------------------------------------

    # Compute the indices for locating the window function.
    indf1 = np.where(f>=f1)[0][0] 
    indf2 = np.where(f>=f2)[0][0]

    # In case of f2=fs/2, the following is necessary because the negative frequencies
    # (second half of the fft vector) do not include the fs/2 frequency.
    indf2Neg = np.where(f<=-f2)

    if f2 >= fs/2 or indf2Neg[0].size == 0:
        indf2Neg = onesidedLength
    else:
        indf2Neg = indf2Neg[0][-1]

    # Compute the window -------------------------------------------------------
    windowLength = indf2-indf1+1 # Compute the window length.

    if FilterType in list_winftype.keys():
        functwinFType = list_winftype.get(FilterType, lambda:"Invalid method")
        windowFunction = functwinFType(FDFcfg['freqWindowParam'],windowLength)

    # Compute the filter in frequency domain. -----------------------------------
    H = np.zeros((signal.shape[0],))
    H[indf1:indf1+windowLength] = windowFunction
    H[indf2Neg:indf2Neg+windowLength] = np.flipud(windowFunction)

    #In case of f1=0Hz or f2=fs/2, this is necessary because the negative frequencies
    #(second half of the fft vector) do not include the 0Hz and fs/2 frequencies.
    H = H[0:nfft]

    # --------------------------------------------------------------------------

    # Windowing in time domain (Window method for FIR filter design) -----------

    # Compute the impulse response of the filter: CONTROLAR LOS RESULTADOS Y VER. PROBAMOS SACAR ESTO.
    h = np.fft.ifft(H,nfft,0) # The zero-time component is in the center of the array.

    # Rearranges h by shifting the zero-time component to the left of the array. 
    h = np.fft.fftshift(h) 

    # Apply the window in time domain.
    win = spectrum.function_window(FDFcfg['timeWindowParam'], nfft)
    h = np.multiply(h,win)

    # Rearranges h by shifting the zero-time component back to the center of the array.
    h = np.fft.fftshift(h)
    # -------------------------------------------------------------------------

    # # Implementar la convolucion lineal
    # # Linear convolution in time domain -----------------------------
    # if FDFcfg['conv'] == 'linear':

    #     # Update the signal's length required so the product of the FFTs results to the LINEAR convolution in time domain.
    #     nfft = 2*nfft # Minimum required length = size(signal,dim) + size(h,dim) - 1 = 2*nfft - 1;
    #     onesidedLength = (nfft - nfft%2)/2 + 1 # [samples]
        
    #     # Update the frequency vector.
    #     f = np.linspace(0,+fs/2,onesidedLength).T
    #     f = [f(1:end); -f(end-1:-1:2)];
        
    #     # Apply the zero-padding on the filter kernel.
    #     if FDFcfg.causal, # Causal filter.
            
    #         #Rearranges h by moving the zero-time component to the left of the array. 
    #         h = fftshift(h); 
            
    #         h(nfft) = 0; %Zero-pad h to make its length equals to nfft.
            
    #     else %Non-causal filter.

    #         Nh = length(h); # Compute the kernel length.
            
    #         h1 = h(1:Nh/2); %Extract the first half of the kernel.
    #         h2 = h(Nh/2+1:end); %Extract the last half of the kernel.
            
    #         # Apply the zero-padding to the first half of the kernel.
    #         h1(nfft/2) = 0;
            
    #         # Apply the zero-padding to the last half of the kernel.
    #         h2 = flipud(h2); 
    #         h2(nfft/2) = 0;
    #         h2 = flipud(h2); 
            
    #         # Reconstruct the zero-padded filter kernel.
    #         h = [h1; h2];
            
    #     end
        
    #     # Apply the zero-padding on the input signal.        
    #     signal(nfft,:) = 0; %Zero-pad signal to make its length equals to nfft.

    # ---------------------------------------------------------------------      

    #H = np.fft.fft(h,nfft,0) # Compute the fft of the filter impulse response h
    FFT = np.fft.fft(signal,nfft,0) # Compute the fft of the signal

    # Reshape "H" to get dimensions to match those of "signal" -----------------
    # Thus, in matrix "HH" the column "H" is replicated over the channels ("Nch" times).
    HH = np.transpose(np.kron(H,np.ones((Nch,1))))

    # --------------------------------------------------------------------------

    # Filtering in frequency domain --------------------------------------------

    # Apply the window in the frequency domain.
    filteredFFT = np.multiply(FFT,HH)

    # --------------------------------------------------------------------------

    # Recover the filtered signal ----------------------------------------------
    filteredSignal = np.fft.ifft(filteredFFT,nfft,0)

    # --------------------------------------------------------------------------

    # In the case of non-causal linear filtering, recover the original signal
    # length by removing the zero-padding --------------------------------------

    # Frequency response of nfft samples. 
    # H1 = fft(filteredSignal);


    # --------------------------------------------------------------------------

    if (not FDFcfg['causal']) and (nfft >= Ns):
        filteredSignal = filteredSignal[0:Ns,:]

    # We have verified that the first Ns = length(signal) samples corresponds to
    # the non-causal linear convolution. See the linear convolution with a Kronecker's delta
    # in the script "conv_circular_vs_linear.m".
    # Refs:
    # matlab_functions/filtering/Frequency_Domain_Filtering/misNotas/
    # FDFnotes.docx
    # conv_circular_vs_linear.m

    # Frequency response of Ns samples. 
    # H2 = fft(filteredSignal);

    # IMPORTANT:
    # We have eliminated the discontinuity in "filteredSignal" introduced by the
    # zero-padding before the computation of H2. As a consequence,
    # H2 show less oscillations in the pass-band (Gibbs effect) that H1.

    #filteredSignal = np.squeeze(filteredSignal)
    return filteredSignal, indSettling, FilterMag, fmag

def function_eegfilt(signal, FDFcfg, fs=1):
    return 0,0,0,0

def function_butterBPF(signal, FDFcfg, fs=1):
    return 0,0,0,0

FILTERS_SWITCHER = {'function_FDF': function_FDF,
                   'function_eegfilt':function_eegfilt,
                    'function_butterBPF':function_butterBPF} 

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

def function_checkFilter(BPFcfg,fs,Ns=1,plotFlag=0):

    """
    Description:
    In this function we check the band-pass filters before applying them.
    We simply plot the magnitude of the filters' frequency response and,
    return the settling time of the filter.

    Inputs:
    - BPFcfg: Structure. 
              Band-Pass Filter configuration for the comodulogram's "x(y)" axis.
              - 'function': String {'function_butterBPF', function_eegfilt, function_FDF'} 
                            It specifies the function for the Band-Pass Filter:
                            * 'function_butterBPF', a BPF IIR filter is implemented using a series connection of a
                            High-Pass followed by a Low-Pass Butterworth filters.
                            * 'function_eegfilt', a FIR filter is implemented using the "eegfilt.m" function from EEGLAB toolbox.
                            * 'function_FDF', a Frequency Domain Filtering is implemented using a window function. 
    - fs: Scalar. Sampling rate [Hz].
    - Ns: Scalar. Number of samples of the signal.
    - plotFlag: Boolean {0,1}. Flag to plot the magnitude of the filters' frequency response.
        * 0: Do not plot.
        * 1: Does plot.

    Outputs:
    - indSettling: Int value. Indices corresponding to the transient response of the BPFs.
    - A plot corresponding to the Magnitude of the frequency response of the filters.

    Refs:
    Mike X. Cohen, Analyzing Neural Time Series Data, Theory and Practice, MIT press, 2014, p186

    """

    # Argument completion ------------------------------------------------
    if 'f1' in BPFcfg.keys() and 'f2' in BPFcfg.keys(): #Compute the cutoff frequencies.
        BPFcfg['f0'] = (BPFcfg['f1'] + BPFcfg['f2']) / 2  # Arithmetic mean.
        BPFcfg['Bw'] = BPFcfg['f2'] - BPFcfg['f1']
    # --------------------------------------------------------------------

    # Default values of the outputs --------------------------------------
    Nf  = np.size(BPFcfg['f0']) # Number of frequencies.
    NBw = np.size(BPFcfg['Bw']) # Number of Bandwidths.
    fnyq = fs/2 # [Hz] - Nyquist frequency.
    # --------------------------------------------------------------------

    # Plot configuration (No implementado) -------------------------------
    if plotFlag:
        print('No se ha implementado')
    # --------------------------------------------------------------------

    # Compute the settling time (percLevel) ------------------------------    
    indSettling = np.zeros((Nf,NBw)) # Memory pre-allocation.

    for ii in range(NBw): #Loop for Bandwidths.
        for jj in range(Nf): # Loop for frequencies.
    
            #Extract the parameters for the BPF configuration (compatible with the parfor).
            BPFcfg_local = BPFcfg
            BPFcfg_local['Bw'] = np.atleast_1d(BPFcfg['Bw'])[ii]
            BPFcfg_local['f0'] = np.atleast_1d(BPFcfg['f0'])[jj]
                                   
            if (BPFcfg_local['f0']-BPFcfg_local['Bw']/2)<=0 or (BPFcfg_local['f0']+BPFcfg_local['Bw']/2)/fnyq>=1:
                continue
            # Ref: Lega 2014 PAC in human hippocampus.pdf

            filter_function = FILTERS_SWITCHER.get(BPFcfg_local['function'], lambda: "Invalid method") # Switch for filter selection.
            _, indSettling[jj,ii], BPFmag, f = filter_function(np.full((Ns,1),np.nan), BPFcfg_local, fs)

            ## VER: Para el caso butterBPF, se ejecutaba esto antes de filter_function
            # case 'function_butterBPF', # Band-Pass Filter (IIR) using a series connection of a High-Pass followed by a Low-Pass Butterworth filters.
            #     if length(BPFcfg_local.times)>1, %Adaptive number of BPFs connected in series.
            #         BPFcfg_local.times = BPFcfg.times(jj);


    return indSettling