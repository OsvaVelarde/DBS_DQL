# -*- coding: utf-8 -*-
"""
Created on September 2019
@authors: Osvaldo M Velarde - Damián Dellavale - Javier Velez
@title: Module - "comodulogram"
"""

import numpy as np

from sklearn.preprocessing import scale
from scipy.signal import hilbert

import filtering

def function_setCFCcfg(CFCin):

	"""
	Description:
	In this function we compute the structures for the "x" and "y" axis of the comodulogram.

	Inputs:
	- CFCin: Structure. Parameters of the comodulogram.
				- 'fXmin': Numeric value. Minimum frequency for the LF band [Hz].
				- 'fXmax': Numeric value. Maximum frequency for the LF band [Hz].
				- 'fYmin': Numeric value. Minimum frequency for the HF band [Hz].
				- 'fYmax': Numeric value. Maximum frequency for the HF band [Hz].
				- 'fXres': Numeric value. Frequency resolution for the LF band [Hz].
				- 'fYres': Numeric value. Frequency resolution for the HF band [Hz].
				- 'fXlookAt': String. 
							  Parameter of the signal observed in the range of
	                       	  frequency corresponding to the "x" axis.
				- 'fYlookAt': String.
							  Parameter of the signal observed in the range of
                        	  frequency corresponding to the "y" axis.
				- 'nX': Int value. Harmonic number for detection of fbX.n:fbY.n phase locking.
				- 'nY': Int value. Harmonic number for detection of fbX.n:fbY.n phase locking.
				- 'BPFXcfg': Structure. Band-Pass Filter configuration for the comodulogram's "x" axis.
				- 'BPFYcfg': Structure. Band-Pass Filter configuration for the comodulogram's "y" axis.
				- 'LPFcfg': Structure. Low-Pass Filter configuration to smooth the frequency time series.
                        	The LPF Filter is used to smooth the frequency time series 
							(fYlookAt = 'FREQUENCY' or 'PHASEofFREQUENCY').
				- 'saveBPFsignal': {0,1}. Flag to return the Band-Pass Filtered signals.
                             0 - Return a NaN.
                             1 - Return the filtered signals.
				- 'Nbins': Int value. 
						   Number of phase/amplitude bins used to compute the Histograms (p) of the comodulogram. 
				- 'sameNumberOfCycles': {0,1}. Flag to configure the processing mode for signal x:
                             0 - Do not truncate the signal "x" to obtain the same number of cycles.
                             1 - Process the same number of cycles of signal "x" for all "fX" frequencies.
				- 'CFCmethod': String. {'plv','mi'}
							Defines the approach to compute the Cross frequency Coupling 
							(PLV / methods to compute the MI).
				- 'verbose': Boolean {0,1}. 
							 0: no message are shown.
                         	 1: show the messages.
				- 'perMethod': String. Method by which the surrogated time series are built. Options
							* 'trialShuffling'
							* 'sampleShuffling'
                         	* 'FFTphaseShuffling'
                         	* 'cutShuffling'
				- 'Nper': Int value. Number of permutations.
						  It defines the number of surrogate histograms per
						  repetition. It is worth noting that in each repetition, "Nper" surrogate histograms of size
						  "Nbins x NfY x NfX" are stored in memory (RAM).
				- 'Nrep': Int value. Number of repetitions.
						  In each repetition a ".mat" file is written to disk,
						  containing "Nper" surrogate histograms of size "Nbins x NfY x NfX". 
						  As a consequence, the final number of surrogate histograms is "Nper x Nrep".
				- 'Pvalue': Numeric value. P-value for the statistically significant level.
				- 'corrMultComp': String {'Bonferroni', 'pixelBased'}.
								  Method to correct for multiple comparisons.
				- 'fs': Numeric value.

	Outputs:
	- CFCout: Structure. Parameters of the comodulogram.
				-'fXcfg', 'fYcfg': Structure. Parameters of the Frequency Band in "x(y)" axis.
									- 'start': 	Numeric value. Start frequency [Hz].
									- 'end':	Numeric value. End frequency [Hz].
									- 'res': 	Numeric value. Frequency resolution [Hz].
												Define the frequency separation between two consecutive BPFs.
									- 'BPFcfg': Structure. 
												Band-Pass Filter configuration for the comodulogram's "x(y)" axis.
									- 'lookAt': String. Parameter of the signal (phase/amplitude) observed in the range of
												frequency corresponding to the "x(y)" axis [none] (string).
									- 'n':      Int value. Harmonic number for detection of fbX.n:fbY.n phase locking.
												Ref: Detection of n,m Phase Locking from Noisy Data (Tass, 1998).pdf 
									- 'LPFcfg'  Structure.
												Low-Pass Filter configuration to smooth the frequency time series (structure array).
                                           		The LPF Filter is used to smooth the frequency time series (fYlookAt = 'FREQUENCY'
                                           		or 'PHASEofFREQUENCY').
									- 'saveBPFsignal': Boolean. Flag to return the Band-Pass Filtered signals.
												0: Return a NaN.
												1: Return the filtered signals.
									- 'Nbins': Int value. Number of phase/amplitude bins used to compute the Histograms 
												(p) of the comodulogram. 
                               		- 'sameNumberOfCycles': Boolean. Flag to configure the processing mode for signal x.
                                                       0: Do not truncate the signal "x" to obtain the same number of cycles.
                                                       1: Process the same number of cycles of signal "x" for all "fX" frequencies.
				- 'CFCmethod'
				- 'verbose'
				- 'perMethod'
				- 'Nper'
				- 'Nrep'
				- 'Pvalue'
				- 'corrMultComp'
				- 'fs'
	"""

	# Default values of the outputs --------------------------------------------------
	fXcfg = {'start': CFCin['fXmin'], 'end': CFCin['fXmax'], 'res': CFCin['fXres'],
		 	 'BPFcfg': CFCin['BPFXcfg'], 'lookAt': CFCin['fXlookAt'],
		 	 'n': CFCin['nX'], 'Nbins': CFCin['Nbins'],
		 	 'sameNumberOfCycles': CFCin['sameNumberOfCycles'],
		 	 'saveBPFsignal': CFCin['saveBPFsignal']}

	fYcfg = {'start': CFCin['fYmin'], 'end': CFCin['fYmax'], 'res': CFCin['fYres'],
			 'BPFcfg': CFCin['BPFYcfg'], 'lookAt': CFCin['fYlookAt'],
		 	 'n': CFCin['nY'],
		 	 'saveBPFsignal': CFCin['saveBPFsignal']}

	if fYcfg['lookAt'].lower == 'frequency' or fYcfg['lookAt'].lower == 'phaseoffrequency':
		fYcfg['LPFcfg'] = CFCin['LPFcfg']
	# --------------------------------------------------------------------------------

	# Compute the start frequency for "x" axis taking into account the bandwidth of the band-pass filter.
	if CFCin['fXmin'] <= CFCin['BPFXcfg']['Bw']/2:
		fXcfg['start'] = CFCin['fXmin'] + CFCin['BPFXcfg']['Bw']/2

	# Compute the vector of frequency for the "x" axis ------------------------------- 
	fXcfg['BPFcfg']['f0'] =  np.linspace(fXcfg['start'],fXcfg['end'],np.ceil((fXcfg['end']-fXcfg['start'])/fXcfg['res']))
	#np.arange(fXcfg['start'],fXcfg['end']+fXcfg['res'],fXcfg['res'])

	# Compute the adaptive number of BPFs connected in series ------------------------
	if 'times' in fXcfg['BPFcfg'].keys() and len(fXcfg['BPFcfg']['times'])>1:
		fXcfg['BPFcfg']['times'] = np.linspace(fXcfg['BPFcfg']['times'][0],fXcfg['BPFcfg']['times'][-1],len(fXcfg['BPFcfg']['times']))

	# Compute the bandwidth for the BPFs in the "y" axis ----------------------------- 
	if type(fYcfg['BPFcfg']['Bw']*1.0) == float: 	#Constant bandwidth			
		fYcfg['BPFcfg']['Bw'] = fYcfg['BPFcfg']['Bw']*np.ones(np.shape(fXcfg['BPFcfg']['f0']))
	else:											# Adaptive
		fYcfg['BPFcfg']['Bw'] = 2*fXcfg['BPFcfg']['f0']

	# Compute the start frequency for "y" axis taking into account the bandwidth of the band-pass filter.
	if fYcfg['start'] <= fYcfg['BPFcfg']['Bw'][0]/2:
		fYcfg['start'] = fYcfg['start'] + fYcfg['BPFcfg']['Bw'][0]/2

	# Compute the vector of frequency for the "y" axis --------------------------------
	fYcfg['BPFcfg']['f0'] =  np.linspace(fYcfg['start'],fYcfg['end'],np.ceil((fYcfg['end']-fYcfg['start'])/fYcfg['res'])) 
	#fYcfg['BPFcfg']['f0'] =  np.arange(fYcfg['start'],fYcfg['end']+fYcfg['res'],fYcfg['res']) 

	# Compute the adaptive number of BPFs connected in series -------------------------
	if 'times' in fYcfg['BPFcfg'].keys() and len(fYcfg['BPFcfg']['times'])>1:
		fYcfg['BPFcfg']['times'] = np.linspace(fYcfg['BPFcfg']['times'][0],fYcfg['BPFcfg']['times'][-1],len(fYcfg['BPFcfg']['times']))

	# Compute the output structure ----------------------------------------------------
	CFCout =   {'fXcfg': fXcfg, 'fYcfg': fYcfg,
	  		  	'CFCmethod': CFCin['CFCmethod'],
				'verbose': CFCin['verbose'], 'perMethod': CFCin['perMethod'],
				'Nper': CFCin['Nper'], 'Nrep': CFCin['Nrep'],
				'Pvalue': CFCin['Pvalue'], 'corrMultComp': CFCin['corrMultComp'],
				'fs': CFCin['fs']}

	return CFCout

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

FILTERS_SWITCHER = {'function_FDF': filtering.function_FDF,
                    'function_eegfilt':filtering.function_eegfilt,
                    'function_butterBPF':filtering.function_butterBPF} 

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def function_comodulogramBPF(signal,BPFcfg,fs,indSettlingExt):

    """
    Description:
    In this function we implement the Band-Pass Filtering of the input signal.
    The input signal is supposed to be a raw (unfiltered) time series.


    Inputs:
    - signal: Numeric array (Nsamples x 1). Data.
    - BPFcfg: Structure. 
              Band-Pass Filter configuration for the comodulogram's "x(y)" axis.
              - 'function': string {'function_butterBPF', 'function_eegfilt', 'function_FDF'}
                            It specifies the function for the Band-Pass Filter:
                            * 'function_butterBPF', a BPF IIR filter is implemented using a series connection of a
                            High-Pass followed by a Low-Pass Butterworth filters.
                            * 'function_eegfilt', a BPF FIR filter is implemented using the "eegfilt.m" function from
                            EEGLAB toolbox.
                            * 'function_FDF', a Frequency Domain Filtering is implemented using a window function. 
    - fs: Numeric value. Sampling rate [Hz].
    - indSettlingExt: Int value. External index for cutting out the transient response of the BPFs.
                      If "indSettlingExt" is empty or NaN, the index for the longest settling time is used.

    Outputs:
    - indSettlingMax: Int value. Index corresponding to the longest transient response of the BPFs.
    - BPFsignal: Numeric array (Nsamples x Nf x NBw). Band-Pass Filtered signals.
                 where: Ns = np.shape[signal,0]. Number of samples.
                        Nf = len(fcfg['BPFcfg']['f0']). Number of frequencies.
                        NBw = len(fcfg['BPFcfg']['Bw']).  Number of Bandwidths.
    """

    # Argument completion ------------------------------------------------------
    # if (nargin < 4)||isempty(signal)...
    #                ||isempty(BPFcfg)...
    #                ||isempty(fs)...
    #                ||isempty(indSettlingExt),...
    #                error('MATLAB:function_comodulogramBPF','Input argument error.');
    # end

    if 'f1' in BPFcfg.keys() and 'f2' in BPFcfg.keys():
        # Compute the cutoff frequencies.
        BPFcfg['f0'] = (BPFcfg['f1'] + BPFcfg['f2']) / 2 # Arithmetic mean.
        # BPFcfg['f0'] = np.sqrt(BPFcfg['f1'] * BPFcfg['f2']) %Geometric mean.
        # %Ref: https://en.wikipedia.org/wiki/Center_frequency
        BPFcfg['Bw'] = BPFcfg['f2'] - BPFcfg['f1']
    #elseif ~isfield(BPFcfg, 'f0') || ~isfield(BPFcfg, 'Bw'),
    #    error('MATLAB:function_comodulogramBPF','Error in the BPF configuration (BPFcfg).');

    # --------------------------------------------------------------------------

    # Check the input arguments ------------------------------------------------
    #assert(size(signal,2)==1, 'Input argument error in function "function_comodulogramBPF": The signal must be a column array.');
    #assert(isstruct(BPFcfg), 'Input argument error in function "function_comodulogramBPF": BPFcfg must be a structure array.');
    #assert(isnumeric(indSettlingExt)&&(indSettlingExt>0)&&(length(indSettlingExt)==1),...
    #       'Input argument error in function "function_comodulogramBPFandFeature": The value for "indSettlingExt" is not valid.');
    # --------------------------------------------------------------------------

    # Default values of the outputs --------------------------------------------
    Nf  = np.size(BPFcfg['f0']) # Number of frequencies.
    NBw = np.size(BPFcfg['Bw']) # Number of Bandwidths.
    fnyq = fs/2 # [Hz] Nyquist frequency.
    Ncycle = np.round(fs / np.atleast_1d(BPFcfg['f0'])[0]) # Compute the samples per period for the minimum frequency.   
    Ns = np.shape(signal)[0]    # Compute the number of samples of the input signal.
    Ns_cropped = Ns - 2*(indSettlingExt-1) # Compute the final length of the time series after clipping.

    # if Ncycle >= Ns_cropped:
    #         error('MATLAB:function_comodulogramBPF',...
    #           'The time series is too short: it does not include at least one period of the minimum frequency.')

    # --------------------------------------------------------------------------

    # Initializes the index corresponding to the maximum settling time with the external value.
    indSettlingMax = indSettlingExt
    # --------------------------------------------------------------------------

    ## Band-Pass Filtering -----------------------------------------------------
    
    BPFsignal = np.zeros((Ns_cropped, Nf, NBw)) # Memory pre-allocation.

    for ii in range(NBw): # Loop for Bandwidths.
        BPFsignal_local = np.zeros((Ns, Nf)) # Memory pre-allocation.
        indSettling = np.zeros((1, Nf)) # Memory pre-allocation.

        for jj in range(Nf): # Loop for frequencies.
            BPFcfg_local = BPFcfg # Extract the parameters for the BPF configuration.
            BPFcfg_local['Bw'] = np.atleast_1d(BPFcfg['Bw'])[ii]
            BPFcfg_local['f0'] = np.atleast_1d(BPFcfg['f0'])[jj] 

            # Do not compute the cases in which,
            # 1) the lower cutoff frequency is lesser than or equal to zero.
            # 2) the higher cutoff frequency is greater than or equal to one.
            # Ref: Lega 2014 PAC in human hippocampus.pdf

            if (BPFcfg_local['f0']-BPFcfg_local['Bw']/2)<=fs/Ns or (BPFcfg_local['f0']+BPFcfg_local['Bw']/2)/fnyq>=1:
                continue
            # -------------------------------------------------------------------

            filter_function = FILTERS_SWITCHER.get(BPFcfg_local['function'], lambda: "Invalid method") # Switch for filter selection.
            BPFsignal_localjj, indSettling[jj], _ , _ = filter_function(signal, BPFcfg_local, fs)
            BPFsignal_local[:,jj] = np.real(np.squeeze(BPFsignal_localjj))
            
            # VER: Para el caso butterBPF, se ejecutaba esto antes de filter_function
            # case 'function_butterBPF', # Band-Pass Filter (IIR) using a series connection of a High-Pass followed by a Low-Pass Butterworth filters.
            #     if length(BPFcfg_local.times)>1, %Adaptive number of BPFs connected in series.
            #         BPFcfg_local.times = BPFcfg.times(jj);
       
        # -----------------------------------------------------------------------
            
        # Cut out the transient response of the BPFs -----------------------------
        indSettlingMax = max([indSettling, indSettlingMax]) # Compute the index for the largest settling time.

        if indSettlingMax > indSettlingExt:         # Compare the internal and external settling time indices.
            print('Un msj no implementado- function_comodulogramBPF_v1_225')
            #warning('MATLAB:function_comodulogramBPF',...
            # 'The transient response have not completely removed using "indSettlingExt":');
            #display(['Relative difference = ' num2str(100*(indSettlingExt-indSettlingMax)/indSettlingMax) '%.']); 

        BPFsignal_local = BPFsignal_local[indSettlingExt-1:BPFsignal_local.shape[0]-(indSettlingExt-1),:] # Cutting out the BPFs' transient response.   
        # -----------------------------------------------------------------------

        BPFsignal[:,:,ii] = BPFsignal_local

    #This is required in the case of a single Bandwidth.
    #if NBw==1:
    #    BPFsignal = np.squeeze(BPFsignal)

    return BPFsignal, indSettlingMax

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def function_feature_phase(signal):
    """ 
    Description:
    Compute the phase of the z-scored BPF signal.

    Remark:
    Before the computation of the phase signal, the time series should be
    normalized, de-trended, or mean-subtracted to have the DC-component removed.
    this ensures that phase values are not limited in range.
    
    Ref: 
    Assessing transient cross-frequency coupling in EEG data (Cohen 2008).pdf

    Angle [rad] in (-pi,pi]
    """

    return np.angle(hilbert(scale(signal),axis=0))

def function_feature_amplitude(signal):
    """ 
    Description:
    Compute the amplitude (signal envelope).
    Amplitude envelope of the signal (AM demodulation).

    """

    return np.abs(hilbert(signal,axis=0))

def function_feature_phofamp(signal):
    """ 
    Description:
    Phase of the signal's amplitude envelope.
    
    Remark:
    Before the computation of the phase signal, the time series should be
    normalized, de-trended, or mean-subtracted to have the DC-component removed;
    this ensures that phase values are not limited in range.
    
    Ref: Assessing transient cross-frequency coupling in EEG data (Cohen 2008).pdf

    """

    BPFfeature = np.abs(hilbert(signal,axis=0)) # Compute the amplitudes (signal envelope).
    BPFfeature = scale(BPFfeature) # Normalization in order to avoid phase skew.              
    BPFfeature = np.angle(hilbert(BPFfeature,axis=0)) # Compute the phase of the envelope. [rad] range:(-pi,pi]

    return BPFfeature

def function_feature_frequency(signal):
    print('Sin implementar. Devuelve 0')
    return 0

def function_feature_phoffreq(signal):
    print('Sin implementar. Devuelve 0')
    return 0

LIST_FEATURES = {'phase':function_feature_phase,
                 'amplitude':function_feature_amplitude,
                 'phaseofamplitude':function_feature_phofamp,
                 'frequency':function_feature_frequency,
                 'phaseoffrequency':function_feature_phoffreq}

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def function_comodulogramFeature(signal,fcfg,fs,indSettlingExt):

    """
    Description:
    In this function we implement the extraction of the phase/amplitude/frequency 
    time series from the input signals. The input signals are supposed to be 
    previously Band-Pass Filtered signals around the frequency bands of interest.

    Inputs:
    - signal. Numeric array (Ns x Nf x NBw)
              Band-Pass Filtered signals. Notation:
              Ns: Number of samples.
              Nf: Number of frequencies. len(fcfg['BPFcfg']['f0'])
              NBw: Number of Bandwidths. len(fcfg['BPFcfg']['Bw'])

    - fcfg. Structure. Parameters of the Frequency Band in "x(y)" axis.
            - 'start': Numeric value. Start frequency [Hz].
            - 'end': Numeric value. End frequency [Hz].
            - 'res': Numeric value. Frequency resolution [Hz].
                     Define the frequency separation between two consecutive BPFs.   
            - 'BPFcfg': Structure.
                        Band-Pass Filter configuration for the comodulogram's "x(y)" axis.
            - 'lookAt': String. Parameter of the signal (phase/amplitude/frequency) observed in the range of
                                frequency corresponding to the "x(y)" axis [none].
            - 'LPFcfg': Structure. Low-Pass Filter configuration to smooth the frequency time series.
                        The LPF Filter is used to smooth the frequency time series (fYlookAt = 'FREQUENCY' or 'PHASEofFREQUENCY').
            - 'saveBPFsignal': Boolean value. Flag to return the Band-Pass Filtered signals.
                               *0: Return a NaN.
                               *1: Return the filtered signals. 
            - 'Nbins': Integer value. 
                       Number of phase/amplitude/frequency bins used to compute the Histograms (p) of the comodulogram. 

            - 'sameNumberOfCycles': Boolean value. Flag to configure the processing mode for signal x.
                               *0: Do not truncate the signal "x" to obtain the same number of cycles.
                               *1: Process the same number of cycles of signal "x" for all "fX" frequencies.
    - fs: Numeric value. Sampling rate [Hz].
    - indSettlingExt: Integer value. 
                      External index for cutting out the transient response of the BPFs.
                      If "indSettlingExt" is empty or NaN, the index for the longest settling time is used.

    Outputs:
    - BPFfeature: Numeric array (Ns x NF x NBw)
                  Phase/amplitud/frequency time series for the "x" or "y" axis of the comodulogram

    - croppedSignal: Numeric array (Ns-2*(indSettlingExt-1) x Nf x NBw) 
                     Cropped Band-Pass Filtered signals (in the case of saveBPFsignal=1)
    """

    # %Argument completion ------------------------------------------------------
    # if (nargin < 4)||isempty(signal)...
    #                ||isempty(fcfg)...
    #                ||isempty(fs)...
    #                ||isempty(indSettlingExt),...
    #                error('MATLAB:function_comodulogramFeature','Input argument error.');
    # end

    if 'f1' in fcfg['BPFcfg'].keys() and 'f2' in fcfg['BPFcfg'].keys():
        # Compute the cutoff frequencies.
        fcfg['BPFcfg']['f0'] = (fcfg['BPFcfg']['f1'] + fcfg['BPFcfg']['f2']) / 2 # Arithmetic mean.
        #%fcfg.BPFcfg.f0 = sqrt(fcfg.BPFcfg.f1 * fcfg.BPFcfg.f2); %Geometric mean.
        #%Ref: https://en.wikipedia.org/wiki/Center_frequency
        fcfg['BPFcfg']['Bw'] = fcfg['BPFcfg']['f2'] - fcfg['BPFcfg']['f1']
    #elif ~isfield(fcfg.BPFcfg, 'f0') || ~isfield(fcfg.BPFcfg, 'Bw'),
    #    error('MATLAB:function_comodulogramFeature','Error in the BPF configuration (BPFcfg).');

    # Check the input arguments ------------------------------------------------
    # assert(max(size(signal))==size(signal,1), 'Input argument error in function "function_comodulogramFeature": The signal must be a column array.');
    # assert(isstruct(fcfg), 'Input argument error in function "function_comodulogramFeature": fcfg must be a structure array.');
    # assert(isstruct(fcfg.BPFcfg), 'Input argument error in function "function_comodulogramFeature": BPFcfg structure not found.');
    # assert(isnumeric(indSettlingExt)&&(indSettlingExt>0)&&(length(indSettlingExt)==1),...
    #        'Input argument error in function "function_comodulogramBPFandFeature": The value for "indSettlingExt" is not valid.');


    # Default values of the outputs --------------------------------------------
    croppedSignal = []
    Nf = np.size(fcfg['BPFcfg']['f0']) # Number of frequencies.
    NBw = np.size(fcfg['BPFcfg']['Bw']) # Number of Bandwidths.
    fnyq = fs/2  # [Hz] Nyquist frequency.
    Ns = np.shape(signal)[0] # Compute the number of samples of the input signal.
    Ns_cropped = Ns - 2*(indSettlingExt-1) # Compute the final length of the time series after clipping.
    # --------------------------------------------------------------------------
    
    # Feature extraction -------------------------------------------------------
    BPFfeature = np.zeros((Ns_cropped, Nf, NBw)) # Memory pre-allocation for speed up the loop.

    if fcfg['saveBPFsignal']:
        croppedSignal = np.zeros((Ns_cropped, Nf, NBw))

    for ii in range(NBw): # Loop for Bandwidths.

        signal_local = signal[:,:,ii]
        
        # Selection and computation of features --------------------------------
        feature = fcfg['lookAt'].lower()
        function_feature = LIST_FEATURES.get(feature, lambda: "Invalid method")
        BPFfeature_local= function_feature(signal_local)

        # ----------------------------------------------------------------------

        BPFfeature_local = BPFfeature_local[indSettlingExt-1:BPFfeature_local.shape[0]-(indSettlingExt-1),:] # We remove the transient due to the Hilbert transform. 
        BPFfeature[:,:,ii] = BPFfeature_local

        if fcfg['saveBPFsignal']:
            # Cutting out the transient response AFTER the phase/amplitude/frequency extraction.
            croppedSignal[:,:,ii] = signal_local[indSettlingExt-1:signal_local.shape[0]-(indSettlingExt-1),:] 

        # ----------------------------------------------------------------------
    # This is required in the case of a single Bandwidth. (VER)
    # if NBw==1:
    #    BPFfeature = np.squeeze(BPFfeature)
    #    croppedSignal = np.squeeze(croppedSignal)

    return BPFfeature, croppedSignal

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def function_PLV(x,y, wx, wy, CFCcfg):
	"""
	Description: 
	In this function we compute the Phase Locking Values.

	Refs:
	[1] /PhaseLockingValue/function_PhaseLockingValue_v1.m
	[2] Measuring Phase-Amplitude Coupling Between Neuronal Oscillations (Tort, 2010).pdf, p. 1198
	[3] High gamma power is phase-locked to theta oscillations (Canolty, 2006).pdf
	[4] Phase Locking from Noisy Data (Tass, 1998).pdf
       Ref: Detection of n,m Phase Locking from Noisy Data (Tass, 1998).pdf 

	Inputs:
		- x:  Numeric array (Nsamples x NfX).
		 	  Data for the comodulogram's "x" axis (matrix: samples x NfX).
		- y:  Numeric array (Nsamples x NfY x NfX). 
		      Data for the comodulogram's "y" axis (matrix: samples x NfY x NfX).
		- wx: Numeric array (Nsamples x NfX). 
			  Weights related to the comodulogram's "x" axis (matrix: samples x NfX).
		- wy: Numeric array (Nsamples x NfY x NfX).
		      Weights related to the comodulogram's "y" axis (matrix: samples x NfY x NfX).
		- CFCcfc: structure. 
				  Parameters of the comodulogram (structure array)
				  - 'fXcfg': structure.
				  			 Parameters of the Frequency Band in "x" axis.
				  - 'fYcfg': structure. 
				  			 Parameters of the Frequency Band in "y" axis.
				  			-'start':  Numeric value. Start frequency [Hz].
				  			-'end':    Numeric value. End frequency [Hz].
				  			-'res':    Numeric value. Frequency resolution.
				  					   Define the frequency separation between two consecutive BPFs.
				  			-'BPFcfg': Structure. Band-Pass Filter configuration for the comodulogram's axis.
							-'lookAt': String.
									   Parameter of the signal (phase/amplitude) observed in the range of
									   frequency [none] (string).
							-'n': Int value. Harmonic number for detection of phase locking.
							-'saveBPFsignal': {0,1}. Flag to return the Band-Pass Filtered signals. 
											  0 - Return a NaN.
                                              1 - Return the filtered signals.
                            -'Nbins': Int value.
                            		  Number of phase/amplitude bins used to compute the Histograms (p) of the comodulogram. 
                            -'sameNumberOfCycles': {0,1}. Flag to configure the processing mode for signal x.
                            					   0 - Do not truncate the signal "x" to obtain the same number 
                            					   	   of cycles.
                                                   1 - Process the same number of cycles of signal "x" for all 
													   "fX" frequencies.
				  - 'CFCmethod': String.
				  				 Defines the approach to compute the Cross frequency Coupling. E.g: 'plv'.
				  - 'verbose': Boolean. Display flag. 
				  - 'perMethod': {'trialShuffling', 'sampleShuffling', 'FFTphaseShuffling', 'cutShuffling'}. 
				  				 Method by which the surrogated time series are built.
				  - 'Nper': Int value. 
				  		    Number of permutations. It defines the number of surrogate histograms per
                            repetition. It is worth noting that in each repetition, "Nper" surrogate 
							histograms of size "Nbins x NfY x NfX" are stored in memory (RAM).
				  - 'Nrep': Int value.
				            Number of repetitions. In each repetition a ".mat" file is written to disk,
	                        containing "Nper" surrogate histograms of size "Nbins x NfY x NfX". 
                          	As a consequence, the final number of surrogate histograms is "Nper x Nrep".
				  - 'Pvalue': Numeric value. P-value for the statistically significant level.
				  - 'corrMultComp': {'Bonferroni', 'pixelBased'} Method to correct for multiple comparisons.
				  - 'fs': Numeric value. Sampling rate [Hz].

	Outputs:
		- PLV:   Numeric array (NfY x NfX).
			     Phase Locking Value.
		- wxPLV: Numeric array (NfY x NfX).
				 Weighted Phase Locking Values using the wx weights (matrix: NfY x NfX).
		- wyPLV: Numeric array (NfY x NfX). 
				 Weighted Phase Locking Values using the wy weights (matrix: NfY x NfX).

	 			 NfX = length(CFCcfg['fXcfg']['BPFcfg']['f0'])
	 			 NfY = length(CFCcfg['fYcfg']['BPFcfg']['f0'])
	"""

	## Argument completion

	# if (nargin < 5)||isempty(x)||isempty(y)||isempty(CFCcfg),...
	#    error('MATLAB:function_PLV','Input argument error.');
	# end

	## Check the input arguments
	# assert(isstruct(CFCcfg), 'Input argument error in function "function_PLV": CFCcfg must be a structure array.');

	# if ~isfield(CFCcfg.fXcfg, 'n')||isempty(CFCcfg.fXcfg.n)||isnan(CFCcfg.fXcfg.n),
	#     CFCcfg.fXcfg.n = 1; %Default value.
	#     warning('MATLAB:function_PLV', ['"CFCcfg.fXcfg.n" is not specified, the default value is used: CFCcfg.fXcfg.n = ',...
	#                                     num2str(CFCcfg.fXcfg.n)]);      
	# end 

	# if ~isfield(CFCcfg.fYcfg, 'n')||isempty(CFCcfg.fYcfg.n)||isnan(CFCcfg.fYcfg.n),
	#     CFCcfg.fYcfg.n = 1; %Default value.
	#     warning('MATLAB:function_PLV', ['"CFCcfg.fYcfg.n" is not specified, the default value is used: CFCcfg.fYcfg.n = ',...
	#                                     num2str(CFCcfg.fYcfg.n)]);      
	# end 

	# assert(length(size(x))==2 &&...
	#        size(x,2)==length(CFCcfg.fXcfg.BPFcfg.f0) &&...
	#        size(x,1)==max(size(x)),...
	#        'Input argument error in function "function_PLV": Wrong shape of the input matrix "x".');

	# assert(length(size(y))<=3 &&...
	#        size(y,3)==length(CFCcfg.fXcfg.BPFcfg.f0) &&...
	#        size(y,2)==length(CFCcfg.fYcfg.BPFcfg.f0) &&...
	#        size(y,1)==max(size(y)),...
	#        'Input argument error in function "function_PLV": Wrong shape of the input matrix "y".');

	# if ~isempty(wx),
	#     assert(isequal(size(wx),size(x)),...
	#            'Input argument error in function "function_PLV": Wrong shape of the input matrix "wx".');
	# end

	# if ~isempty(wy),
	#     assert(isequal(size(wy),size(y)),...
	#            'Input argument error in function "function_PLV": Wrong shape of the input matrix "wy".');
	# end

	# Default values of the outputs ----------------------------------
	wxPLV = []
	wyPLV = []
	# ----------------------------------------------------------------

	# Parameters -----------------------------------------------------
	NfX = np.size(CFCcfg['fXcfg']['BPFcfg']['f0']) # Compute the length of the frequency vectors.
	NfY = np.size(CFCcfg['fYcfg']['BPFcfg']['f0']) # Compute the length of the frequency vectors.
	Ns = np.shape(x)[0]	# Number of samples
	nX = CFCcfg['fXcfg']['n'] # Compute the harmonic number for detection of nX:nY phase locking
	nY = CFCcfg['fYcfg']['n'] # Compute the harmonic number for detection of nX:nY phase locking
	# ----------------------------------------------------------------

	# Compute the modulation index "PLV" ---------------------------------------
	PLV = np.zeros((NfY,NfX),dtype=complex) # Memory pre-allocation for speed up the loop. 

	for ii in range(NfY): # Loop across the "y" frequencies.
		PLV[ii,:] = np.sum(np.exp(1j * (nX*x - nY*y[:,ii,:])),0) / Ns
	# ---------------------------------------------------------------------------

	# # Compute the modulation index "wxPLV" -------------------------------------
	# if ~isempty(wx):
	# 	wxPLV = np.zeros((NfY,NfX)) # Memory pre-allocation for speed up the loop.
	# 	for ii in range(NfY): # Loop across the "y" frequencies.
	# 		wxPLV(ii,:) = sum(wx.*exp(1j*(nX*x-nY*squeeze(y(:,ii,:)))),1) / Ns;
	# # -------------------------------------------------------------------------

	# # Compute the modulation index "wyPLV" -------------------------------------
	# if ~isempty(wy):
	# 	wyPLV = np.zeros((NfY,NfX)) # Memory pre-allocation for speed up the loop.
	# 	for ii in range(NfY): # Loop across the "y" frequencies.
	# 		wyPLV(ii,:) = sum(squeeze(wy(:,ii,:)).*exp(1j*(nX*x-nY*squeeze(y(:,ii,:)))),1) / Ns;
	# # -------------------------------------------------------------------------

	return PLV, wxPLV, wyPLV

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------