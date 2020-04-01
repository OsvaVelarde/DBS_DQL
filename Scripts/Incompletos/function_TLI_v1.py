"""%Laboratorio de Bajas Temperaturas - Centro At�mico Bariloche.
%Author: Dami�n Dellavale.
%Date: 15/08/2017

%Description:
%In this script the Time Locked Index (TLI) is computed.
%The zero-crossings of the LF-phase time series are used
%to find the indices for the peaks of LF and HF time series.

%Tree of dependencies:
%function_TimeLockedIndex_v1.m
% function_FDF_v1.m
% function_eegfilt_v3.m
%  eegfilt.m
% function_butterBPF_v1.m
% function_zscore_v1.m
% function_showData_v1.m

%Changes from previous versions:
%None.

%Refs:
%function_TimeLokedPlot_v4.m """

import numpy as np
from 

def pulseZC_peaks(s1,s2):
    return np.logical_and(np.multiply(s1,s2)<0, np.absolute(s1-s2)<np.pi)

def pulseZC_troughs(s1,s2):
    return np.logical_and(np.multiply(s1,s2)<0, np.absolute(s1-s2)>=np.pi)

LIST_PULSEZC = {'peaks':pulseZC_peaks,
                'troughs': pulseZC_troughs}

def function_TLI_v1(rawSignal, LFsignal, HFsignal, TLPcfg):

    """
    Inputs:
        - rawSignal: Column numeric array (Nsamples x 1 ). Input signal.
        - LFsignal:  Column numeric array (Nsamples x 1 ). 
                     Band-pass filtered signal using TLPcfg['BPFcfg_LF'].
        - HFsignal:  Column numeric array (Nsamples x 1 ).
                     Band-pass filtered signal using TLPcfg['BPFcfg_HF'].
        - TLPcfg:    Structure. Time Locked Plot configuration.
                    - 'BPFcfg': Structure.  Band-Pass Filter configuration (structure array).
                                - 'function': String {'function_butterBPF', 'function_eegfilt, 'function_FDF'}
                                              It specifies the function for the Band-Pass Filter.
                                              * 'function_butterBPF', a BPF IIR filter is implemented using a series connection of a
                                              High-Pass followed by a Low-Pass Butterworth filters.
                                              * 'function_eegfilt', a FIR filter is implemented using the "eegfilt.m" function from EEGLAB toolbox.
                                              * 'function_FDF', a Frequency Domain Filtering is implemented using a window function.

                    - 'abs_HFpeaks': Flag for using the absolute value in detecting the peaks of the HF signal and
                                     to compute the HF signal timed-locked averaged to the FAST oscillations peaks "HFSignaltimeLokedHFpeaks".
                    - 'abs_HFSignaltimeLokedLFpeaks': 
                                     Flag for using the absolute value to compute the HF signal timed-locked averaged to the SLOW
                                     oscillations peaks "HFSignaltimeLokedLFpeaks".
                    - 'LFphase': Phase of the LF ("peaks" or "troughs") to define the time interval interval to find the HF peaks.
                    - 'NT': Interger value. Number of LF periods around the fast oscillation peaks [none].
                    - 'fs': Numeric value. Sampling rate [Hz].
                    - 'plot': Flag to plot the signals. 0: Do not plot. 1: Does plot.

    Outputs:
        - rawSignaltimeLokedHFpeaks: Raw signal timed-locked averaged to the FAST oscillations peaks [signal] (column array: samples x 1).
        - rawSignaltimeLokedLFpeaks: Raw signal timed-locked averaged to the SLOW oscillations peaks [signal] (column array: samples x 1).
        - HFSignaltimeLokedHFpeaks:  HF signal timed-locked averaged to the FAST oscillations peaks [signal] (column array: samples x 1).
        - HFSignaltimeLokedLFpeaks:  HF signal timed-locked averaged to the SLOW oscillations peaks [signal] (column array: samples x 1).
        - LFSignaltimeLokedHFpeaks:  LF signal timed-locked averaged to the FAST oscillations peaks [signal] (column array: samples x 1).
        - LFSignaltimeLokedLFpeaks:  LF signal timed-locked averaged to the SLOW oscillations peaks [signal] (column array: samples x 1).
        - TLI:                       Time-Locked Index (1 x 1).
        - indPeak_HF:                Peak indices corresponding to the HF fast oscillations.
        - indPeak_LF:                Peak indices corresponding to the LF slow oscillations.
        - sampleT:                   Epoch length (multiple of the LF signal period).

    """


    return rawSignaltimeLokedHFpeaks, rawSignaltimeLokedLFpeaks, HFSignaltimeLokedHFpeaks, HFSignaltimeLokedLFpeaks, LFSignaltimeLokedHFpeaks, LFSignaltimeLokedLFpeaks, TLI, indPeak_HF, indPeak_LF, sampleT





%% OSVA
if (sum(isnan(rawSignal))~=0 || std(rawSignal)==0)
    rawSignaltimeLokedHFpeaks=zeros(size(rawSignal,1),1);
    rawSignaltimeLokedLFpeaks=zeros(size(rawSignal,1),1);
    HFSignaltimeLokedHFpeaks=zeros(size(rawSignal,1),1);
    HFSignaltimeLokedLFpeaks=zeros(size(rawSignal,1),1);
    LFSignaltimeLokedHFpeaks=zeros(size(rawSignal,1),1);
    LFSignaltimeLokedLFpeaks=zeros(size(rawSignal,1),1);

    TLI=0;
    indPeak_HF=0;
    indPeak_LF=0;
    sampleT=0;

else
%%     
%Argument completion ------------------------------------------------------

#if (nargin < 4)||isempty(rawSignal)||isempty(LFsignal)||isempty(HFsignal)||isempty(TLPcfg),...
#   error('MATLAB:function_TimeLockedIndex','Input argument error.');

if 'f1' in TLPcfg['BPFcfg_LF'].keys() and 'f2' in TLPcfg['BPFcfg_LF'].keys():

    # Compute the center frequency.
    TLPcfg['BPFcfg_LF']['f0'] = ( TLPcfg['BPFcfg_LF']['f2'] + TLPcfg['BPFcfg_LF']['f1'] ) / 2  #%Arithmetic mean. 
    #%TLPcfg.BPFcfg_LF.f0 = sqrt( TLPcfg.BPFcfg_LF.f2 * TLPcfg.BPFcfg_LF.f1 ); %Geometric mean.
    #%Ref: https://en.wikipedia.org/wiki/Center_frequency
    
    #%Compute the bandwidth.
    TLPcfg['BPFcfg_LF']['Bw'] = TLPcfg['BPFcfg_LF']['f2'] - TLPcfg['BPFcfg_LF']['f1']

#elseif ~isfield(TLPcfg.BPFcfg_LF, 'f0') || ~isfield(TLPcfg.BPFcfg_LF, 'Bw'),    
#    error('MATLAB:function_TimeLockedIndex','Error in the BPF configuration (BPFcfg_LF).');
    

if 'f1' in TLPcfg['BPFcfg_HF'].keys() and 'f2' in TLPcfg['BPFcfg_HF'].keys():

    # Compute the center frequency.
    TLPcfg['BPFcfg_HF']['f0'] = ( TLPcfg['BPFcfg_HF']['f2'] + TLPcfg['BPFcfg_HF']['f1'] ) / 2  #%Arithmetic mean. 
    #%TLPcfg.BPFcfg_HF.f0 = sqrt( TLPcfg.BPFcfg_HF.f2 * TLPcfg.BPFcfg_HF.f1 ); %Geometric mean.
    #%Ref: https://en.wikipedia.org/wiki/Center_frequency
    
    #%Compute the bandwidth.
    TLPcfg['BPFcfg_HF']['Bw'] = TLPcfg['BPFcfg_HF']['f2'] - TLPcfg['BPFcfg_HF']['f1']

#elseif ~isfield(TLPcfg.BPFcfg_HF, 'f0') || ~isfield(TLPcfg.BPFcfg_HF, 'Bw'),    
#    error('MATLAB:function_TimeLockedIndex','Error in the BPF configuration (BPFcfg_HF).');

#Check the input arguments ------------------------------------------------
#assert(size(rawSignal,2)==1, 'Input argument error in function "function_TimeLockedIndex": The rawSignal must be a column array.');
#assert(size(LFsignal,2)==1, 'Input argument error in function "function_TimeLockedIndex": The LFsignal must be a column array.');
#assert(size(HFsignal,2)==1, 'Input argument error in function "function_TimeLockedIndex": The HFsignal must be a column array.');
#assert(isstruct(TLPcfg), 'Input argument error in function "function_TimeLockedIndex": TLPcfg must be a structure array.');

if not 'NT' in TLPcfg.keys():
    TLPcfg['NT']= 1

if not 'fs' in TLPcfg.keys():
    TLPcfg['fs']= 1

# Default values of the outputs --------------------------------------------

#Parameters ---------------------------------------------------------------
# Compute the number of samples for the lower cutoff frequency.
lowerFrec = TLPcfg['BPFcfg_LF']['f0'] - TLPcfg['BPFcfg_LF']['Bw']/2
sampleT = np.round( TLPcfg.NT * TLPcfg.fs / lowerFrec )
halfSampleT = np.round(sampleT/2)

# Compute the length of the raw signal.
Nsamples = np.shape(rawSignal)[0]

#IMPORTANT: The following normalization is critical in cases where HF 
#and LF signals have very different amplitudes ----------------------------  
#Normalization in order to have zero mean and unit variance (unit amplitude) in HF and LF signals.
HFsignal = sklearn.preprocessing.scale(HFsignal)
LFsignal = sklearn.preprocessing.scale(LFsignal)

# %Verify the normalizations:
# %mean(HFsignal,1)
# %mean(LFsignal,1)
# %var(HFsignal,1)
# %var(LFsignal,1)

#Add the HF and LF signals.
HF_LF_signal = HFsignal + LFsignal;

#Find the index for the peak corresponding to phase = 0 or pi rad. --------

# Compute the LF-phase time series.
LFphase = np.angle(hilbert(LFsignal)) # [rad] range: [-pi,pi]
# %NOTE: The LFphase time series is only used to identify the peaks or troughs
# %of the LF signal. As a consequence, here we can disregard the transient 
# %response due to the Hilbert transform.
# %The Hilbert transform is computed in the frequency domain in the Matlab
# %script hilbert.m via "fft" and "ifft" built-in functions. 

# Find zero-crossings in LF-phase time series.
s1 = LFphase[0:Nsamples-1]
s2 = LFphase[1:Nsamples]


LFphase_f = TLPcfg['LFphase'].lower()
function_pulseZC = LIST_PULSEZC.get(LFphase_f, lambda:"Invalid method")
pulseZC = function_pulseZC(s1,s2) 

# Re-arrange the pulses in order to emulate a causal zero-crossing detection.
pulseZC =np.concatenate(([False],pulseZC))

# Compute the indices for the peaks of the LF signal (zeroes of the LF-phase signal).
indPeak_LF = [i for i,val in enumerate(pulseZC) if val]

# % %DEBUGGING: Show the signals ---
# % function_showData_v1([LFphase/max(abs(LFphase)), LFsignal/max(abs(LFsignal))]);
# % 
# % figure, plot(LFphase,'.-b')
# % hold on, plot(pulseZC,'or')
# % 
# % figure, plot(LFsignal,'.-b')
# % hold on, plot(indPeak_LF,LFsignal(indPeak_LF),'+r','Markersize',20)
# % %---

# Compute the number of LF peaks.
Npeak = len(indPeak_LF)

# Compute the auxiliary signals according to the flags for the absolute value
if TLPcfg['abs_HFpeaks']:
    HFsignal_aux1 = abs(HFsignal)
else:
    HFsignal_aux1 = HFsignal

if TLPcfg.abs_HFSignaltimeLokedLFpeaks:
    HFsignal_aux2 = abs(HFsignal)        
else:
    HFsignal_aux2 = HFsignal

# Compute the peaks of HF --------------------------------------------------

# Memory pre-allocation for speed up the loop.
indPeak_HF = NaN(Npeak-1,1)
# HFperiod = indPeak_HF;
# N_HFcicles = indPeak_HF;
# HFpower = indPeak_HF;
HFpowerRatio = indPeak_HF

ME QUEDE AQUIIIIIIIIIIIIIIIIIIIIII!!!

for ii in range(Npeak-1): # Loop across all the "Npeaks-1" intervals between the LF peaks.        
    #Find the index for the peak corresponding to the maximum (absolute)
    #amplitude of the HF signal in each period.
    [~, indPeak_HF(ii)] = max( HFsignal_aux1(indPeak_LF(ii):indPeak_LF(ii+1)) );

    indPeak_HF(ii) = indPeak_HF(ii) + indPeak_LF(ii) - 1;
     
# Extract the epochs centered at the time points corresponding to the
# FAST and SLOW OSCILLATION PEAKS -----------------------------------

# Compute the valid indices for the HF and LF peaks.
JJstart = NaN;
JJend = NaN;
for jj=1:+1:Npeak, %Loop across all the LF peaks.
    if (indPeak_LF(jj)-halfSampleT > 0) && (indPeak_HF(jj)-halfSampleT > 0),
        JJstart = jj;
        break,
    end
end
for jj=Npeak-1:-1:1, %Skip the last LF peak: We start from the last HF peak.
    if (indPeak_LF(jj)+halfSampleT < Nsamples) && (indPeak_HF(jj)+halfSampleT < Nsamples),
        JJend = jj;
        break,
    end
end
if isnan(JJstart)||isnan(JJend),
    error('MATLAB:function_TimeLockedIndex',...
          'Undefined bounds for the indices of the HF/LF peaks.');
end

# Memory pre-allocation for speed up the loop.
rawSignaltimeLokedHFpeaks = zeros(2*halfSampleT+1,1);
rawSignaltimeLokedLFpeaks = rawSignaltimeLokedHFpeaks;
HFSignaltimeLokedHFpeaks = rawSignaltimeLokedHFpeaks;
HFSignaltimeLokedLFpeaks = rawSignaltimeLokedHFpeaks;
LFSignaltimeLokedHFpeaks = rawSignaltimeLokedHFpeaks;
LFSignaltimeLokedLFpeaks = rawSignaltimeLokedHFpeaks;

# Using the rawSignal.
for jj=JJstart:+1:JJend, %It probably skip some initial and final peak indices.
    rawSignaltimeLokedHFpeaks = rawSignaltimeLokedHFpeaks + rawSignal(indPeak_HF(jj)-halfSampleT:indPeak_HF(jj)+halfSampleT);
    rawSignaltimeLokedLFpeaks = rawSignaltimeLokedLFpeaks + rawSignal(indPeak_LF(jj)-halfSampleT:indPeak_LF(jj)+halfSampleT);
    
    HFSignaltimeLokedHFpeaks = HFSignaltimeLokedHFpeaks + HFsignal_aux1(indPeak_HF(jj)-halfSampleT:indPeak_HF(jj)+halfSampleT);
    HFSignaltimeLokedLFpeaks = HFSignaltimeLokedLFpeaks + HFsignal_aux2(indPeak_LF(jj)-halfSampleT:indPeak_LF(jj)+halfSampleT);
    
    LFSignaltimeLokedHFpeaks = LFSignaltimeLokedHFpeaks + LFsignal(indPeak_HF(jj)-halfSampleT:indPeak_HF(jj)+halfSampleT);
    LFSignaltimeLokedLFpeaks = LFSignaltimeLokedLFpeaks + LFsignal(indPeak_LF(jj)-halfSampleT:indPeak_LF(jj)+halfSampleT);    
end
    
# % %IMPORTANT: Using the normalized sum (HF_LF_signal) of the HF and LF 
# % %signals is critical in cases where HF and LF signals have very different amplitudes. 
# % for jj=JJstart:+1:JJend, %It probably skip some initial and final peak indices.
# %     rawSignaltimeLokedHFpeaks = rawSignaltimeLokedHFpeaks + HF_LF_signal(indPeak_HF(jj)-halfSampleT:indPeak_HF(jj)+halfSampleT);
# %     rawSignaltimeLokedLFpeaks = rawSignaltimeLokedLFpeaks + HF_LF_signal(indPeak_LF(jj)-halfSampleT:indPeak_LF(jj)+halfSampleT);
# %     
# %     HFSignaltimeLokedHFpeaks = HFSignaltimeLokedHFpeaks + HFsignal_aux1(indPeak_HF(jj)-halfSampleT:indPeak_HF(jj)+halfSampleT);
# %     HFSignaltimeLokedLFpeaks = HFSignaltimeLokedLFpeaks + HFsignal_aux2(indPeak_LF(jj)-halfSampleT:indPeak_LF(jj)+halfSampleT); 
# %     
# %     LFSignaltimeLokedHFpeaks = LFSignaltimeLokedHFpeaks + LFsignal(indPeak_HF(jj)-halfSampleT:indPeak_HF(jj)+halfSampleT);
# %     LFSignaltimeLokedLFpeaks = LFSignaltimeLokedLFpeaks + LFsignal(indPeak_LF(jj)-halfSampleT:indPeak_LF(jj)+halfSampleT);     
# % end

# Compute the average of the time-locked epochs.
rawSignaltimeLokedHFpeaks = rawSignaltimeLokedHFpeaks / (JJend-JJstart+1);
rawSignaltimeLokedLFpeaks = rawSignaltimeLokedLFpeaks / (JJend-JJstart+1);
HFSignaltimeLokedHFpeaks = HFSignaltimeLokedHFpeaks / (JJend-JJstart+1);
HFSignaltimeLokedLFpeaks = HFSignaltimeLokedLFpeaks / (JJend-JJstart+1);
LFSignaltimeLokedHFpeaks = LFSignaltimeLokedHFpeaks / (JJend-JJstart+1);
LFSignaltimeLokedLFpeaks = LFSignaltimeLokedLFpeaks / (JJend-JJstart+1);

# %% Compute the Time-Locked Index --------------------------------------------
# % %Compute the circular cross-correlations.
# % ccorr_rawSignal = cconv(rawSignaltimeLokedHFpeaks,conj(flipud(rawSignaltimeLokedLFpeaks)),2*halfSampleT+1) /...
# %                   (norm(rawSignaltimeLokedHFpeaks)*norm(conj(flipud(rawSignaltimeLokedLFpeaks))));
# %               
# % ccorr_LFsignal = cconv(LFSignaltimeLokedHFpeaks,conj(flipud(LFSignaltimeLokedLFpeaks)),2*halfSampleT+1) /...
# %                  (norm(LFSignaltimeLokedHFpeaks)*norm(conj(flipud(LFSignaltimeLokedLFpeaks))));   
# %              
# % ccorr_HFsignal = cconv(HFSignaltimeLokedHFpeaks,conj(flipud(HFSignaltimeLokedLFpeaks)),2*halfSampleT+1) /...
# %                  (norm(HFSignaltimeLokedHFpeaks)*norm(conj(flipud(HFSignaltimeLokedLFpeaks))));
# % 
# % ccorr_rawLF = cconv(rawSignaltimeLokedHFpeaks,conj(flipud(LFSignaltimeLokedLFpeaks)),2*halfSampleT+1) /...
# %               (norm(rawSignaltimeLokedHFpeaks)*norm(conj(flipud(LFSignaltimeLokedLFpeaks))));
# % 
# % figure, hold on, 
# % plot(ccorr_rawSignal,'-b')
# % plot(ccorr_LFsignal,'-r')
# % plot(ccorr_HFsignal,'-g')
# % plot(ccorr_rawLF,'-k')
# % pause

# %TLI = max(ccorr_rawSignal) - max(abs(HFSignaltimeLokedLFpeaks))/max(abs(HFSignaltimeLokedHFpeaks));
# %TLI = max(ccorr_LFsignal) - max(abs(HFSignaltimeLokedLFpeaks))/max(abs(HFSignaltimeLokedHFpeaks));
# %TLI = max(ccorr_rawLF) - max(abs(HFSignaltimeLokedLFpeaks))/max(abs(HFSignaltimeLokedHFpeaks));
# %TLI = max(ccorr_rawLF) * ( max(abs(HFSignaltimeLokedHFpeaks)) - max(abs(HFSignaltimeLokedLFpeaks)) ) / std(HFSignaltimeLokedHFpeaks,0,1);

TLI = ( max(HFSignaltimeLokedLFpeaks) - min(HFSignaltimeLokedLFpeaks) ) / ( max(HFSignaltimeLokedHFpeaks) - min(HFSignaltimeLokedHFpeaks) );
# %NOTE: This Time-Locked Index is constrained to the interval [0,1] and it is a measure of how much of the HF peak observed in "HFSignaltimeLokedHFpeaks"
# %can be explained by the harmonic content of the raw signal. It is worth noting that the HF peak observed in "HFSignaltimeLokedLFpeaks" is produced by
# %the harmonic content of the raw signal, that is, HF components almos perfectly time-locked to the LF component.

# %TLI = ( max(abs(HFSignaltimeLokedHFpeaks)) - max(abs(HFSignaltimeLokedLFpeaks)) ) / std(HFSignaltimeLokedHFpeaks,0,1);
# %TLI = ( max(abs(HFSignaltimeLokedHFpeaks)) - max(abs(HFSignaltimeLokedLFpeaks)) ) / ( max(abs(HFSignaltimeLokedHFpeaks)) + max(abs(HFSignaltimeLokedLFpeaks)) );
# %TLI = ( std(HFSignaltimeLokedHFpeaks,0,1) - std(HFSignaltimeLokedLFpeaks,0,1) ) / std(HFSignaltimeLokedHFpeaks,0,1);
# %TLI = max(abs(HFSignaltimeLokedHFpeaks)) / std(HFSignaltimeLokedHFpeaks,0,1);
# %TLI = max(abs(HFSignaltimeLokedLFpeaks)) / std(HFSignaltimeLokedLFpeaks,0,1);

if TLPcfg.plot, %Plot the signals.
    
    %Compute the circular cross-correlations.
    ccorr_rawSignal = cconv(rawSignaltimeLokedHFpeaks,conj(flipud(rawSignaltimeLokedLFpeaks)),2*halfSampleT+1) /...
                      (norm(rawSignaltimeLokedHFpeaks)*norm(conj(flipud(rawSignaltimeLokedLFpeaks))));
              
    ccorr_LFsignal = cconv(LFSignaltimeLokedHFpeaks,conj(flipud(LFSignaltimeLokedLFpeaks)),2*halfSampleT+1) /...
                     (norm(LFSignaltimeLokedHFpeaks)*norm(conj(flipud(LFSignaltimeLokedLFpeaks))));  
    
    %Compute the plot parameters ------------------------------------------
    %Get the screen size.
    scrsz = get(0,'ScreenSize'); %scrsz -> [left, bottom, width, height].

    figure_location = scrsz; %Figure in full screen mode.
    %figure_location = [scrsz(3)/4 scrsz(4)/4 scrsz(3)/2 scrsz(4)/2]; %[left, bottom, width, height]. 

    %Define the colors.
    %rgbDARKBLUE  = [0 0 0.7];
    %rgbDARKRED  = [0.7 0 0];
    %rgbLIGHTRED   = [1 0.3 0.3];
    %rgbLIGHTBLUE  = [0.3 0.3 1];
    %rgbRED   = [1 0 0];
    rgbBLUE  = [0 0 1];
    %rgbGREEN = [0 0.65 0];
    rgbGRAY  = [0.5 0.5 0.5];
    rgbBLACK = [0 0 0];
    %rgbWHITE = [1 1 1];
    %----------------------------------------------------------------------    
    
    t = (0:+1:size(LFsignal,1)-1) / TLPcfg.fs;
    
    %Plot the band-pass filtered signals ----------------------------------
    function_showData_v1(t,[HFsignal, LFsignal, rawSignal],'subplot');
    
    function_showData_v1(t,[LFsignal, HFsignal]);
    hold on, 
    for jj=JJstart:+1:JJend, %It probably skip some initial and final peak indices.
        %plot(t([jj*sampleT jj*sampleT]),[min(HFsignal) max(HFsignal)],'LineStyle',':','LineWidth',2,'Color',rgbBLACK)
        plot(t([indPeak_LF(jj)+0*halfSampleT indPeak_LF(jj)+0*halfSampleT]),[min(HFsignal) max(HFsignal)],'LineStyle',':','LineWidth',2,'Color',rgbBLACK)
        plot(t(indPeak_HF(jj)),HFsignal(indPeak_HF(jj)),'Marker','+','MarkerSize',20,'MarkerEdgeColor',rgbBLACK,'MarkerFaceColor',rgbBLACK,'LineStyle','none','LineWidth',2,'Color',rgbBLACK)
        plot(t(indPeak_LF(jj)),LFsignal(indPeak_LF(jj)),'Marker','+','MarkerSize',20,'MarkerEdgeColor',rgbGRAY,'MarkerFaceColor',rgbGRAY,'LineStyle','none','LineWidth',2,'Color',rgbGRAY)
    end
    %xlim([0 t(7*sampleT)]);

    %Compute the HF amplitude envelope.
    HFamp = abs(hilbert(HFsignal));
    
    function_showData_v1(t,[LFphase/max(abs(LFphase)), HFamp/max(abs(HFamp))]);
    hold on, 
    for jj=JJstart:+1:JJend, %It probably skip some initial and final peak indices.
        %plot(t([jj*sampleT jj*sampleT]),[min(HFsignal) max(HFsignal)],'LineStyle',':','LineWidth',2,'Color',rgbBLACK)
        plot(t([indPeak_LF(jj)+0*halfSampleT indPeak_LF(jj)+0*halfSampleT]),[-1 1],'LineStyle',':','LineWidth',2,'Color',rgbBLACK)
    end
    %xlim([0 tcropped(7*sampleT)]);
    
# %     LFamp = abs(hilbert(LFsignal));
# %     HFamp = abs(hilbert(HFsignal));
# %     function_showData_v1(t,[LFamp/max(abs(LFamp)), HFamp/max(abs(HFamp))]);
# %     hold on, 
# %     for jj=JJstart:+1:JJend, %It probably skip some initial and final peak indices.
# %         %plot(t([jj*sampleT jj*sampleT]),[min(HFsignal) max(HFsignal)],'LineStyle',':','LineWidth',2,'Color',rgbBLACK)
# %         plot(t([indPeak_LF(jj)+0*halfSampleT indPeak_LF(jj)+0*halfSampleT]),[-1 1],'LineStyle',':','LineWidth',2,'Color',rgbBLACK)
# %     end
# %     %xlim([0 tcropped(7*sampleT)]);    
    %----------------------------------------------------------------------

    %Plot the Time Locked Average (Raw signal) ----------------------------
    figure('Position',figure_location); %Create a new figure.

    tlocked = t(1:+1:size(rawSignaltimeLokedHFpeaks,1));

    H(1).mainLine = plot(tlocked,rawSignaltimeLokedLFpeaks,'Marker','none','MarkerSize',5,'MarkerEdgeColor',rgbGRAY,'MarkerFaceColor',rgbGRAY,'LineStyle',':','LineWidth',2,'Color',rgbGRAY);
    hold on,
    H(2).mainLine = plot(tlocked,rawSignaltimeLokedHFpeaks,'Marker','none','MarkerSize',5,'MarkerEdgeColor',rgbBLUE,'MarkerFaceColor',rgbBLUE,'LineStyle','-','LineWidth',2,'Color',rgbBLUE);
    
    %Generate the legends.
    legend([H(1).mainLine, H(2).mainLine],{'Raw signal time-locked to LF peaks','Raw signal time-locked to HF peaks'},'orientation','vertical','location','best')
    
    %legend('boxoff')
    legend('boxon') 

    box on, grid off,
    %axis tight;
    axis([tlocked(1) tlocked(end) min([rawSignaltimeLokedHFpeaks; rawSignaltimeLokedLFpeaks]) max([rawSignaltimeLokedHFpeaks; rawSignaltimeLokedLFpeaks])]);

    set(gca,'fontname','Times New Roman','fontsize',16','fontWeight','bold');
    set(gca,'TickDirMode','auto','TickDir','in');
    set(gca,'YMinorTick','On','XMinorTick','On','LineWidth',1);
    %set(gca,'XTick',[])
    %set(gca,'XTickLabel',{'';''})
    %set(gca,'YTick',[])
    %set(gca,'YTickLabel',{'';''})
    xlabel(strcat('$\mathbf{Time} [sec.] ; Res: $',num2str(1/TLPcfg.fs,'%g'),'$[sec.]$'),'interpreter','LaTex')
    ylabel('$\mathbf{Amplitude~[a.u.]}$','interpreter','LaTex')
    title(['Time Locked Plot.',...
           sprintf('\n'),...
           'LF = ', num2str(TLPcfg.BPFcfg_LF.f0), ' Hz (Bw = ', num2str(TLPcfg.BPFcfg_LF.Bw), ' Hz) ; ',...
           'HF = ', num2str(TLPcfg.BPFcfg_HF.f0), ' Hz (Bw = ', num2str(TLPcfg.BPFcfg_HF.Bw), ' Hz) ; ',...
           'ccorr = ', num2str(max(ccorr_rawSignal),'%g')],...  
           'interpreter','none');   
    hold off,
    %----------------------------------------------------------------------

    %Plot the Time Locked Average (HF signal) -----------------------------
    figure('Position',figure_location); %Create a new figure.

    tlocked = t(1:+1:size(HFSignaltimeLokedHFpeaks,1));

    H(1).mainLine = plot(tlocked,HFSignaltimeLokedLFpeaks,'Marker','none','MarkerSize',5,'MarkerEdgeColor',rgbGRAY,'MarkerFaceColor',rgbGRAY,'LineStyle',':','LineWidth',2,'Color',rgbGRAY);
    hold on,
    H(2).mainLine = plot(tlocked,HFSignaltimeLokedHFpeaks,'Marker','none','MarkerSize',5,'MarkerEdgeColor',rgbBLUE,'MarkerFaceColor',rgbBLUE,'LineStyle','-','LineWidth',2,'Color',rgbBLUE);
    
    %Generate the legends.
    legend([H(1).mainLine, H(2).mainLine],{'HF signal time-locked to LF peaks','HF signal time-locked to HF peaks'},'orientation','vertical','location','best')
    %legend('boxoff')
    legend('boxon') 

    box on, grid off,
    %axis tight;
    axis([tlocked(1) tlocked(end) min([HFSignaltimeLokedHFpeaks; HFSignaltimeLokedLFpeaks]) max([HFSignaltimeLokedHFpeaks; HFSignaltimeLokedLFpeaks])]);

    set(gca,'fontname','Times New Roman','fontsize',16','fontWeight','bold');
    set(gca,'TickDirMode','auto','TickDir','in');
    set(gca,'YMinorTick','On','XMinorTick','On','LineWidth',1);
    %set(gca,'XTick',[])
    %set(gca,'XTickLabel',{'';''})
    %set(gca,'YTick',[])
    %set(gca,'YTickLabel',{'';''})
    xlabel(strcat('$\mathbf{Time} [sec.] ; Res: $',num2str(1/TLPcfg.fs,'%g'),'$[sec.]$'),'interpreter','LaTex')
    ylabel('$\mathbf{Amplitude~[a.u.]}$','interpreter','LaTex')
    title(['Time Locked Plot.',...
           sprintf('\n'),...
           'LF = ', num2str(TLPcfg.BPFcfg_LF.f0), ' Hz (Bw = ', num2str(TLPcfg.BPFcfg_LF.Bw), ' Hz) ; ',...
           'HF = ', num2str(TLPcfg.BPFcfg_HF.f0), ' Hz (Bw = ', num2str(TLPcfg.BPFcfg_HF.Bw), ' Hz) ; ',...
           'TLI = ', num2str(TLI,'%g')],...
           'interpreter','none');
    hold off,
    %----------------------------------------------------------------------    
    
    %Plot the Time Locked Average (HF signal) -----------------------------
    figure('Position',figure_location); %Create a new figure.

    tlocked = t(1:+1:size(LFSignaltimeLokedHFpeaks,1));

    H(1).mainLine = plot(tlocked,LFSignaltimeLokedLFpeaks,'Marker','none','MarkerSize',5,'MarkerEdgeColor',rgbGRAY,'MarkerFaceColor',rgbGRAY,'LineStyle',':','LineWidth',2,'Color',rgbGRAY);
    hold on,
    H(2).mainLine = plot(tlocked,LFSignaltimeLokedHFpeaks,'Marker','none','MarkerSize',5,'MarkerEdgeColor',rgbBLUE,'MarkerFaceColor',rgbBLUE,'LineStyle','-','LineWidth',2,'Color',rgbBLUE);
    
    %Generate the legends.
    legend([H(1).mainLine, H(2).mainLine],{'LF signal time-locked to LF peaks','LF signal time-locked to HF peaks'},'orientation','vertical','location','best')
    %legend('boxoff')
    legend('boxon') 

    box on, grid off,
    %axis tight;
    axis([tlocked(1) tlocked(end) min([LFSignaltimeLokedHFpeaks; LFSignaltimeLokedLFpeaks]) max([LFSignaltimeLokedHFpeaks; LFSignaltimeLokedLFpeaks])]);

    set(gca,'fontname','Times New Roman','fontsize',16','fontWeight','bold');
    set(gca,'TickDirMode','auto','TickDir','in');
    set(gca,'YMinorTick','On','XMinorTick','On','LineWidth',1);
    %set(gca,'XTick',[])
    %set(gca,'XTickLabel',{'';''})
    %set(gca,'YTick',[])
    %set(gca,'YTickLabel',{'';''})
    xlabel(strcat('$\mathbf{Time} [sec.] ; Res: $',num2str(1/TLPcfg.fs,'%g'),'$[sec.]$'),'interpreter','LaTex')
    ylabel('$\mathbf{Amplitude~[a.u.]}$','interpreter','LaTex')
    title(['Time Locked Plot.',...
           sprintf('\n'),...
           'LF = ', num2str(TLPcfg.BPFcfg_LF.f0), ' Hz (Bw = ', num2str(TLPcfg.BPFcfg_LF.Bw), ' Hz) ; ',...
           'HF = ', num2str(TLPcfg.BPFcfg_HF.f0), ' Hz (Bw = ', num2str(TLPcfg.BPFcfg_HF.Bw), ' Hz) ; ',...    
           'ccorr = ', num2str(max(ccorr_LFsignal),'%g')],...      
           'interpreter','none'); 
    hold off,
    %----------------------------------------------------------------------    
    
end %Plot the signals.    

end %if Osva

end %function

