import numpy as np
import filtering

def downsampling(input_signal,fs,decFactor,LPFcfg):
    # Anti-alias Filtering              
    indSettling = np.shape(input_signal)[0] + 1 # Set the settling time.         
    signal = np.concatenate((input_signal[::-1],input_signal,input_signal[::-1])) # Reflect the time series to minimize edge artifacts due to the transient response of the BPFs.

    signal = signal.reshape(signal.shape[0],-1)

    signal, _, _, _ = filtering.function_FDF(signal, LPFcfg, fs) # Filtering in frequency domain.
    signal = np.real(signal)

    signal = signal[indSettling-1:signal.shape[0]-(indSettling-1),:] # Restore the length of the raw signal.
    signal = signal[::decFactor,:]

    signal = np.squeeze(signal)
    return signal

