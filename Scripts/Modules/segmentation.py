import numpy as np

def function_segmentation(segmentWidth, overlapPercent, Ns, fs):

	"""
	Description:
	In this function we compute the indices for the sliding window (segments)
	corresponding to a time series of length Ns (number of samples).

	Inputs:
		- segmentWidth: Numeric value. Width of the segment [sec].
		- overlapPercent: Numeric value [0-100]. Percentual overlaping between successive segments.
		- Ns: Integer value. Number of samples of the time series.
		- fs: Numeric value. Sampling rate [Hz]

	Outputs:
		- indSegments: Integer array (Number segments x 2) 
					   Indices corresponding to the segments.

					   indSegments_ij = ind(j)_segment(i)
	"""

	# %% Argument completion.
	# if (nargin < 4)||isempty(fs)||isempty(Ns)||isempty(overlapPercent)||isempty(segmentWidth),
	#    error('MATLAB:function_segmentation','Input argument error.');
	# end

	# %% Check the input arguments.
	# %assert(Ns >= fix(segmentWidth*fs), 'Input argument error in function "function_segmentation": Ns < segmentLength*fs.');

	if np.fix(segmentWidth*fs) >= Ns:
	    indSegments = np.array([[0,Ns]])
	    return indSegments

	## Default values of the outputs.

	## Parameters.

	## Compute the indices.           
	# Compute the segment length.
	segmentLength = np.fix(segmentWidth*fs)

	# Compute the number of overlaped samples.
	overlapedSamples = np.round(segmentLength*overlapPercent/100)

	# Number of overlaped segments.
	numberOfSegments = np.fix((Ns - overlapedSamples)/(segmentLength - overlapedSamples))

	# Memory pre-allocation for speed up the loop.
	indSegments = np.zeros((numberOfSegments,2))
	ind2 = overlapedSamples

	for ii in range(numberOfSegments): #  %Loop across the segments.
		ind1 = ind2 - overlapedSamples
		ind2 = ind1 + segmentLength
		indSegments[ii,:] = [ind1,ind2] 
	
	return indSegments