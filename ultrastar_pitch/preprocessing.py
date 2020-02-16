#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          preprocessing.py
@brief         turn an audio block of arbitrary length into
               a transformed output of unified length
@author        paradigm
"""

import numpy as np

def zero_pad_array(array, new_size):
    """ pads signal array with zeros to the desired size\n
    @param array     signal data which has to be extended\n
    @param new_size  desired new array size\n
    @return  padded array
    """
    return np.pad(array, (0, (new_size - len(array))), 'constant')

class PCA:
    """ uses principal component analysis to reduce input features and increase variance\n
    @param pca_mean        mean value vector fom a pca model\n
    @param pca_components  component matrix fom a pca model\n
    @note  the parameter can be gained by training a sklearn pca model and retrieving
           the "mean_" and "components_" member variable from the model
           this approach was choosen to avoid a dependency on the sklearn and pickle packages
    """
    def __init__(self, pca_mean, pca_components):
        self.__mean = np.load(pca_mean)
        self.__comp = np.load(pca_components)

    def transform(self, features):
        """ apply pca by means of matrix calculation\n
        @param   features  input data for pca
        @return  reduced feature set
        """
        return np.dot((features - self.__mean), self.__comp.T)

class AverageFourier:
    """ calculate the averaged right half of the signals power spectrum and normalizes it\n
    @param sample_rate  sampling frequency of the input data\n
    @param fft_len      fft window length\n
    @param adv_len      advance length for the averaging process\n
    @param fg_l         lowest cutoff-freqency set lower frequencies to zero\n
    @param noise_th     reduce noise floor by setting magnitudes below this threshold to zero
    """
    def __init__(self, sample_rate=16000, fft_len=2048, adv_len=512, fg_l=0, noise_th=0.0):
        err_str1 = "{0} has to be an positive integer value"
        err_str2 = "{0} has to be an positive float value"
        assert (isinstance(sample_rate, int) and sample_rate > 0), err_str1.format("sample_rate")
        assert (isinstance(fft_len, int) and fft_len > 0), err_str1.format("fft_len")
        assert (isinstance(adv_len, int) and adv_len > 0), err_str1.format("adv_len")
        assert (isinstance(fg_l, int) and fg_l >= 0), err_str1.format("fg_l")
        assert (isinstance(noise_th, float) and fg_l >= 0), err_str2.format("noise_th")
        self.__adv_len = adv_len
        self.__fft_len = fft_len
        self.__noise_th = noise_th
        # calculate the first sample based an the lowest cutoff-freqency
        self.__sample_l = int(np.floor(fg_l * (fft_len / sample_rate)))
        # calculate hanning window for short-term fourier transform
        self.__fft_win = np.hanning(fft_len)

    def transform_audio_segment(self, segment):
        """ turn audio segment into an averaged fft\n
        @param segment  an audio segment of arbitrary length\n
        @return  averaged fft
        """
        avg_fft = 0
        if len(segment) < self.__fft_len:
            avg_steps = 1
        else:
            # I don't feel save mergin the divisions due to possible rounding errors
            avg_steps = len(segment) // self.__adv_len - self.__fft_len // self.__adv_len + 2
        for i in range(avg_steps):
            # sliding indexes for fft window
            idx_0 = i * self.__adv_len
            idx_1 = self.__fft_len + idx_0
            if i == avg_steps - 1:
                # last frame needs to be zero padded
                frame = zero_pad_array(segment[idx_0:] * np.hanning(len(segment) - idx_0),
                                       self.__fft_len)
            else:
                # multiply segment slice by window function to reduce artifacts
                frame = segment[idx_0:idx_1] * self.__fft_win
            # average fft and cut off low frequencies based on fg_l
            avg_fft += abs(np.fft.rfft(frame))
        # set frequencies lower fg_l to zero
        avg_fft[:self.__sample_l] = 0
        # scaling the data between 0 and 1. ptp() could introduce a dividing by zero exception, if avg_fft is a zero array
        # in this case somethink else went wrong beforehand!
        # old approach: avg_fft = MinMaxScaler().fit_transform(avg_fft.reshape(-1, 1))
        avg_fft -= avg_fft.min()
        avg_fft /= avg_fft.ptp()
        # reduce noise floor by static threshold level
        avg_fft[avg_fft < self.__noise_th] = 0
        return avg_fft

class AverageWavelet:
    """ calculate averaged wavelet transform of the signal """
    def __init__(self):
        # tbd!
        pass
