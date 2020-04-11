#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          preprocessing.py
@brief         turn an audio block of arbitrary length into
               a transformed output of unified length
@author        paradigm
"""

import os
import sys
import numpy as np

class PCA:
    """ uses principal component analysis to reduce input features and increase variance """
    def __init__(self, pca_mean=None, pca_components=None):
        """ load class variables\n
        @param   pca_mean         mean value vector fom a pca model\n
        @param   pca_components   component matrix fom a pca model\n
        @note    the parameter can be gained by training a sklearn pca model and retrieving\n
                 the "mean_" and "components_" member variable from the model\n
                 this approach was choosen to avoid a dependency on the sklearn and pickle packages
        """
        if pca_mean and pca_components:
            # load parameter from external source
            self.__mean = pca_mean
            self.__comp = pca_components
        elif getattr(sys, 'frozen', False):
            # load parameter from meipass in case of binary execution
            self.__mean = np.load(os.path.join(sys._MEIPASS, "pca_mean.npy"))
            self.__comp = np.load(os.path.join(sys._MEIPASS, "pca_components.npy"))
        else:
            # load parameter from binary folder
            self.__mean = np.load(os.path.join(
                os.path.dirname(__file__), "binaries", "pca_mean.npy"))
            self.__comp = np.load(os.path.join(
                os.path.dirname(__file__), "binaries", "pca_components.npy"))

    def transform(self, features):
        """ apply pca by means of matrix calculation\n
        @param    features   input data for pca\n
        @return   reduced feature set
        """
        return np.dot((features - self.__mean), self.__comp.T)

class Fourier:
    """ calculate the averaged right half of the signals power spectrum and normalizes it """
    def __init__(self, sample_rate=16000, fft_len=2048, stride=1024, fg_l=80):
        """ load class variables\n
        @param   sample_rate   sampling frequency of the input data\n
        @param   fft_len       fft window length\n
        @param   stride        advance length for the averaging process\n
        @param   fg_l          lowest cutoff-freqency set lower frequencies to zero
        """
        err_str1 = "{0} has to be an positive integer value"
        assert (isinstance(sample_rate, int) and sample_rate > 0), err_str1.format("sample_rate")
        assert (isinstance(fft_len, int) and fft_len > 0), err_str1.format("fft_len")
        assert (isinstance(stride, int) and stride > 0), err_str1.format("stride")
        assert (isinstance(fg_l, int) and fg_l >= 0), err_str1.format("fg_l")
        self.__stride = stride
        self.__fft_len = fft_len
        # calculate the first sample based an the lowest cutoff-freqency
        self.__sample_l = int(np.floor(fg_l * (fft_len / sample_rate)))
        # calculate hanning window for short-term fourier transform
        self.__fft_win = np.hanning(fft_len)

    def full_spectrum(self, segment):
        """ turn an audio segment into a list of shor time ffts\n
        @param    segment    an audio segment of arbitrary length\n
        @return   spectrum   stft array
        """
        spectrum = []
        if len(segment) < self.__fft_len:
            steps = 1
        else:
            steps = (len(segment) - self.__fft_len) // self.__stride + 2
        for i in range(steps):
            # sliding indexes for fft window
            idx_0 = i * self.__stride
            idx_1 = idx_0 + self.__fft_len
            if i != steps - 1:
                # multiply frame by window function to reduce artifacts
                frame = segment[idx_0:idx_1] * self.__fft_win
            else:
                # last frame is smaller therefore a smaller window is required
                frame = segment[idx_0:] * np.hanning(len(segment) - idx_0)
            # calculate spectrum
            stft = abs(np.fft.rfft(frame, n=self.__fft_len))
            # set frequencies lower fg_l to zero
            stft[:self.__sample_l] = 0
            # scaling the data between 0 and 1. max() could introduce a dividing by zero exception,
            # if avg_fft is a zero array. in this case somethink else went wrong beforehand!
            # old approach: avg_fft = MinMaxScaler().fit_transform(avg_fft.reshape(-1, 1))
            stft -= stft.min()
            stft /= stft.max()
            if np.isnan(stft).any():
                raise ValueError("incorrect input introduced nan values!")
            spectrum.append(stft)
        return np.array(spectrum)

    def average_fft(self, segment):
        """ turn audio segment into an averaged fft\n
        @param    segment   an audio segment of arbitrary length\n
        @return   averaged fft
        """
        avg_fft = 0
        spectrum = self.full_spectrum(segment)
        # average ffts
        for fft in spectrum:
            avg_fft += fft
        # normalize output
        avg_fft -= avg_fft.min()
        avg_fft /= avg_fft.ptp()
        return avg_fft
