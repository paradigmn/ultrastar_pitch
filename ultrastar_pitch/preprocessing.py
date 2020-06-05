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
        """ turn an audio segment into a list of short time ffts\n
        @param    segment    an audio segment of arbitrary length\n
        @return   spectrum   stft array
        """
        if len(segment) < self.__fft_len:
            # frame is smaller therefore a smaller window is required
            frame = segment * np.hanning(len(segment))
            # zero pad frame to match size
            frame = [np.pad(frame, (0, self.__fft_len - len(frame)), mode="constant")]
            # calculate spectrum
            spectrum = abs(np.fft.rfft(frame, axis=1))
        else:
            # create stride matrix from segment
            rows = (len(segment) - self.__fft_len) // self.__stride + 1
            n_stride = segment.strides[0]
            frames = np.lib.stride_tricks.as_strided(segment, shape=(rows, self.__fft_len),
                                                     strides=(self.__stride * n_stride , n_stride))
            # calculate spectrum
            spectrum = abs(np.fft.rfft(frames * self.__fft_win, axis=1))
        # scaling the data between 0 and 1
        spectrum -= spectrum.min(keepdims=True, axis=1)
        spectrum /= spectrum.max(keepdims=True, axis=1)
        # set frequencies lower fg_l to zero
        spectrum[:,:self.__sample_l] = 0
        if np.isnan(spectrum).any():
            raise ValueError("incorrect input introduced nan values!")
        return spectrum

    def average_fft(self, segment):
        """ turn audio segment into an averaged fft\n
        @param    segment   an audio segment of arbitrary length\n
        @return   averaged fft
        """
        # average ful spectrum ffts
        avg_fft = np.sum(self.full_spectrum(segment), axis=0)
        # normalize output
        avg_fft -= avg_fft.min()
        avg_fft /= avg_fft.max()
        return avg_fft
