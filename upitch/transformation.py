#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          transformer.py
@brief         turn an audio block of arbitrary length into
               a transformed output of unified length
@author        paradigm
"""

import numpy as np

def zero_pad_array(array, new_size):
    """ pads signal array with zeros to the desired size\n
    @param array     signal data which has to be extended\n
    @param new_size  desired new array size
    """
    return np.pad(array, (0, (new_size - len(array))), 'constant')

def apply_pca(features, pca_params):
    """ uses principal component analysis to reduce variance and input features\n
    @param features    feature array for decomposition\n
    @param pca_params  parameter of a fitted pca decompositor\n

    @note at this point I am uncertain if this method should be binded to a class or not
    """
    # tbd!
    pass

class AverageFourier:
    """ calculate the averaged right half of the signals power spectrum and normalizes it\n
    @param sample_rate  sampling frequency of the input data\n
    @param fft_len      fft window length\n
    @param adv_len      advance length for the averaging process\n
    @param fg_l         lowest cutoff-freqency (shortens output array)\n
    """
    def __init__(self, sample_rate=16000, fft_len=2048, adv_len=512, fg_l=140):
        err_str1 = "{0} has to be an positive integer"
        assert (isinstance(sample_rate, int) and sample_rate > 0), err_str1.format("sample_rate")
        self.__sample_rate = sample_rate
        assert (isinstance(fft_len, int) and fft_len > 0), err_str1.format("fft_len")
        self.__fft_len = fft_len
        assert (isinstance(adv_len, int) and adv_len > 0), err_str1.format("adv_len")
        self.__adv_len = adv_len
        assert (isinstance(fg_l, int) and fg_l >= 0), err_str1.format("fg_l")
        self.__fg_l = fg_l
        # output data
        self.__out = None

    @property
    def out(self):
        """ getter method for out variable """
        return self.__out

    def load_audio_segment(self, segment):
        """ turn audio segment into an averaged fft\n
        @param segment  an audio segment of arbitrary length
        """
        # I don't feel save mergin the divisions due to possible rounding errors
        avg_steps = len(segment) // self.__adv_len - self.__fft_len // self.__adv_len + 1
        # tbd!

class AverageWavelet:
    """ calculate averaged wavelet transform of the signal """
    def __init__(self):
        # tbd!
        pass
