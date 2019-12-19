#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          transformer.py
@brief         turn an audio block of arbitrary length into
               a transformed output of unified length
@author        paradigm
@date          19-Dec-2019 15:25:00
@version       19.12.2019, rp
               @li creation
"""

import numpy as np

class AverageFourier:
    """ calculate the averaged right half of the signals power spectrum for an signal input\n
    @param sample_rate  sampling frequency of the input data\n
    @param fft_len      fft window length\n
    @param adv_len      advance length for the averaging process\n
    @param fg_l         lowest cutoff-freqency (shortens output array)\n
    @param fg_h         highest cutoff-freqency (shortens output array)
    """
    def __init__(self, sample_rate=16000, fft_len=2048, adv_len=512, fg_l=140, fg_h=880):
        err_str1 = "{0} has to be an positive integer"
        assert (isinstance(sample_rate, int) and sample_rate > 0), err_str1.format("sample_rate")
        self.__sample_rate = sample_rate
        assert (isinstance(fft_len, int) and fft_len > 0), err_str1.format("fft_len")
        self.__fft_len = fft_len
        assert (isinstance(adv_len, int) and adv_len > 0), err_str1.format("adv_len")
        self.__adv_len = adv_len
        assert (isinstance(fg_l, int) and fg_l > 0), err_str1.format("fg_l")
        self.__fg_l = fg_l
        assert (isinstance(fg_h, int) and fg_h > 0), err_str1.format("fg_h")
        self.__fg_h = fg_h

# def AverageWavelet:
#     """ calculate the signal averaged wavelet transform """
#    pass
