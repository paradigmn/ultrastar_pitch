#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          preprocessing.py
@brief         turn an audio block of arbitrary length into
               a transformed output of unified length
@author        paradigm
"""

import logging
import numpy as np
from scipy.fft import dct


class AudioPreprocessor:
    """ calculate the averaged right half of the signals power spectrum and normalizes it """

    def __init__(
        self, sr: int = 16000, win_len: int = 4096, stride: int = 1024
    ) -> np.ndarray:
        """ init state variables """
        err_str1 = "{0} has to be an positive integer value"
        assert isinstance(sr, int) and sr > 0, err_str1.format("sample_rate")
        assert isinstance(win_len, int) and win_len > 0, err_str1.format(
            "win_len"
        )
        assert isinstance(stride, int) and stride > 0, err_str1.format("stride")
        self.stride = stride
        self.win_len = win_len

    def transform(self, segment: np.ndarray) -> np.ndarray:
        """ turn an audio segment into mdct spectras """
        if len(segment) < self.win_len:
            # zero pad frame to window length before transformation
            mdct = abs(
                dct([segment], type=4, n=self.win_len, axis=1)[
                    :, : self.win_len // 2
                ]
            )
        else:
            # use dynamic stride for larger inputs
            dyn_stride = self.stride * (len(segment) // self.win_len)
            # create stride matrix from segment
            rows = (len(segment) - self.win_len) // dyn_stride + 1
            n_stride = segment.strides[0]
            frames = np.lib.stride_tricks.as_strided(
                segment,
                shape=(rows, self.win_len),
                strides=(dyn_stride * n_stride, n_stride),
            )
            mdct = abs(dct(frames, type=4, axis=1)[:, : self.win_len // 2])
        # remove spectral offset
        mdct -= mdct.min(keepdims=True, axis=1)
        # strip zero rows from spectrum to avoid division by zero
        mdct = mdct[~np.all(mdct == 0, axis=1)]
        # scale spectrum between 0 and 1
        mdct /= mdct.max(keepdims=True, axis=1)
        # segment is either silent or corrupt -> return zeros array
        if np.isnan(mdct).any() or len(mdct) == 0:
            logging.warning(
                "invalid input introduced nan values. check audio file integrity!"
            )
            mdct = np.zeros((1, self.win_len // 2))
        return mdct
