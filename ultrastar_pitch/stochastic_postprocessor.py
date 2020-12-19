#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          postprocessing.py
@brief         use stochastic postprocessing to optimize the predicted pitches
@author        paradigm
"""

import numpy as np


class StochasticPostprocessor:
    """ improve the pitch detection by applying statistical postprocessing """

    # circle of five as stochastic matrix
    # the rows represent the pseudo keys, the columns correspond to the pitch probabilities
    # note: the probabilities where experimentally determined and manually cleaned up
    KEY_TABLE = np.matrix(
        [
            [0.19, 0.00, 0.21, 0.00, 0.21, 0.08, 0.00, 0.13, 0.00, 0.11, 0.00, 0.07],
            [0.09, 0.21, 0.00, 0.17, 0.00, 0.18, 0.08, 0.00, 0.13, 0.00, 0.14, 0.00],
            [0.00, 0.07, 0.19, 0.00, 0.18, 0.00, 0.17, 0.08, 0.00, 0.19, 0.00, 0.12],
            [0.11, 0.00, 0.10, 0.26, 0.00, 0.16, 0.00, 0.17, 0.07, 0.00, 0.13, 0.00],
            [0.00, 0.14, 0.00, 0.07, 0.28, 0.00, 0.17, 0.00, 0.13, 0.06, 0.00, 0.15],
            [0.15, 0.00, 0.16, 0.00, 0.13, 0.17, 0.00, 0.16, 0.00, 0.13, 0.10, 0.00],
            [0.00, 0.15, 0.00, 0.16, 0.00, 0.12, 0.17, 0.00, 0.14, 0.00, 0.14, 0.12],
            [0.09, 0.00, 0.16, 0.00, 0.16, 0.00, 0.11, 0.17, 0.00, 0.16, 0.00, 0.15],
            [0.18, 0.07, 0.00, 0.15, 0.00, 0.14, 0.00, 0.09, 0.19, 0.00, 0.18, 0.00],
            [0.00, 0.18, 0.10, 0.00, 0.14, 0.00, 0.15, 0.00, 0.10, 0.17, 0.00, 0.16],
            [0.13, 0.00, 0.15, 0.09, 0.00, 0.16, 0.00, 0.18, 0.00, 0.12, 0.17, 0.00],
            [0.00, 0.11, 0.00, 0.19, 0.13, 0.00, 0.16, 0.00, 0.12, 0.00, 0.08, 0.21]
        ]
    )
    
    # circle of five as binary matrix
    KEY_TABLE_BIN = np.matrix(
        [
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
        ]
    )

    @classmethod
    def detect_key(cls, pitches: list) -> int:
        """ find the pseudo key of a song by analysing the detected pitches """
        # count the occurance of every pitch
        pitch_distribution = [
            np.count_nonzero(np.array(pitches) == x) for x in range(12)
        ]
        # calculate the weight for each key to dertermine the most likely one
        return (cls.KEY_TABLE @ pitch_distribution).argmax()

    @classmethod
    def correct_pitches(cls, key: int, pitches: list) -> list:
        """ correct the pitches by using the pseudo key probabilities """
        for idx, pitch in enumerate(pitches):
            # if pitch doesn't harmonize with key, find a better on
            if not cls.KEY_TABLE[key, pitch]:
                # compare adjacent pitches
                prob_1 = cls.KEY_TABLE[key, (pitch + 1) % 12]
                prob_2 = cls.KEY_TABLE[key, (pitch - 1) % 12]
                pitches[idx] = (
                    (pitch + 1) % 12 if prob_1 >= prob_2 else (pitch - 1) % 12
                )
        return pitches
