#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          postprocessing.py
@brief         use statistics to optimize the predicted pitches
@author        paradigm
"""

import os
import sys
import numpy as np

class Markov:
    """ imporoove the pitch detection by applying statistical postprocessing with markov chains """
    # circle of five as matrix representation
    # the rows represent the dur key, while the columns correspond to the pitches
    key_table = np.matrix([[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
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
                           [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]])

    def __init__(self, trans_cube=None):
        """ load class variables\n
        @param   trans_cube   12x12x12 numpy array containing the transition matrices for each key
        """
        if trans_cube:
            # load transition matrices from external source
            self.trans_cube = trans_cube
        elif getattr(sys, 'frozen', False):
            # load transition matrices from meipass in case of binary execution
            self.trans_cube = np.load(os.path.join(sys._MEIPASS, "makov_trans_cube.npy"))
        else:
            # load transition matrices from binary folder
            self.trans_cube = np.load(os.path.join(
                os.path.dirname(__file__), "binaries", "makov_trans_cube.npy"))

    def detect_key(self, pitches):
        """ find the key of a song by analysing the detected pitches\n
        @param   pitches   list of detected pitches\n
        @return  the key of the song (0-11)
        """
        # count the occurance of every pitch
        pitch_distribution = [np.count_nonzero(np.array(pitches) == x) for x in range(12)]
        # calculate the weight for each key to dertermine the most likely one
        key_weights = self.key_table @ pitch_distribution
        return key_weights.argmax()
