#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          postprocessing.py
@brief         use statistics to optimize the predicted pitches
@author        paradigm
"""

import numpy as np

class Markov:
    """ imporoove the pitch detection by applying statistical postprocessing
    @note: originally this class was based on markov chains (hence the name), 
    but other methodes prooved to archieve better results
    """
    # circle of five as matrix representation
    # the rows represent the dur key, while the columns correspond to the probability of the pitch
    # note: the probabilities where experimentally determined and manually cleaned up
    key_table = np.matrix([[0.19, 0.00, 0.21, 0.00, 0.21, 0.08, 0.00, 0.13, 0.00, 0.11, 0.00, 0.07],
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
                           [0.00, 0.11, 0.00, 0.19, 0.13, 0.00, 0.16, 0.00, 0.12, 0.00, 0.08, 0.21]]
                          )

    def detect_key(self, pitches):
        """ find the key of a song by analysing the detected pitches\n
        @param   pitches   list of detected pitches\n
        @return  the key of the song (0-11)
        """
        # count the occurance of every pitch
        pitch_distribution = [np.count_nonzero(np.array(pitches) == x) for x in range(12)]
        # calculate the weight for each key to dertermine the most likely one
        return (self.key_table @ pitch_distribution).argmax()

    def correct_pitches(self, key, pitches):
        """ correct the pitches by using the key probabilities\n
        @param   key       the key of the song (0-11)\n
        @param   pitches   list of detected pitches\n
        @return  array with adjusted pitches
        """
        for idx, pitch in enumerate(pitches):
            # if pitch doesn't harmonize with key, find a better on
            if not self.key_table[key, pitch]:
                # compare one halfttone higher and lower for the best suiting one
                prob_1 = self.key_table[key, (pitch + 1) % 12]
                prob_2 = self.key_table[key, (pitch - 1) % 12]
                pitches[idx] = (pitch + 1) % 12 if prob_1 >= prob_2 else (pitch - 1) % 12
        return pitches

    def weighted_prob(self, key, prob):
        """ revaluate the prediction probabilities by using weighting methodes / penalties  (w.i.p.) 
        @param   key    the key of the song (0-11)\n
        @param   prob   (n,12)-matrix of probabilities 
        @return  matrix array with updated probabilities
        """
        return prob * np.array(self.key_table[key])
