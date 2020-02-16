#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          classification.py
@brief         determine the pitch by analysing the provided data
@author        paradigm
"""

import numpy as np
import tensorflow as tf

class SlidingHarmonic:
    """ calculate the pitch based on the sum of basetone, first and second harmonic\n
    @param sample_rate   sampling frequency of the original signal\n
    @param fg_l lowest   basetone to analyse\n
    @param fg_h highest  basetone to analyse
    """
    def __init__(self, sample_rate=16000, fft_len=2048, fg_l=140, fg_h=880):
        self.__sample_rate = sample_rate
        self.__idx_min = (fg_l * fft_len) // sample_rate
        self.__idx_max = (fg_h * fft_len) // sample_rate
        self.__fft_len = fft_len

    def predict(self, fft):
        """ return the predicted pitch
        @param fft  unshortend signal spectrum\n
        @return  best fitting pitch
        """
        idx, weight_max = 0, 0
        # iterate between lowest and highest basetone frequency
        for i in range(self.__idx_min, self.__idx_max):
            # accumulate the base tone and the first two harmonics
            weight = fft[i]
            weight += 1/3 * sum(fft[2 * i - 1:2 * i + 1])
            weight += 1/5 * sum(fft[3 * i - 2:3 * i + 2])
            # catch the peak value
            if weight > weight_max:
                idx, weight_max = i, weight
        f_max = (idx * self.__sample_rate) / self.__fft_len
        return int(round(12 * np.log2(f_max / (440 * 2**(-4.75)))) % 12)

class NeuronalNetwork:
    """ determines pitch by a trained neuronal network
    @param model  deep learning model\n
    """
    def __init__(self, model):
        self.__model = tf.keras.models.load_model(model)

    def predict(self, features):
        """
        @param features  transformed data as input for a matching trained model\n
        @return  best fitting pitch
        """
        features = np.squeeze(features)
        preds = self.__model.predict(np.array([features]))
        pitch = preds.argmax(axis=1)[0]
        return pitch
    
    def predict_batch(self, feature_list):
        """
        @param feature_list  list of features to classify\n
        @return  best fitting pitch list
        """
        predictions = self.__model.predict(np.array(feature_list), batch_size=32)
        return [pred.argmax() for pred in predictions]
