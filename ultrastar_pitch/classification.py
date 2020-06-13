#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          classification.py
@brief         determine the pitch by analysing the provided data
@author        paradigm
"""

import os
import sys
import numpy as np
import onnxruntime  as rt

class NeuronalNetwork:
    """ determines pitch by a trained neuronal network """
    def __init__(self, model=None):
        """ load class variables\n
        @param   model   path to previous trained tf-keras model
        """
        if model:
            # load model from external source
            self.sess = rt.InferenceSession(model)
        elif getattr(sys, 'frozen', False):
            # load tmodel from meipass in case of binary execution
            self.sess = rt.InferenceSession(os.path.join(sys._MEIPASS, "model.onnx"))
        else:
            # load models from binary folder
            self.sess = rt.InferenceSession(os.path.join(os.path.dirname(__file__), 
                                                         "binaries",  "model.onnx"))
        self.input_name = self.sess.get_inputs()[0].name 

    def predict(self, features):
        """ predict pitch probabilities of the given feature list
        @param    feature_list   list of features to classify (batch_size, feature_size)\n
        @return   list of probability lists (batch_size, label_size)
        """
        preds = np.squeeze(self.sess.run(None, {self.input_name : features.astype(np.float32)}))
        return np.squeeze(preds)