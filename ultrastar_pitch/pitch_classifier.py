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
import onnxruntime as rt


MODEL_NAME = "pitchnet_2020_12_14.onnx"

class PitchClassifier:
    """ determines pitch by a trained neuronal network """

    def __init__(self, model: str = None):
        """ load and init onnx model """
        sess_options = rt.SessionOptions()
        # enable parallel graph execution
        sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        # optimize graph at runtime
        sess_options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if model:
            # load model from external source
            self.sess = rt.InferenceSession(model, sess_options)
        elif getattr(sys, "frozen", False):
            # load tmodel from meipass (pyinstaller)
            self.sess = rt.InferenceSession(
                os.path.join(sys._MEIPASS, MODEL_NAME), sess_options
            )
        else:
            # load models from binary folder
            self.sess = rt.InferenceSession(
                os.path.join(os.path.dirname(__file__), "binaries", MODEL_NAME),
                sess_options,
            )
        self.input_name = self.sess.get_inputs()[0].name

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ predict pitch probabilities from a given feature set (batch_size, feature_size) """
        return self.sess.run(None, {self.input_name: X.astype(np.float32)})[0]
