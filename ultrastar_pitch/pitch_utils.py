#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          pitch_utils.py
@brief         additional utilities for ultrastar-pitch
@author        paradigm
"""

import numpy as np

# convert between numerical and alphabetic pitch notation
PITCH_MAP = {0 : "C_", 1 : "C#", 2 : "D_", 3 : "D#", 4 : "E_", 5 : "F_",
             6 : "F#", 7 : "G_", 8 : "G#", 9 : "A_", 10 : "A#", 11 : "B_"}

def prediction_score(y_true: np.ndarray, y_pred: np.ndarray):
    """ evaluate the prediction by showing accuracy and confusion matrix """
    # calculate confusion matrix
    c_mat = np.zeros((len(PITCH_MAP), len(PITCH_MAP)))
    for y_t, y_p in zip(y_true, y_pred):
        c_mat[y_t][y_p] += 1
    # print detailed confusion matrix
    print("pred", end="\t")
    for label in PITCH_MAP.values():
        print(label, end="\t")
    print("\ntrue")
    for key, label in PITCH_MAP.items():
        print(label, end="\t")
        for val in c_mat[key]:
            print(int(val), end="\t")
        print("")   
    print("\naccuracy: " + str(np.mean(np.array(y_true) == np.array(y_pred)) * 100) + "%")
