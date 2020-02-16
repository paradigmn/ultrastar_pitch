#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          ultrastar_pitch.py
@brief         main module to start pitch detection from command line
@author        paradigm
"""

import os
import sys
import argparse
import numpy as np

from .project_parser import ProjectParser
from .preprocessing import AverageFourier
from .preprocessing import PCA
from .classification import NeuronalNetwork

tf_model = "tf2_256_96_12_astft_pca_1.model"
pca_comp = "pca_components.npy"
pca_mean = "pca_mean.npy"

# define flags
parser = argparse.ArgumentParser(usage="%(prog)s [options] [args]")
parser.add_argument('input', nargs='?', default="notes.txt",
                    help="name of project input file (default=notes.txt)")
parser.add_argument("-o", "--output", default="notes_new.txt",
                    help="name of project output file (default=notes_new.txt)")
parser.add_argument("-a", "--accuracy", action='store_true',
                    help="output the prediction accuracy (for debugging)")
args = parser.parse_args()

def prediction_score(y_true, y_pred):
    """ evaluate the prediction by showing accuracy and confusion matrix 
    @param y_true  original pitches (human labeld)
    @param y_pred  predicted pitches
    """
    correct = len(y_true[y_true == y_pred])
    print(str(correct / len(y_true) * 100) + "% accuracy\n")
    labels=["C_", "C#", "D_", "D#", "E_", "F_", "F#", "G_", "G#", "A_", "A#", "B_"]
    # calculate confusion matrix
    c_mat = np.zeros((len(labels),len(labels)))
    for y_t, y_p in zip(y_true, y_pred):
        c_mat[y_t][y_p] += 1         
    # print detailed confusion matrix
    print("pred", end="\t")
    for label in labels:
        print(label, end="\t")
    print("\ntrue") 
    for i, label in enumerate(labels):
        print(label, end="\t")
        for val in c_mat[i]:
            print(int(val), end="\t")
        print("")

def main():
    print("executing ultrastar-pitch pitch detection")

    proj_file = args.input
    dest_file = args.output

    # init project parser
    notes = ProjectParser()
    # init data preprocessor
    trafo = AverageFourier()
    # init pitch classifier
    if getattr(sys, 'frozen', False):
        # use meipass in case of binary execution
        model = os.path.join(sys._MEIPASS, tf_model)
        comp = os.path.join(sys._MEIPASS, pca_comp)
        mean = os.path.join(sys._MEIPASS, pca_mean)
    else:
        bin_path = os.path.join(os.path.dirname(__file__), "binaries")
        model = os.path.join(bin_path, tf_model)
        comp = os.path.join(bin_path, pca_comp)
        mean = os.path.join(bin_path, pca_mean)
    clf = NeuronalNetwork(model)
    # init pca decomposer
    decomp = PCA(mean, comp)

    # load and parse project file
    notes.load_note_file(proj_file)
    pitches_old = notes.dump_pitches()
    feature_list = []
    # divide audio into pitch segments
    audio_segments = notes.process_audio()
    # analyse each segment
    for segment in audio_segments:
        # emphasize audio segment
        segment_emph = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])
        # perform average fft on the segment
        astft = trafo.transform_audio_segment(segment_emph)
        # use pca to reduce feature space
        feature_list.append(decomp.transform(astft))
    # predict new pitches in batches 
    pitches_new = clf.predict_batch(feature_list)
    notes.update_pitches(pitches_new)
    notes.save_note_file(dest_file)

    if args.accuracy:
        prediction_score(np.array(pitches_old), np.array(pitches_new))
