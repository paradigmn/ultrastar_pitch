#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          ultrastar_pitch.py
@brief         main module to start pitch detection from command line
@author        paradigm
"""

import argparse
import numpy as np

from .project_parser import ProjectParser
from .preprocessing import Fourier
from .preprocessing import PCA
from .classification import NeuronalNetwork
from .postprocessing import Markov

# convert between numerical and alphabetic pitch notation
PITCH_MAP = {0 : "C_", 1 : "C#", 2 : "D_", 3 : "D#", 4 : "E_", 5 : "F_",
             6 : "F#", 7 : "G_", 8 : "G#", 9 : "A_", 10 : "A#", 11 : "B_"}

def prediction_score(y_true, y_pred):
    """ evaluate the prediction by showing accuracy and confusion matrix
    @param y_true  original pitches (human labeld)
    @param y_pred  predicted pitches
    """
    print(str(np.mean(np.array(y_true) == np.array(y_pred)) * 100) + "% accuracy\n")
    labels = ["C_", "C#", "D_", "D#", "E_", "F_", "F#", "G_", "G#", "A_", "A#", "B_"]
    # calculate confusion matrix
    c_mat = np.zeros((len(labels), len(labels)))
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

    # define flags
    parser = argparse.ArgumentParser(usage="%(prog)s [options] [args]")
    parser.add_argument('input', nargs='?', default="notes.txt",
                        help="name of project input file (default=notes.txt)")
    parser.add_argument("-o", "--output", default="notes_new.txt",
                        help="name of project output file (default=notes_new.txt)")
    parser.add_argument("-a", "--accuracy", action='store_true',
                        help="output the prediction accuracy (for debugging)")
    parser.add_argument("-m", "--no-postproc", action='store_true',
                        help="disable statistical postprocessing")
    args = parser.parse_args()

    proj_file = args.input
    dest_file = args.output

    # init project parser
    notes = ProjectParser()
    # init data preprocessor
    trafo = Fourier(stride=128)
    # init pitch classifier
    clf = NeuronalNetwork()
    # init pca decomposer
    decomp = PCA()
    # init postprocessor
    postproc = Markov()

    # load and parse project file
    notes.load_note_file(proj_file)
    pitches_old = notes.dump_pitches()
    pitches_new = []
    # number of blocks per segment
    len_arr = []
    # all singable block features of the song
    features = []
    # divide audio into pitch segments
    for audio_segment in notes.process_audio():
        # get block ffts for the segment
        spectrals = trafo.full_spectrum(audio_segment)
        # save number of blocks
        len_arr.append(len(spectrals))
        # append segment features to song features
        features.append(decomp.transform(spectrals))
    # predict all block features at once (increases performance a lot!)
    pitches_prob = clf.predict_batch_prob(np.concatenate(features))
    # calculate the segment indexes from length array
    idx_0 = 0
    for lenght in len_arr:
        idx_1 = idx_0 + lenght
        # sum up the pitch probabilities of the segment
        segment_prob = np.sum(pitches_prob[idx_0:idx_1], axis=0)
        # determine the pitch with the highest probability of the segment
        pitches_new.append(segment_prob.argmax())
        idx_0 = idx_1

    if not args.no_postproc:
        # guess song key from pitch distribution
        key = postproc.detect_key(pitches_new)
        print("Song was written in: " + PITCH_MAP[key])
        pitches_new[:] = postproc.correct_pitches(key, pitches_new)

    if args.accuracy:
        prediction_score(np.array(pitches_old), np.array(pitches_new))
    
    notes.update_pitches(pitches_new)
    notes.save_note_file(dest_file)
