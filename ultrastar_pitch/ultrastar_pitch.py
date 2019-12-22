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

from .project_parser import ProjectParser
from .preprocessing import AverageFourier
from .classification import NeuronalNetwork

default_model = "keras_tf_1025_240_120_12_fft_0.model"

# define flags
parser = argparse.ArgumentParser(usage="%(prog)s [options] [args]")
parser.add_argument('input', nargs='?', default="notes.txt",
                    help="path / name of project input file (default=notes.txt)")
parser.add_argument("-o", "--output", action="store_true", default="notes_new.txt",
                    help="path / name of project output file (default=notes_new.txt)")
args = parser.parse_args()

def main():
    print("executing ultrastar-pitch pitch detection")

    proj_file = args.input
    dest_file = args.output

    # init project parser
    notes = ProjectParser()
    # init data preprocessor
    trafo = AverageFourier()
    # init pitch classifier
    if getattr(sys,'frozen',False):
        # use meipass in case of binary execution
        model = os.path.join(sys._MEIPASS, default_model)
    else:
        model_path = os.path.join(os.path.dirname(__file__), "models")
        model = os.path.join(model_path, default_model)
    clf = NeuronalNetwork(model)

    pitches = []
    # load and parse project file
    notes.load_note_file(proj_file)
    # divide audio into pitch segments
    audio_segments = notes.process_audio()
    # analyse each segment
    for segment in audio_segments:
        # transform segment into features for prediction
        features = trafo.transform_audio_segment(segment)
        # predict pitch based on the choosen classifier
        pitches.append(clf.predict(features))
    notes.update_pitches(pitches)
    notes.save_note_file(dest_file)
