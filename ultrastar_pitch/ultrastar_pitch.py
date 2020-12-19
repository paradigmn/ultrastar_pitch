#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          ultrastar_pitch.py
@brief         main module to start pitch detection from command line
@author        paradigm
"""

import sys
import logging
import argparse
import numpy as np

from .version import __version__
from .project_parser import ProjectParser
from .audio_preprocessor import AudioPreprocessor
from .pitch_classifier import PitchClassifier
from .stochastic_postprocessor import StochasticPostprocessor
from .detection_pipeline import DetectionPipeline
from .pitch_utils import prediction_score
from .gui import Gui


def main():
    """ ultrastar-pitch entry point """
    print("executing ultrastar-pitch " + __version__ + "\n")

    # define and parse flags
    parser = argparse.ArgumentParser(usage="%(prog)s [options] [args]")
    parser.add_argument(
        "input",
        nargs="?",
        default="notes.txt",
        help="name of project input file (default=notes.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="notes_new.txt",
        help="name of project output file (default=notes_new.txt)",
    )
    parser.add_argument(
        "-a",
        "--accuracy",
        default=False,
        action="store_true",
        help="output the prediction accuracy (for debugging)",
    )
    parser.add_argument(
        "-m",
        "--no-postproc",
        default=True,
        action="store_false",
        help="disable statistical postprocessing",
    )
    parser.add_argument(
        "-g",
        "--gui",
        default=False,
        action="store_true",
        help="enable graphical interface",
    )
    parser.add_argument(
        "-l",
        "--log",
        default="warning",
        help="set logging level, e.g.  --log debug, --log warning",
    )
    args = parser.parse_args()
    # define and set logging level
    log_levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    logging.basicConfig(
        format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        level=log_levels.get(args.log.lower()),
    )
    # configure and init pipeline pitch detection
    detection_pipeline = DetectionPipeline(
        ProjectParser(), AudioPreprocessor(stride=128), PitchClassifier(), StochasticPostprocessor()
    )
    # run graphical user interface for pitch detection
    if args.gui or getattr(sys, "frozen", False):
        Gui(detection_pipeline)
        return
    # run command line interface for pitch detection
    pitches_old, pitches_new = detection_pipeline.transform(
        args.input, args.output, args.no_postproc
    )
    # output confusion matrix with prediction score
    if args.accuracy:
        prediction_score(np.array(pitches_old), np.array(pitches_new))
