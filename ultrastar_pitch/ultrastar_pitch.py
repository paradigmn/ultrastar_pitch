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
from .vocal_separator import MODEL_ENV_VAR, default_model_path
from .rmvpe import RMVPE_MODEL_ENV_VAR, default_model_path as default_rmvpe_path


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
        "-s",
        "--separate",
        default=False,
        action="store_true",
        help="isolate the vocals before the detection (slow, needs the separation model); "
        "the vocals are saved as vocals.wav next to the note file and reused next time",
    )
    parser.add_argument(
        "--vocal-model",
        default=None,
        help="path of the vocal separation model (default: $" + MODEL_ENV_VAR
        + " or the user data folder)",
    )
    parser.add_argument(
        "--rmvpe-model",
        default=None,
        help="path of the RMVPE pitch model, combined with the built-in classifier (default: $"
        + RMVPE_MODEL_ENV_VAR + " or the user data folder; without the model only the built-in "
        "classifier is used)",
    )
    parser.add_argument(
        "--no-rmvpe",
        default=True,
        dest="use_rmvpe",
        action="store_false",
        help="use the built-in classifier only, even if the RMVPE model is installed",
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
    vocal_model = args.vocal_model or default_model_path()
    if args.separate and not vocal_model:
        parser.error(
            "--separate needs a vocal separation model: pass --vocal-model or set $" + MODEL_ENV_VAR
        )
    detection_pipeline = DetectionPipeline(
        ProjectParser(),
        AudioPreprocessor(stride=128),
        PitchClassifier(),
        StochasticPostprocessor(),
        vocal_model=vocal_model,
        rmvpe_model=args.rmvpe_model,
        use_rmvpe=args.use_rmvpe,
    )
    if args.use_rmvpe and not (args.rmvpe_model or default_rmvpe_path()):
        print(
            "note: rmvpe.onnx not found, using the built-in classifier only (see README for the download)"
        )
    # run graphical user interface for pitch detection
    if args.gui or getattr(sys, "frozen", False):
        from .gui import Gui

        Gui(detection_pipeline)
        return
    # run command line interface for pitch detection
    def progress(done: int, total: int):
        print("separating vocals: chunk {0}/{1}".format(done, total), end="\r", flush=True)

    pitches_old, pitches_new = detection_pipeline.transform(
        args.input, args.output, args.no_postproc, separate_vocals=args.separate, progress=progress
    )
    if args.separate:
        print()
    print("models: {0}, audio: {1}".format(detection_pipeline.last_models, detection_pipeline.last_sources))
    # output confusion matrix with prediction score
    if args.accuracy:
        prediction_score(np.array(pitches_old), np.array(pitches_new))
