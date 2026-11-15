#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          ultrastar_pitch.py
@brief         main module to start pitch detection from command line
@author        paradigm
"""

import os
import sys
import logging
import argparse
import collections
import numpy as np

from .version import __version__
from .project_parser import ProjectParser
from .audio_preprocessor import AudioPreprocessor
from .pitch_classifier import PitchClassifier
from .stochastic_postprocessor import StochasticPostprocessor
from .detection_pipeline import DetectionPipeline, check_note_file, find_note_files
from .pitch_utils import prediction_score
from .vocal_separator import MODEL_ENV_VAR, default_model_path
from .rmvpe import RMVPE_MODEL_ENV_VAR, default_model_path as default_rmvpe_path
from .model_download import MODELS, ModelDownloadError, download_models
from .model_files import user_model_dir
from . import note_hygiene


def fix_list(text: str) -> tuple:
    """ argparse type of --fix """
    try:
        return note_hygiene.parse_fixes(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err))


def check_only(args, parser: argparse.ArgumentParser) -> int:
    """ --check-only: check one note file or all note files of a folder, without the pitch
        detection; returns the exit code (0, also if problems are found) """
    if os.path.isdir(args.input):
        if args.output:
            parser.error("-o needs a single note file, in a folder the fixed copies are named <name>_fixed.txt")
        files, skipped = find_note_files(args.input)
        root = args.input
    elif os.path.isfile(args.input):
        files, skipped, root = [args.input], 0, None
    else:
        parser.error("no such file or folder: {0}".format(args.input))
    counts = collections.Counter()
    for path in files:
        result = check_note_file(path, style=args.lint, fixes=args.fix, output=args.output)
        lines = result.report()
        if root is None or lines:
            print(os.path.relpath(path, root) if root else path)
            print("  " + ("\n  ".join(lines) if lines else "ok"))
        counts["unreadable"] += bool(result.error)
        counts["errors"] += any(i.severity == note_hygiene.ERROR for i in result.issues)
        counts["parser"] += bool(result.parser_warnings)
        counts["style"] += any(i.severity == note_hygiene.STYLE for i in result.issues)
        counts["written"] += bool(result.written)
        counts["write_failed"] += bool(result.write_error)
    if root is not None:
        summary = "{0} note files checked ({1} other .txt files skipped): {2} with structure errors, " \
                  "{3} not readable by the tool, {4} with parser warnings".format(
                      len(files), skipped, counts["errors"], counts["unreadable"], counts["parser"])
        if args.lint:
            summary += ", {0} with style hints".format(counts["style"])
        if args.fix:
            summary += ", {0} fixed copies written".format(counts["written"])
            if counts["write_failed"]:
                summary += " ({0} could not be written)".format(counts["write_failed"])
        print(summary)
    return 0


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
        default=None,
        help="name of project output file (default=notes_new.txt; with --check-only <name>_fixed.txt "
        "next to the note file)",
    )
    parser.add_argument(
        "--check-only",
        default=False,
        action="store_true",
        help="only check the note file, without the pitch detection (no audio, no models); the input can "
        "also be a folder, all note files in it and its subfolders are checked. With --fix a fixed copy "
        "<name>_fixed.txt is written next to every file a fix changes, the originals stay as they are",
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
        "--pitchnet-model",
        default=None,
        help="path of another pitch classifier onnx (default: the built-in pitchnet_2026_11_13; the model of "
        "the 1.x versions is ultrastar_pitch/binaries/pitchnet_2020_12_14.onnx)",
    )
    parser.add_argument(
        "--no-rmvpe",
        default=True,
        dest="use_rmvpe",
        action="store_false",
        help="use the built-in classifier only, even if the RMVPE model is installed",
    )
    parser.add_argument(
        "--no-timing-correction",
        default=True,
        dest="correct_timing",
        action="store_false",
        help="analyse the notes exactly at the timing of the file; by default a clear offset between notes "
        "and singing (wrong #GAP) is measured with RMVPE and used for the analysis (the file keeps its timing)",
    )
    parser.add_argument(
        "--no-context-model",
        default=True,
        dest="use_context_model",
        action="store_false",
        help="postprocess without the note context network (only repetition pooling and the soft key prior); "
        "by default the network corrects uncertain notes from their neighbours (with a frame model such as the "
        "built-in classifier)",
    )
    parser.add_argument(
        "--fix",
        default=(),
        metavar="LIST",
        type=fix_list,
        help="fix the note file in the output, comma separated: structure (overlapping notes, line breaks "
        "outside the pause, double line breaks, lines after E), linebreaks (line breaks where "
        "barbeque-squared's ultrastar-scripts put them), whitespace (spaces at the start of the next "
        "syllable instead of the end, ultrastar-scripts) or all; by default only the pitches are changed",
    )
    parser.add_argument(
        "--lint",
        default=False,
        action="store_true",
        help="also report style hints: the conventions of barbeque-squared's ultrastar-scripts (line break "
        "positions, spaces, golden notes, #YEAR/#LANGUAGE/#GENRE), #VERSION, encoding and missing "
        "cover/background/video files",
    )
    parser.add_argument(
        "--download-models",
        nargs="?",
        const="all",
        choices=["all"] + sorted(MODELS),
        default=None,
        help="download the optional models (rmvpe, vocals or all) into the user data folder and exit",
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
    if args.download_models:
        names = sorted(MODELS) if args.download_models == "all" else [args.download_models]
        print("downloading into " + user_model_dir())

        def show(file_name: str, done: int, total: int):
            print("  {0}: {1:.0f} / {2:.0f} MB".format(file_name, done / 1e6, total / 1e6), end="\r", flush=True)

        try:
            for path in download_models(names, progress=show):
                print("\n  ok: " + path)
        except ModelDownloadError as err:
            print()
            sys.exit("error: {0}".format(err))
        return
    if args.check_only:
        sys.exit(check_only(args, parser))
    vocal_model = args.vocal_model or default_model_path()
    if args.separate and not vocal_model:
        parser.error(
            "--separate needs the vocal separation model: run --download-models vocals, pass "
            "--vocal-model or set $" + MODEL_ENV_VAR
        )
    detection_pipeline = DetectionPipeline(
        ProjectParser(),
        AudioPreprocessor(stride=128),
        PitchClassifier(args.pitchnet_model),
        StochasticPostprocessor(),
        vocal_model=vocal_model,
        rmvpe_model=args.rmvpe_model,
        use_rmvpe=args.use_rmvpe,
        correct_timing=args.correct_timing,
        use_context_model=args.use_context_model,
        fixes=args.fix,
        style_check=args.lint,
    )
    if args.use_rmvpe and not (args.rmvpe_model or default_rmvpe_path()):
        print(
            "note: rmvpe.onnx not found, using the built-in classifier only "
            "(download it with: ultrastar-pitch --download-models)"
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
        args.input, args.output or "notes_new.txt", args.no_postproc, separate_vocals=args.separate, progress=progress
    )
    if args.separate:
        print()
    print("models: {0}, audio: {1}".format(detection_pipeline.last_models, detection_pipeline.last_sources))
    timing = detection_pipeline.last_timing
    if timing is not None:
        print("timing: offset {0:+.0f} ms (correlation {1:.2f}), {2}".format(
            1000 * timing.offset_s, timing.height,
            "used for the analysis" if timing.applied_s else "analysed at the timing of the file"))
    # note file hygiene: the structure errors were logged as warnings, the style hints on request
    if args.lint:
        hints = note_hygiene.summarize(detection_pipeline.last_hygiene, note_hygiene.STYLE, args.fix)
        print("style: " + ("no hints" if not hints else "\n  " + "\n  ".join(hints)))
    if detection_pipeline.last_fixes:
        print("fixed: " + ", ".join(
            "{0} {1}".format(count, what) for what, count in detection_pipeline.last_fixes.items()))
    # output confusion matrix with prediction score
    if args.accuracy:
        prediction_score(np.array(pitches_old), np.array(pitches_new))
