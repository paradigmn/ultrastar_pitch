#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          detection_pipeline.py
@brief         apply pitch detection with one pipeline
@author        paradigm
"""

import logging
import numpy as np

from .project_parser import ProjectParser
from .audio_preprocessor import AudioPreprocessor
from .pitch_classifier import PitchClassifier
from .stochastic_postprocessor import StochasticPostprocessor


class DetectionPipeline:
    """ pitch detection pipeline with file handling """

    def __init__(
        self,
        project_parser: ProjectParser,
        audio_preprocessor: AudioPreprocessor,
        pitch_classifier: PitchClassifier,
        stochastic_postprocessor: StochasticPostprocessor
    ):
        self.project_parser = project_parser
        self.audio_preprocessor = audio_preprocessor
        self.pitch_classifier = pitch_classifier
        self.stochastic_postprocessor = stochastic_postprocessor

    def transform(
        self, notes_org: str, notes_new: str, postproc: bool = True
    ) -> (np.ndarray, np.ndarray):
        """ load note file, extract audio, detect pitches, and write to new notes.txt """
        self.project_parser.load_note_file(notes_org)
        pitches_org = self.project_parser.dump_pitches()
        pitches_new = []
        # number of blocks per segment
        len_arr = []
        # all singable block features of the song
        X = []
        # divide audio into pitch segments
        for audio_segment in self.project_parser.process_audio():
            # get block ffts for the segment
            features = self.audio_preprocessor.transform(audio_segment)
            # save number of blocks
            len_arr.append(len(features))
            # append segment features to song features
            X.append(features)
        # predict all block features at once (increases performance a lot!)
        pitches_prob = self.pitch_classifier.predict(np.vstack(X))
        # calculate the segment indexes from length array
        idx_0 = 0
        for lenght in len_arr:
            idx_1 = idx_0 + lenght
            # sum up the pitch probabilities of the segment
            segment_prob = np.sum(pitches_prob[idx_0:idx_1], axis=0)
            # determine the pitch with the highest probability of the segment
            pitches_new.append(segment_prob.argmax())
            idx_0 = idx_1
        if postproc:
            # guess song key from pitch distribution
            key = self.stochastic_postprocessor.detect_key(pitches_new)
            logging.info("Song was written in key: %d", key)
            pitches_new[:] = self.stochastic_postprocessor.correct_pitches(key, pitches_new)
        self.project_parser.update_pitches(pitches_new)
        self.project_parser.save_note_file(notes_new)
        return np.array(pitches_org), np.array(pitches_new)
