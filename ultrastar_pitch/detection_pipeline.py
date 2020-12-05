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
from .preprocessing import Fourier
from .preprocessing import PCA
from .classification import NeuronalNetwork
from .postprocessing import Markov
from .pitch_utils import PITCH_MAP


class DetectionPipeline():
    """ pitch detection pipeline with file handling """
    def __init__(self,
                 project_parser: ProjectParser,
                 fourier: Fourier,
                 pca: PCA,
                 neuronal_network: NeuronalNetwork,
                 markov: Markov):
        self.project_parser = project_parser
        self.fourier = fourier
        self.pca = pca
        self.neuronal_network = neuronal_network
        self.markov = markov

    def transform(self, notes_org: str,
                  notes_new: str,
                  postproc: bool=True) -> (np.ndarray, np.ndarray):
        """ load note file, extract audio, detect pitches, and write new notes.txt """
        self.project_parser.load_note_file(notes_org)
        pitches_org = self.project_parser.dump_pitches()
        pitches_new = []
        # number of blocks per segment
        len_arr = []
        # all singable block features of the song
        features = []
        # divide audio into pitch segments
        for audio_segment in self.project_parser.process_audio():
            # get block ffts for the segment
            spectrals = self.fourier.full_spectrum(audio_segment)
            # save number of blocks
            len_arr.append(len(spectrals))
            # append segment features to song features
            features.append(self.pca.transform(spectrals))
        # predict all block features at once (increases performance a lot!)
        pitches_prob = self.neuronal_network.predict(np.concatenate(features))
        # calculate the segment indexes from length array
        idx_0 = 0
        for lenght in len_arr:
            idx_1 = idx_0 + lenght
            # sum up the pitch probabilities of the segment
            segment_prob = np.sum(pitches_prob[idx_0:idx_1], axis=0)
            # determine the pitch with the highest probability of the segment
            pitches_new.append(segment_prob.argmax())
            idx_0 = idx_1
        if postproc == True:
            # guess song key from pitch distribution
            key = self.markov.detect_key(pitches_new)
            logging.info("Song was written in key: %d", key)
            pitches_new[:] = self.markov.correct_pitches(key, pitches_new)
        self.project_parser.update_pitches(pitches_new)
        self.project_parser.save_note_file(notes_new)
        return np.array(pitches_org), np.array(pitches_new)
