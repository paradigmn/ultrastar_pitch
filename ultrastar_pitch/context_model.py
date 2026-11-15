#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          context_model.py
@brief         note context network: corrects the pitch class scores of a song from its neighbouring notes
@author        paradigm

Small bidirectional GRUs over the notes of a song (four networks of 98k parameters each, about 20 milliseconds per
song). Per note they see the combined log scores of the pitch models before and after the repetition pooling, the
log prior of the detected key, the pause before the note, the note length and the phrase starts; each adds a
correction to exactly what the postprocessing decides otherwise (repetition pooling + soft key prior), so it starts
from that result. The model file holds all four and returns the mean of their log softmax.

Trained with dev/context_gru_2026.py on the pipeline scores of the 4781 training songs of the frame model
(pitchnet_2026_11_13 with the tuning correction, RMVPE weight 0.25): two networks on the mix (seeds 0, 1), two on
the mix and the mix + vocals versions of the songs (seeds 0, 1), each epoch chosen on the 249 validation songs,
joined by `context_gru_2026.py combine`. Results on the 197 evaluation songs: dev/README.md.
"""

import logging
import os
import sys
from typing import Optional, Sequence

import numpy as np
import onnxruntime as rt

MODEL_NAME = "context_gru_ensemble_2026_11_13.onnx"
LOGGER = logging.getLogger(__name__)

# pause before a note: < 0.05 s, 0.05 - 0.15, 0.15 - 0.4, 0.4 - 1.0, >= 1.0 s or a phrase break (or a jump back)
GAP_EDGES = [0.0, 0.05, 0.15, 0.4, 1.0]


def gap_bins(beats: Sequence[float], lengths: Sequence[float], durations: Sequence[float], phrases: Sequence[int]):
    """ (bin of the pause before every note [n], phrase start [n]); the seconds per beat come from the note
        lengths (beats) and durations (seconds), no start times needed """
    beats, lengths, durations = (np.asarray(x, dtype=np.float64) for x in (beats, lengths, durations))
    ok = lengths > 0
    spb = float(np.median(durations[ok] / lengths[ok])) if ok.any() else 0.0
    gap = np.full(len(beats), np.inf)
    gap[1:] = (beats[1:] - beats[:-1] - lengths[:-1]) * spb
    b = np.searchsorted(GAP_EDGES[1:], gap, side="right")
    brk = np.zeros(len(beats), dtype=bool)
    brk[1:] = np.asarray(phrases)[1:] != np.asarray(phrases)[:-1]
    b[brk | (gap < -1e-6)] = len(GAP_EDGES) - 1        # phrase break, or a jump back (second voice)
    return np.minimum(b, len(GAP_EDGES) - 1), brk


def note_features(scores: np.ndarray, pooled: np.ndarray, key_logprior: np.ndarray, durations: Sequence[float],
                  beats: Sequence[float], lengths: Sequence[float], phrases: Sequence[int]):
    """ inputs of the network per note: circular part [n, 36] (scores - max, key log prior, pooled - max; they
        rotate together with the pitch classes) and the other part [n, 7] (pause bin one hot, log length in
        seconds, phrase start) """
    scores, pooled, key_logprior = (np.asarray(x, dtype=np.float64) for x in (scores, pooled, key_logprior))
    gap, brk = gap_bins(beats, lengths, durations, phrases)
    circ = np.concatenate([scores - scores.max(axis=1, keepdims=True), key_logprior,
                           pooled - pooled.max(axis=1, keepdims=True)], axis=1)
    other = np.concatenate([np.eye(len(GAP_EDGES))[gap], np.log(np.asarray(durations, dtype=np.float64) + 1e-3)[:, None],
                            brk[:, None]], axis=1)
    return circ.astype(np.float32), other.astype(np.float32)


def default_model_path() -> str:
    """ the packaged network (binaries/, or the PyInstaller folder) """
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, MODEL_NAME)   # pylint: disable=protected-access
    return os.path.join(os.path.dirname(__file__), "binaries", MODEL_NAME)


class ContextModel:
    """ the note context network on the CPU (it is tiny) """

    def __init__(self, model: Optional[str] = None):
        options = rt.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model_path = model or default_model_path()
        self.sess = rt.InferenceSession(self.model_path, options, providers=["CPUExecutionProvider"])

    def correct(self, scores: np.ndarray, pooled: np.ndarray, key_logprior: np.ndarray, decided: np.ndarray,
                durations: Sequence[float], beats: Sequence[float], lengths: Sequence[float],
                phrases: Sequence[int]) -> np.ndarray:
        """ corrected scores [n, 12] of one song. scores: combined log scores of the pitch models per note,
            pooled: after the repetition pooling, key_logprior: log prior of the detected key per note,
            decided: what the postprocessing decides without the network (pooled + weight * key_logprior);
            durations in seconds, beats / lengths / phrases from ProjectParser.note_context() """
        decided = np.asarray(decided, dtype=np.float64)
        if not len(decided):
            return decided
        circ, other = note_features(scores, pooled, key_logprior, durations, beats, lengths, phrases)
        out = self.sess.run(None, {"circ": circ[None], "other": other[None], "base": decided.astype(np.float32)[None]})
        return out[0][0].astype(np.float64)
