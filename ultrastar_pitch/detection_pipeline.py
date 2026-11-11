#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          detection_pipeline.py
@brief         apply pitch detection with one pipeline
@author        paradigm

How the pitches are determined (measured on 197 songs, see dev/results/ensemble_eval_summary.txt):

  1. pitchnet (packaged): probabilities of the 12 pitch classes per note, sum of the frame
     probabilities. Exact pitch class on its own: 68.4 % with the timing of the file.
  2. RMVPE (optional, rmvpe.onnx): the salience map summed per note and folded to the 12 pitch
     classes (tuning corrected). The log-linear combination 0.6 * log RMVPE + 0.4 * log pitchnet
     gives 71.6 % (+3.2 points); pitchnet alone is used if the model is not installed.
  3. Vocal separation (optional): with the vocals separated, both models run on the mix AND on
     the vocals and the four votes are combined (vocals share 0.35): 73.1 % (+1.5 points).
  4. Postprocessing: the pseudo key of the song is detected and its pitch class probabilities are
     added as a soft prior to the log scores (StochasticPostprocessor.correct_scores): 73.7 % resp.
     75.5 %. The old hard correction (correct_pitches) is kept for reference only.
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from .project_parser import AudioSeparationError, ProjectParser
from .vocal_separator import SeparatorModelError, VocalSeparator
from .audio_preprocessor import AudioPreprocessor
from .pitch_classifier import PitchClassifier
from .stochastic_postprocessor import StochasticPostprocessor
from . import rmvpe as rmvpe_mod

logger = logging.getLogger(__name__)

EPS = 1e-6


class DetectionPipeline:
    """ pitch detection pipeline with file handling """

    # weight of RMVPE against pitchnet in the log-linear ensemble
    RMVPE_WEIGHT = 0.6
    # weight of the separated vocals against the mix when both are available
    VOCALS_WEIGHT = 0.35
    # RMVPE voicing threshold for the tuning estimate
    RMVPE_THRESHOLD = 0.15

    def __init__(
        self,
        project_parser: ProjectParser,
        audio_preprocessor: AudioPreprocessor,
        pitch_classifier: PitchClassifier,
        stochastic_postprocessor: StochasticPostprocessor,
        vocal_model: Optional[str] = None,
        rmvpe_model: Optional[str] = None,
        use_rmvpe: bool = True,
    ):
        self.project_parser = project_parser
        self.audio_preprocessor = audio_preprocessor
        self.pitch_classifier = pitch_classifier
        self.stochastic_postprocessor = stochastic_postprocessor
        # path of the onnx model for vocal separation, loaded on first use
        self.vocal_model = vocal_model
        self._separator = None
        self._separator_path = None
        # RMVPE model: None = look it up (environment variable, user data folder); loaded on first use
        self.rmvpe_model = rmvpe_model
        self.use_rmvpe = use_rmvpe
        self._rmvpe = None
        self._rmvpe_path = None
        self._rmvpe_missing_logged = False
        # what the last transform() used, for the user interface: "pitchnet", "pitchnet+rmvpe" and
        # the audio sources ("mix", "vocals", "mix+vocals")
        self.last_models = ""
        self.last_sources = ""

    # ----------------------------------------------------------------------- #
    # models
    # ----------------------------------------------------------------------- #
    def _get_separator(self) -> VocalSeparator:
        """ load the vocal separation model on demand (it needs a couple of seconds) """
        if not self.vocal_model:
            raise AudioSeparationError("no vocal separation model selected")
        if self._separator is None or self._separator_path != self.vocal_model:
            try:
                self._separator = VocalSeparator(self.vocal_model)
            except SeparatorModelError as err:
                raise AudioSeparationError(str(err)) from err
            self._separator_path = self.vocal_model
        return self._separator

    def _get_rmvpe(self) -> Optional[rmvpe_mod.Rmvpe]:
        """ the RMVPE session or None, if disabled or the model is not installed """
        if not self.use_rmvpe:
            return None
        path = self.rmvpe_model or rmvpe_mod.default_model_path()
        if not path:
            if not self._rmvpe_missing_logged:
                logger.warning(
                    "rmvpe.onnx not found, using pitchnet only (set $%s or copy the model into %s "
                    "for better results)", rmvpe_mod.RMVPE_MODEL_ENV_VAR, rmvpe_mod.user_model_dir()
                )
                self._rmvpe_missing_logged = True
            return None
        if self._rmvpe is None or self._rmvpe_path != path:
            self._rmvpe = rmvpe_mod.Rmvpe(path)
            self._rmvpe_path = path
        return self._rmvpe

    # ----------------------------------------------------------------------- #
    # features per note
    # ----------------------------------------------------------------------- #
    def pitchnet_scores(self, samples: np.ndarray, bounds: List[Tuple[int, int]]) -> np.ndarray:
        """ log probabilities of the 12 pitch classes per note [n, 12]: the frame probabilities
            of a note are summed (as before) and normalised """
        features, lengths = [], []
        for start, end in bounds:
            frames = self.audio_preprocessor.transform(samples[start:end])
            features.append(frames)
            lengths.append(len(frames))
        # predict all block features at once (increases performance a lot!)
        prob = self.pitch_classifier.predict(np.vstack(features))
        scores = np.zeros((len(bounds), 12))
        idx = 0
        for i, length in enumerate(lengths):
            p = prob[idx:idx + length].sum(axis=0)
            scores[i] = np.log(p / max(p.sum(), EPS) + EPS)
            idx += length
        return scores

    def rmvpe_scores(self, rmvpe: rmvpe_mod.Rmvpe, samples: np.ndarray, segments) -> np.ndarray:
        """ log of the RMVPE chroma per note [n, 12] (salience folded to pitch classes) """
        audio = samples.astype(np.float32) / 32768.0
        salience = rmvpe.salience(audio)
        tuning = rmvpe_mod.estimate_tuning(salience, segments, self.RMVPE_THRESHOLD)
        logger.info("RMVPE: tuning offset %+.0f cents (reliable: %s)", tuning.offset_cents, tuning.reliable)
        return np.log(rmvpe_mod.chroma_per_note(salience, segments, tuning.offset_cents) + EPS)

    def note_scores(self, samples: np.ndarray, bounds, segments) -> np.ndarray:
        """ combined log scores of one audio source: pitchnet, plus RMVPE if available """
        scores = self.pitchnet_scores(samples, bounds)
        rmvpe = self._get_rmvpe()
        if rmvpe is None:
            self.last_models = "pitchnet"
            return scores
        self.last_models = "pitchnet+rmvpe"
        return self.RMVPE_WEIGHT * self.rmvpe_scores(rmvpe, samples, segments) + (1 - self.RMVPE_WEIGHT) * scores

    # ----------------------------------------------------------------------- #
    # pipeline
    # ----------------------------------------------------------------------- #
    def transform(
        self,
        notes_org: str,
        notes_new: str,
        postproc: bool = True,
        separate_vocals: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
        use_vocals: Optional[bool] = None,
    ) -> (np.ndarray, np.ndarray):
        """ load note file, extract audio, detect pitches, and write to new notes.txt.
            separate_vocals isolates the vocals first (slow) and saves them as vocals.wav next to the
            notes. use_vocals: None = a #VOCALS header or an existing vocals.wav is used when present
            (free), True = like None but separate if there is none (same as separate_vocals), False =
            never. With vocals the models run on the mix and on the vocals and both votes are
            combined. progress(done, total) reports the separation """
        parser = self.project_parser
        parser.load_note_file(notes_org)
        pitches_org = parser.dump_pitches()
        segments = parser.note_segments()
        sample_rate = rmvpe_mod.SR
        bounds = parser.note_bounds(sample_rate)
        n_empty = sum(1 for start, end in bounds if end <= start)
        if n_empty:
            logger.warning("%d of %d notes have zero length, check #BPM and #VERSION!", n_empty, len(bounds))
        # the mix (never the saved vocals)
        mix = parser.load_main_audio(sample_rate)
        if any(start >= len(mix) for start, end in bounds):
            logger.warning("some notes lie beyond the end of the audio file, check #GAP and the audio file!")
        scores = self.note_scores(mix, bounds, segments)
        self.last_sources = "mix"
        # vocals: a #VOCALS header or a saved vocals.wav are used whenever present (free), the
        # separation only on request
        if use_vocals is None:
            use_vocals = separate_vocals or parser.has_vocal_audio()
        if use_vocals:
            separator = None if parser.has_vocal_audio() else self._get_separator()
            vocals = parser.load_audio(sample_rate, separator=separator, progress=progress, use_vocals_file=True)
            if parser.audio_source in ("#VOCALS", "vocals.wav", "separated"):
                logger.info("vocals from %s", parser.audio_source)
                scores = (1 - self.VOCALS_WEIGHT) * scores + self.VOCALS_WEIGHT * self.note_scores(
                    vocals, bounds, segments
                )
                self.last_sources = "mix+vocals"
        if postproc:
            key = self.stochastic_postprocessor.detect_key(scores.argmax(axis=1).tolist())
            logger.info("Song was written in key: %d", key)
            pitches_new = self.stochastic_postprocessor.correct_scores(scores)
        else:
            pitches_new = scores.argmax(axis=1)
        pitches_new = [int(p) for p in pitches_new]
        parser.update_pitches(pitches_new)
        parser.save_note_file(notes_new)
        return np.array(pitches_org), np.array(pitches_new)
