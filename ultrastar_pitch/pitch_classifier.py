#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          classification.py
@brief         determine the pitch by analysing the provided data
@author        paradigm

Two kinds of models are supported, told apart by the name of the onnx input:
  "power"  pitchnet_2026_11_13 (the default, binaries/): the power spectrum of the whole song per 10 ms frame
           (power_spectrogram), log probabilities of the 12 pitch classes and a voicing logit per frame
           (predict_frames); structure and training in dev/pitchnet_2026.ipynb (64 channels; the first
           version with 32 channels, pitchnet_2026_11_11, is in models/ of the development setup)
  "mdct"   pitchnet_2020_12_14 (the model of the 1.x versions, kept in binaries/ for comparisons,
           --pitchnet-model): one feature vector per audio window of a note (AudioPreprocessor),
           probabilities of the 12 pitch classes per window (predict)
"""

import os
import sys
import logging
import numpy as np
import onnxruntime as rt
import scipy.fft


MODEL_NAME = "pitchnet_2026_11_13.onnx"
LOGGER = logging.getLogger(__name__)

# front end of the frame model (the same numbers as dev/pitchnet_model.py)
FRAME_SR = 16000
FRAME_N_FFT = 2048              # 128 ms Hann window
FRAME_HOP = 160                 # 10 ms
FRAME_FFT_BINS = 536            # FFT bins below 4.2 kHz, the input of the model
FRAME_TUNING_MAX = 100.0        # cents: largest tuning offset shift_power corrects ...
FRAME_FFT_BINS_TUNING = int(np.ceil(FRAME_FFT_BINS * 2.0 ** (FRAME_TUNING_MAX / 1200.0))) + 2   # ... with these bins
FRAME_CHUNK = 2000              # frames per onnx call ...
FRAME_MARGIN = 100              # ... with this margin on both sides, dropped afterwards


def power_spectrogram(audio: np.ndarray, chunk: int = 4096, n_bins: int = FRAME_FFT_BINS) -> np.ndarray:
    """ float32 waveform [N] (-1 .. 1) -> power of the first n_bins (FRAME_FFT_BINS) FFT bins per 10 ms frame
        [T, n_bins], T = N // FRAME_HOP + 1, frame t centred at t * 10 ms (reflect padded, periodic
        Hann window, like torch.stft(center=True) in the training) """
    pad = FRAME_N_FFT // 2
    if len(audio) <= pad:   # the reflect padding needs pad + 1 samples (64 ms)
        audio = np.pad(audio, (0, pad + 1 - len(audio)))
    x = np.concatenate([audio[pad:0:-1], audio, audio[-2:-pad - 2:-1]]).astype(np.float32)
    n = 1 + (len(x) - FRAME_N_FFT) // FRAME_HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, FRAME_N_FFT), (x.strides[0] * FRAME_HOP, x.strides[0]))
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME_N_FFT) / FRAME_N_FFT)).astype(np.float32)
    out = np.empty((n, n_bins), dtype=np.float32)
    for s in range(0, n, chunk):
        spec = scipy.fft.rfft(frames[s:s + chunk] * window, axis=1, workers=-1)[:, :n_bins]
        out[s:s + chunk] = spec.real ** 2 + spec.imag ** 2
    return out


def shift_power(power: np.ndarray, cents: float, n_out: int = FRAME_FFT_BINS) -> np.ndarray:
    """ tuning correction in the frequency domain: power spectrum [T, n_in] of a song that is `cents` sharp
        (negative: flat) -> the power spectrum of the same song in tune [T, n_out]. Bin k of the result is
        read at bin k * 2^(cents / 1200) of the input (linear interpolation), which is what resampling the
        audio does to the spectrum, without touching the time axis; n_in must reach that bin
        (power_spectrogram(..., n_bins=FRAME_FFT_BINS_TUNING) for up to FRAME_TUNING_MAX cents) """
    src = np.arange(n_out) * 2.0 ** (float(cents) / 1200.0)
    i0 = np.clip(np.floor(src).astype(int), 0, power.shape[1] - 2)
    frac = np.clip(src - i0, 0.0, 1.0).astype(np.float32)
    return power[:, i0] * (1.0 - frac) + power[:, i0 + 1] * frac


class PitchClassifier:
    """ determines pitch by a trained neuronal network """

    def __init__(self, model: str = None):
        """ load and init onnx model """
        sess_options = rt.SessionOptions()
        # enable parallel graph execution
        sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        # optimize graph at runtime
        sess_options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if model:
            # load model from external source
            self.sess = rt.InferenceSession(model, sess_options, providers=providers)
        elif getattr(sys, "frozen", False):
            # load tmodel from meipass (pyinstaller)
            self.sess = rt.InferenceSession(
                os.path.join(sys._MEIPASS, MODEL_NAME),
                sess_options,
                providers=providers,
            )
        else:
            # load models from binary folder
            self.sess = rt.InferenceSession(
                os.path.join(os.path.dirname(__file__), "binaries", MODEL_NAME),
                sess_options,
                providers=providers,
            )
        active_providers = self.sess.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            LOGGER.warning(
                "ONNX Runtime is using CPU only; CUDAExecutionProvider could not "
                "be activated. Available providers: %s",
                active_providers,
            )
        else:
            LOGGER.info("ONNX Runtime providers: %s", active_providers)
        self.input_name = self.sess.get_inputs()[0].name
        # the frame model (pitchnet_2026) takes the power spectrum of the whole song
        self.frame_based = self.input_name == "power"
        self.model_name = os.path.basename(model) if model else MODEL_NAME

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ predict pitch probabilities from a given feature set (batch_size, feature_size) """
        if self.frame_based:
            raise ValueError("{0} is a frame model, use predict_frames".format(self.model_name))
        return self.sess.run(None, {self.input_name: X.astype(np.float32)})[0]

    def predict_frames(self, samples: np.ndarray, tuning_cents: float = 0.0):
        """ 16 kHz int16 (or float -1 .. 1) samples of the whole song -> (log probabilities of the 12 pitch
            classes per 10 ms frame [T, 12], voicing probability per frame [T]); the song is run in chunks
            with a margin, so the memory stays small and long songs give the same result as short ones.
            tuning_cents: tuning offset of the song (e.g. the RMVPE estimate), corrected in the spectrum """
        audio = samples.astype(np.float32) / (32768.0 if samples.dtype.kind == "i" else 1.0)
        if not tuning_cents:
            return self.predict_power(power_spectrogram(audio))
        cents = float(np.clip(tuning_cents, -FRAME_TUNING_MAX, FRAME_TUNING_MAX))
        return self.predict_power(shift_power(power_spectrogram(audio, n_bins=FRAME_FFT_BINS_TUNING), cents))

    def predict_power(self, power: np.ndarray):
        """ predict_frames on a precomputed power_spectrogram [T, FRAME_FFT_BINS] (several models on one song) """
        if not self.frame_based:
            raise ValueError("{0} is not a frame model, use predict".format(self.model_name))
        power = power[None]
        n = power.shape[1]
        logprob, voicing = [], []
        pos = 0
        while pos < n:
            start, end = max(0, pos - FRAME_MARGIN), min(n, pos + FRAME_CHUNK + FRAME_MARGIN)
            lp, v = self.sess.run(None, {self.input_name: power[:, start:end]})
            keep = slice(pos - start, pos - start + min(FRAME_CHUNK, n - pos))
            logprob.append(lp[0, keep])
            voicing.append(v[0, keep])
            pos += FRAME_CHUNK
        voicing = np.concatenate(voicing)
        return np.concatenate(logprob), 1.0 / (1.0 + np.exp(-voicing))

    @staticmethod
    def frame_scores_per_note(logprob: np.ndarray, segments) -> np.ndarray:
        """ log of the mean probability per note [n, 12]: the frame probabilities are summed over the frames
            whose centre lies inside the note (start, end in seconds), as the tool does for RMVPE """
        prob = np.exp(logprob.astype(np.float64))
        cs = np.vstack([np.zeros((1, 12)), np.cumsum(prob, axis=0)])
        arr = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
        frame_s = FRAME_HOP / FRAME_SR
        lo = np.clip(np.rint(arr[:, 0] / frame_s).astype(int), 0, len(prob))
        hi = np.clip(np.rint(arr[:, 1] / frame_s).astype(int), 0, len(prob))
        hi = np.maximum(hi, np.minimum(lo + 1, len(prob)))
        lo = np.minimum(lo, hi - 1)
        summed = cs[hi] - cs[lo]
        return np.log(summed / np.maximum(summed.sum(axis=1, keepdims=True), 1e-9) + 1e-6)
