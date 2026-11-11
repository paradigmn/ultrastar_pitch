#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          rmvpe.py
@brief         RMVPE vocal pitch estimation (onnx) and the note level features derived from it
@author        paradigm

The onnx model (Hugging Face "Scicom-intl/rmvpe-onnx", MIT) takes mono audio at 16 kHz
[1, N] and returns a salience map [1, T, 360]: one frame every 10 ms (T = N // 160 + 1) with
360 pitch bins 20 cent apart from 31.7 Hz to about 2 kHz. This module contains
  - Rmvpe: the onnx session (block processing of long audio) and the model lookup
  - decode_salience: salience -> f0 per frame (the RVC decoder)
  - estimate_tuning: global tuning offset in cents of the singing voice
  - chroma_per_note: salience summed per note and folded to the 12 pitch classes, which is the
    feature the detection pipeline combines with the pitchnet probabilities
  - timing_search: time offset between notes and audio, given reference pitch classes
    (vectorised over all offsets)
"""

import logging
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import tuning_correction as tc
from .model_files import RMVPE_MODEL_ENV_VAR, RMVPE_MODEL_FILE_NAME, find_model, user_model_dir

logger = logging.getLogger(__name__)

SR = 16000                  # sample rate of the model input
HOP = 160                   # 10 ms per frame
FRAME_S = HOP / SR
N_CLASS = 360
CENTS_BASE = 1997.3794084376191     # cents of bin 0 relative to 10 Hz
CENTS_STEP = 20.0
MIDI_10HZ = 69.0 + 12.0 * math.log2(10.0 / 440.0)
# midi value of every salience bin
BIN_MIDI = MIDI_10HZ + (CENTS_BASE + CENTS_STEP * np.arange(N_CLASS)) / 100.0
EPS = 1e-6


class RmvpeModelError(Exception):
    """ the model file is missing, corrupt or doesn't have the expected signature """


def default_model_path() -> Optional[str]:
    """ path of rmvpe.onnx ($ULTRASTAR_PITCH_RMVPE_MODEL, user data folder, ...) or None """
    return find_model(RMVPE_MODEL_FILE_NAME, RMVPE_MODEL_ENV_VAR)


def providers_for(gpu: bool) -> list:
    """ onnxruntime providers: cpu only, or cuda with cpu fallback. HEURISTIC instead of the
        default cudnn algorithm search, because the input length changes with every song """
    if not gpu:
        return ["CPUExecutionProvider"]
    return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]


class Rmvpe:
    """ RMVPE onnx session: audio [N] float (16 kHz) -> salience [T, 360] """

    def __init__(self, model_path: Optional[str] = None, threads: int = 0, gpu: bool = True):
        import onnxruntime as ort

        model_path = model_path or default_model_path()
        if not model_path:
            raise RmvpeModelError(
                "rmvpe.onnx not found: set $" + RMVPE_MODEL_ENV_VAR + " or copy the model into the user "
                "data folder (see README)"
            )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        if threads > 0:
            options.intra_op_num_threads = threads
        try:
            self.sess = ort.InferenceSession(model_path, options, providers=providers_for(gpu))
        except Exception as err:
            raise RmvpeModelError("can't load {0}: {1}".format(model_path, err)) from err
        inputs = self.sess.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 2:
            raise RmvpeModelError("unexpected model signature, expected one input [1, N]")
        self.model_path = model_path
        self.inp = inputs[0].name
        self.out = self.sess.get_outputs()[0].name
        self.device = "GPU" if "CUDAExecutionProvider" in self.sess.get_providers() else "CPU"
        logger.info("RMVPE model %s on %s", model_path, self.device)

    def _run(self, audio: np.ndarray) -> np.ndarray:
        sal = self.sess.run([self.out], {self.inp: audio[None].astype(np.float32)})[0]
        if sal.ndim != 3 or sal.shape[-1] != N_CLASS:
            raise RmvpeModelError("unexpected output shape {0}, expected [1, T, 360]".format(sal.shape))
        return sal[0]

    def salience(self, audio: np.ndarray, chunk_s: int = 20, margin_s: int = 2) -> np.ndarray:
        """ salience [T, 360], T = N // 160 + 1; long audio in blocks with overlapping margins """
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        chunk, margin = chunk_s * SR, margin_s * SR       # multiples of 160
        core = chunk - 2 * margin
        if n <= chunk:
            return self._run(audio)
        parts, pos = [], 0
        while pos < n:
            s, e = max(0, pos - margin), min(n, pos + core + margin)
            sal = self._run(audio[s:e])
            first = (pos - s) // HOP
            last = pos + core >= n
            parts.append(sal[first:] if last else sal[first:first + core // HOP])
            pos += core
        return np.concatenate(parts, axis=0)


def frame_times(sal: np.ndarray) -> np.ndarray:
    """ time of every salience frame in seconds """
    return np.arange(len(sal)) * FRAME_S


def decode_salience(sal: np.ndarray, threshold: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """ salience [T, 360] -> (f0_hz, confidence); f0 = 0 below the voicing threshold.
        Weighted mean of the 9 bins around the peak (the RVC decoder) """
    sal = np.asarray(sal, dtype=np.float64)
    t = sal.shape[0]
    center = sal.argmax(axis=1)
    padded = np.pad(sal, ((0, 0), (4, 4)))
    cents_map = np.pad(CENTS_STEP * np.arange(N_CLASS) + CENTS_BASE, (4, 4))
    idx = center[:, None] + np.arange(9)[None, :]
    w = padded[np.arange(t)[:, None], idx]
    cents = (w * cents_map[idx]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)
    conf = sal.max(axis=1)
    cents[conf <= threshold] = 0.0
    f0 = 10.0 * 2.0 ** (cents / 1200.0)
    f0[cents == 0.0] = 0.0
    return f0, conf


def note_frame_mask(times: np.ndarray, segments) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, end in segments:
        lo, hi = np.searchsorted(times, start), np.searchsorted(times, end)
        mask[lo:hi] = True
    return mask


def estimate_tuning(sal: np.ndarray, segments, threshold: float = 0.15, min_strength: float = 0.05,
                    min_note_frames: int = 50) -> tc.TuningResult:
    """ global tuning offset (cents) of the voice, estimated from the voiced frames inside the
        notes (outside of them instruments dominate), from all frames if there are too few """
    f0, conf = decode_salience(sal, threshold)
    times = frame_times(sal)
    mask = note_frame_mask(times, segments)
    if int(np.sum((f0 > 0) & mask)) >= min_note_frames:
        f0 = np.where(mask, f0, 0.0)
    return tc.estimate_tuning_offset(f0, conf, min_strength=min_strength)


def chroma_per_note(sal: np.ndarray, segments, offset_cents: float = 0.0, offset_s: float = 0.0) -> np.ndarray:
    """ salience summed over every note window and folded to the 12 pitch classes [n, 12]
        (tuning corrected, rows normalised to sum 1) """
    cls = np.rint(BIN_MIDI - offset_cents / 100.0).astype(int) % 12
    fold = np.zeros((N_CLASS, 12))
    fold[np.arange(N_CLASS), cls] = 1.0
    cs = np.vstack([np.zeros((1, 12)), np.cumsum(np.asarray(sal, dtype=np.float64) @ fold, axis=0)])
    arr = np.asarray(segments, dtype=np.float64).reshape(-1, 2) + offset_s
    times = frame_times(sal)
    lo = np.searchsorted(times, arr[:, 0])
    hi = np.maximum(np.searchsorted(times, arr[:, 1]), lo)
    chroma = cs[hi] - cs[lo]
    return chroma / np.maximum(chroma.sum(axis=1, keepdims=True), EPS)


# --------------------------------------------------------------------------- #
# time offset search (needs reference pitch classes)
# --------------------------------------------------------------------------- #
def segment_classes_offsets(f0, times, segments, offsets_s, offset_cents: float, min_frames: int,
                            max_elems: int = 4_000_000) -> np.ndarray:
    """ pitch class of every segment for many time offsets at once: [n_offsets, n_segments],
        -1 = not enough voiced frames. Identical to tc.segment_pitch_classes per offset """
    midi = tc.hz_to_midi(np.asarray(f0, dtype=np.float64)) - offset_cents / 100.0
    t = np.asarray(times, dtype=np.float64)
    arr = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    offsets_s = np.asarray(offsets_s, dtype=np.float64)
    out = np.full((len(offsets_s), len(arr)), -1, dtype=int)
    frames_per_offset = max(1, int(np.sum(np.searchsorted(t, arr[:, 1]) - np.searchsorted(t, arr[:, 0]))))
    block = max(1, max_elems // frames_per_offset)
    for b0 in range(0, len(offsets_s), block):
        offs = offsets_s[b0:b0 + block]
        lo = np.searchsorted(t, arr[None, :, 0] + offs[:, None], side="left")
        hi = np.searchsorted(t, arr[None, :, 1] + offs[:, None], side="left")
        length = np.maximum(hi - lo, 0)
        for n_win in np.unique(length):
            if n_win < max(min_frames, 1):
                continue
            rows, cols = np.nonzero(length == n_win)
            win = midi[lo[rows, cols][:, None] + np.arange(n_win)[None, :]]
            win.sort(axis=1)                                  # NaN (unvoiced) at the end
            n = np.sum(np.isfinite(win), axis=1)
            ok = n >= min_frames
            if not ok.any():
                continue
            win, n, rows, cols = win[ok], n[ok], rows[ok], cols[ok]
            k = np.arange(len(n))
            med = (win[k, (n - 1) // 2] + win[k, n // 2]) / 2.0
            out[b0 + rows, cols] = np.rint(med).astype(int) % 12
    return out


def shifts_for_classes(classes: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ tc.estimate_shift_semitones for every row of classes [n_offsets, n_segments] """
    ok = (classes >= 0) & (ref[None, :] >= 0)
    diff = (ref[None, :] - classes) % 12
    rows = np.nonzero(ok)[0]
    hist = np.bincount(rows * 12 + diff[ok], minlength=12 * len(classes)).reshape(-1, 12).astype(float)
    total = hist.sum(axis=1)
    best = hist.argmax(axis=1)
    agreement = np.where(total > 0, hist[np.arange(len(hist)), best] / np.maximum(total, 1), 0.0)
    shift = np.where(total > 0, np.where(best > 6, best - 12, best), 0)
    return shift, agreement


def timing_search(f0, times, segments, ref, offset_cents: float, range_ms: int = 2000, step_ms: int = 20,
                  min_frames: int = 3, plateau: float = 0.005) -> Tuple[int, tc.ShiftResult]:
    """ best time offset (ms, > 0 = audio later than the notes) by agreement of the pitch classes
        with the reference, free transposition. The middle of the plateau of offsets within
        `plateau` of the maximum is chosen """
    offsets = list(range(-range_ms, range_ms + 1, step_ms))
    cls = segment_classes_offsets(f0, times, segments, np.array(offsets) / 1000.0, offset_cents, min_frames)
    _, agree = shifts_for_classes(cls, np.asarray(ref))
    cand = np.flatnonzero(agree >= agree.max() - plateau)
    center = float(np.median(np.array(offsets)[cand]))
    pick = int(cand[np.argmin(np.abs(np.array(offsets)[cand] - center))])
    return offsets[pick], tc.estimate_shift_semitones(cls[pick], ref)
