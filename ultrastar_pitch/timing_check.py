#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          timing_check.py
@brief         check the timing of the notes against the audio, without pitches
@author        paradigm

RMVPE's confidence per 10 ms frame says how clearly someone sings. The notes of the file give the
same as an on/off signal. The peak of the cross-correlation of the two is the offset between notes
and audio (a wrong #GAP), its height says how well notes and audio fit together at all. Stretching
the notes by a small tempo factor and taking the best peak shows a wrong #BPM (or another version
of the song), because then the offset drifts through the song.

Measured on 197 songs (dev/gap_xcorr_eval.py, mix):
  - offset: within 50 ms of the pitch based offset (which needs correct pitches) for 86 % of the
    songs, within 100 ms for 97 %;
  - pitch detection at the corrected timing: +3.1 points exact [+2.0 .. +4.4] (the pitch based
    offset, an upper bound, gives +3.6); songs shifted by 60 ms or more gain 9 to 27 points.
    MIN_SHIFT_S and MIN_HEIGHT chosen on one song list and tested on the other: +2.9 [+1.7 .. +4.3];
    end-to-end in the DetectionPipeline with repetition pooling and key prior: 75.3 % -> 78.8 %
    (+3.2 [+2.0 .. +4.4]), with vocals 76.8 % -> 80.3 % (dev/pipeline_eval.py);
  - wrong audio (notes of another song): the peak height separates matching and wrong pairs with
    an AUC of 0.98; below MISMATCH_HEIGHT 85 % of the wrong pairs and 2.5 % of the matching songs;
  - tempo: a #BPM wrong by 0.2 / 0.5 / 1 % is found in 81 / 90 / 94 % of the songs, 2 % of the real
    files are flagged (all with poor pitch agreement, i.e. most likely really broken).

The notes are only shifted for the analysis, the note file keeps its timing.
"""

from typing import List, NamedTuple, Sequence, Tuple

import numpy as np

FRAME_S = 0.01                  # RMVPE frame
RANGE_S = 10.0                  # search range of the offset
# the offset is used for the analysis if it is larger than MIN_SHIFT_S and the peak is clear
MIN_SHIFT_S = 0.03
MIN_HEIGHT = 0.2
# warnings
GAP_WARN_S = 0.1                # offset worth telling the user
MISMATCH_HEIGHT = 0.15          # below: notes and audio hardly fit together
TEMPO_GRID = np.round(np.arange(-0.02, 0.02001, 0.0005), 5)
TEMPO_MIN = 0.0015              # tempo factor deviation worth a warning ...
TEMPO_MIN_GAIN = 0.05           # ... if the peak gets clearly higher with it
EPS = 1e-9


class TimingCheck(NamedTuple):
    offset_s: float             # measured offset, > 0: the audio is later than the notes
    height: float               # height of the correlation peak (0 .. 1)
    applied_s: float            # offset used for the analysis (0.0 if not reliable or too small)
    tempo: float                # best tempo factor - 1 (0.0 = the notes fit the tempo of the audio)
    tempo_gain: float           # increase of the peak height with that tempo factor
    warnings: List[str]


def note_mask(n_frames: int, segments) -> np.ndarray:
    """ 1.0 inside the notes, 0.0 outside, per 10 ms frame """
    times = np.arange(n_frames) * FRAME_S
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    diff = np.zeros(n_frames + 1)
    np.add.at(diff, np.searchsorted(times, seg[:, 0]), 1)
    np.add.at(diff, np.searchsorted(times, seg[:, 1]), -1)
    return (np.cumsum(diff)[:-1] > 0).astype(np.float64)


def cross_correlation(voicing: np.ndarray, notes: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """ normalised c[k] = sum_t v[t] n[t - k] of the mean free signals for k = -max_lag .. max_lag """
    v = voicing - voicing.mean()
    n = notes - notes.mean()
    size = 1 << int(np.ceil(np.log2(len(v) + len(n) + 1)))
    c = np.fft.irfft(np.fft.rfft(v, size) * np.conj(np.fft.rfft(n, size)), size)
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, c[lags % size] / max(float(np.linalg.norm(v) * np.linalg.norm(n)), EPS)


def find_offset(voicing: np.ndarray, segments, range_s: float = RANGE_S) -> Tuple[float, float]:
    """ (offset in seconds, peak height) of the notes against the voicing signal """
    lags, c = cross_correlation(voicing, note_mask(len(voicing), segments), int(round(range_s / FRAME_S)))
    i = int(np.argmax(c))
    return float(lags[i] * FRAME_S), float(c[i])


def find_tempo(voicing: np.ndarray, segments, grid: Sequence[float] = TEMPO_GRID,
               range_s: float = RANGE_S) -> Tuple[float, float]:
    """ (tempo factor - 1, gain of the peak height against factor 1): the notes are stretched around
        the first note by every factor of the grid """
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    t0 = seg[0, 0]
    heights = np.array([find_offset(voicing, t0 + (seg - t0) * (1 + eps), range_s)[1] for eps in grid])
    zero = int(np.argmin(np.abs(np.asarray(grid))))
    best = int(np.argmax(heights))
    return float(grid[best]), float(heights[best] - heights[zero])


def check_timing(voicing: np.ndarray, segments, correct: bool = True) -> TimingCheck:
    """ voicing: RMVPE confidence per frame (maximum of the salience); segments: (start, end) of the
        notes in seconds as in the file. correct=False only measures, the offset is not applied """
    warnings = []
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    if len(seg) == 0 or len(voicing) < 10:
        return TimingCheck(0.0, 0.0, 0.0, 0.0, 0.0, warnings)
    voicing = np.asarray(voicing, dtype=np.float64)
    offset, height = find_offset(voicing, seg)
    tempo, tempo_gain = find_tempo(voicing, seg)
    reliable = height >= MIN_HEIGHT
    applied = offset if correct and reliable and abs(offset) > MIN_SHIFT_S else 0.0
    if height < MISMATCH_HEIGHT:
        warnings.append(
            "notes and audio hardly fit together (correlation {0:.2f}): wrong audio file, another version "
            "of the song or a broken timing? The pitches are probably wrong.".format(height)
        )
    elif abs(offset) >= GAP_WARN_S and reliable:
        warnings.append(
            "#GAP seems to be off: the notes fit the singing best when shifted by {0:+.0f} ms (#GAP {0:+.0f} ms){1}".format(
                1000 * offset, ", used for the analysis, the file keeps its timing" if applied else "")
        )
    if abs(tempo) >= TEMPO_MIN and tempo_gain >= TEMPO_MIN_GAIN:
        warnings.append(
            "the timing drifts through the song: the notes fit best when stretched by {0:+.2f} %, check #BPM "
            "and the audio file (the pitches may be wrong in parts of the song)".format(100 * tempo)
        )
    return TimingCheck(offset, height, applied, tempo, tempo_gain, warnings)
