#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tuning_correction.py – tuning correction and global transposition for F0 data
==============================================================================

Purpose
-------
The script takes the output of an F0 tracker (e.g. FCPE or RMVPE: one value
every 10 ms, 0 Hz = unvoiced) and turns it into one pitch class (pitch % 12)
per note segment. This needs two different global corrections per song. They
solve different problems and are applied one after the other.


1. Tuning correction (cents, continuous)
----------------------------------------
Problem:
    A recording is not necessarily tuned to A = 440 Hz (different tuning,
    tape speed, pitch shift, tempo change of the recording). If a song is,
    e.g., 40 cents too high, all notes lie 40 cents above the semitone grid
    instead of on it. When rounding to the nearest semitone, notes that are
    sung only slightly sharp then cross the 50-cent boundary and are
    misclassified.

Method (estimate_tuning_offset):
    1. Express F0 in semitones:          m = 69 + 12 * log2(f / 440)
    2. Distance to the nearest semitone: d = m - round(m)      (-0.5 .. +0.5)
    3. Circular mean: each distance is treated as an angle 2*pi*d and added
       as a vector, weighted (e.g. by the tracker confidence). Circular
       because +49 and -49 cents are practically the same.
    4. The angle of the sum vector is the offset. Its normalised length R
       (0..1) tells how clearly the grid can be recognised: if all tones lie
       exactly on a grid, R is close to 1; for uniformly distributed
       distances it is close to 0.
    If R is too small or there are too few voiced frames, the offset is set
    to 0 (unreliable). Vibrato largely averages out.

    Important: the offset can only be determined modulo one semitone (-50..+50
    cents). Audio that is 60 cents too high cannot be distinguished from audio
    that is 40 cents too low and sung one semitone higher. This integer
    remainder ends up in the transposition (point 2).

Use:
    The offset belongs in the pipeline itself, also when generating new notes
    without labels. It is subtracted from the F0 values before rounding.


2. Global transposition (shift_semitones, whole semitones)
----------------------------------------------------------
Problem:
    This concerns the note file, not the recording. Ultrastar files are not
    necessarily in the original key: they are often shifted by n semitones.
    Then all notes of the file are off from the audio by the same amount.

Method (estimate_shift_semitones):
    For every note that has both a prediction and a reference, the
    difference (reference - prediction) mod 12 is computed. The histogram
    over the 12 possible differences shows the most frequent shift. The share
    of notes in this maximum ("agreement") is a measure of reliability.

Sign:
    Here:  prediction + shift  ==  reference  (mod 12).
    The sign of `shift_semitones` in dev/label_quality.py may be defined the
    other way round. Check there before comparing!

Why:
    - Evaluation: a model with the correct pitch would be rated "wrong"
      against a shifted file. With the correction, one measures the error of
      the model and not that of the file.
    - Training labels: a consistently shifted file gives systematically
      wrong labels. Either correct them by the shift or leave them out.
    The transposition only exists when comparing with existing notes. When
    generating new notes, absolute pitch classes are written.


Order
-----
    1. Tuning correction in the audio (cents)    -> estimate_tuning_offset()
    2. Pitch class per segment (median, round)   -> segment_pitch_classes()
    3. Comparison with labels                    -> estimate_shift_semitones()


Pitch class convention
----------------------
    pitch_class = round(MIDI) % 12, i.e. C = 0, C# = 1, ... B = 11. This is the
    usual MIDI counting; whether it matches the pitch values of your parser
    (pitch % 12) should be checked once.


Limitations (not tested on real music)
--------------------------------------
    - The thresholds (min_strength, min_frames) are starting values, not
      measured values. For songs with little singing, strong noise or
      instruments that the tracker follows as well, the offset is uncertain.
    - The self test data is synthetic (noise, outliers, detuning).
    - The tuning is assumed to be constant over the song. Drift (tape
      drive, tempo change) is not captured.

Dependencies: numpy only.

Usage
-----
    python dev/tuning_correction.py --selftest
    python dev/tuning_correction.py --f0 f0.csv
        CSV without header row, columns: f0_hz[,confidence]  (0 Hz = unvoiced)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

A4_HZ = 440.0


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def hz_to_midi(f0_hz, ref_hz: float = A4_HZ) -> np.ndarray:
    """Hz -> MIDI semitones (float). Values <= 0 or not finite -> NaN."""
    f0 = np.asarray(f0_hz, dtype=np.float64)
    out = np.full(f0.shape, np.nan)
    ok = np.isfinite(f0) & (f0 > 0)
    out[ok] = 69.0 + 12.0 * np.log2(f0[ok] / ref_hz)
    return out


def apply_tuning(f0_hz, offset_cents: float) -> np.ndarray:
    """Subtracts the tuning offset from the F0 (Hz in, Hz out, 0 stays 0)."""
    f0 = np.asarray(f0_hz, dtype=np.float64)
    return f0 * 2.0 ** (-offset_cents / 1200.0)


# --------------------------------------------------------------------------- #
# 1. Tuning correction
# --------------------------------------------------------------------------- #
@dataclass
class TuningResult:
    offset_cents: float   # in -50..+50; 0.0 if unreliable
    strength: float       # R in 0..1 (length of the circular mean)
    n_frames: int         # number of frames used
    reliable: bool        # False -> offset was set to 0


def estimate_tuning_offset(
    f0_hz,
    confidence=None,
    min_confidence: float = 0.0,
    min_frames: int = 50,
    min_strength: float = 0.1,
) -> TuningResult:
    """Estimates the global detuning of a song in cents.

    f0_hz       F0 per frame in Hz, 0 = unvoiced.
    confidence  optional weights (e.g. tracker confidence), same length.
    Returns     TuningResult; offset_cents > 0 means: audio is too high.
    """
    midi = hz_to_midi(f0_hz)
    valid = np.isfinite(midi)
    if confidence is None:
        w = np.ones_like(midi)
    else:
        w = np.asarray(confidence, dtype=np.float64)
        if w.shape != midi.shape:
            raise ValueError("confidence and f0_hz must have the same length")
        valid &= np.isfinite(w) & (w >= min_confidence)
    n = int(valid.sum())
    if n < min_frames:
        return TuningResult(0.0, 0.0, n, False)

    m = midi[valid]
    wv = w[valid]
    if wv.sum() <= 0:
        return TuningResult(0.0, 0.0, n, False)

    d = m - np.round(m)                       # -0.5 .. +0.5 semitone
    ang = 2.0 * np.pi * d
    c = np.sum(wv * np.cos(ang))
    s = np.sum(wv * np.sin(ang))
    strength = float(np.hypot(c, s) / wv.sum())
    offset_cents = float(np.arctan2(s, c) / (2.0 * np.pi) * 100.0)

    reliable = strength >= min_strength
    return TuningResult(offset_cents if reliable else 0.0, strength, n, reliable)


# --------------------------------------------------------------------------- #
# 2. Pitch class per segment
# --------------------------------------------------------------------------- #
def segment_pitch_classes(
    f0_hz,
    frame_times,
    segments,
    offset_cents: float = 0.0,
    min_voiced_frames: int = 3,
):
    """Pitch class (0..11) and median MIDI pitch per note segment.

    f0_hz         F0 per frame in Hz, 0 = unvoiced.
    frame_times   time of each frame in seconds, sorted ascending (same length as f0_hz).
    segments      list/array of (start_s, end_s).
    offset_cents  tuning offset from estimate_tuning_offset().
    Returns       (pitch_class[int, -1 = unknown], midi_median[float, NaN])
    """
    f0 = np.asarray(f0_hz, dtype=np.float64)
    t = np.asarray(frame_times, dtype=np.float64)
    if f0.shape != t.shape:
        raise ValueError("f0_hz and frame_times must have the same length")
    midi = hz_to_midi(f0) - offset_cents / 100.0

    classes = np.full(len(segments), -1, dtype=int)
    medians = np.full(len(segments), np.nan)
    for i, (start, end) in enumerate(segments):
        # frame_times is sorted ascending: find the range by binary search instead of a mask
        lo = int(np.searchsorted(t, start, side="left"))
        hi = int(np.searchsorted(t, end, side="left"))
        vals = midi[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size < min_voiced_frames:
            continue
        med = float(np.median(vals))
        medians[i] = med
        classes[i] = int(round(med)) % 12
    return classes, medians


# --------------------------------------------------------------------------- #
# 3. Global transposition
# --------------------------------------------------------------------------- #
@dataclass
class ShiftResult:
    shift_semitones: int       # in -5..+6; prediction + shift == reference (mod 12)
    agreement: float           # share of notes with this shift
    agreement_unshifted: float # share at shift 0
    n_notes: int


def estimate_shift_semitones(pred_classes, ref_classes, weights=None) -> ShiftResult:
    """Determines the most frequent shift between prediction and reference.

    Notes with class < 0 (unknown) are ignored.
    """
    p = np.asarray(pred_classes, dtype=int)
    r = np.asarray(ref_classes, dtype=int)
    if p.shape != r.shape:
        raise ValueError("pred_classes and ref_classes must have the same length")
    w = np.ones(p.shape) if weights is None else np.asarray(weights, dtype=float)

    ok = (p >= 0) & (r >= 0)
    if not ok.any():
        return ShiftResult(0, 0.0, 0.0, 0)
    diff = (r[ok] - p[ok]) % 12
    hist = np.bincount(diff, weights=w[ok], minlength=12)
    total = hist.sum()
    best = int(np.argmax(hist))
    shift = best - 12 if best > 6 else best      # -5 .. +6
    return ShiftResult(
        shift, float(hist[best] / total), float(hist[0] / total), int(ok.sum())
    )


# --------------------------------------------------------------------------- #
# Self test with synthetic data
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    rng = np.random.default_rng(0)
    hop = 0.01
    n_notes = 60
    true_offset = 40.0       # cents, audio too high (close to the 50-cent rounding boundary)
    true_shift = 2           # file shifted by +2 semitones

    # notes: 0.4..1.0 s each, with 0.1 s pause in between
    midi_notes = rng.integers(55, 76, n_notes)
    durs = rng.uniform(0.4, 1.0, n_notes)
    starts = np.concatenate([[0.0], np.cumsum(durs[:-1] + 0.1)])
    ends = starts + durs
    total = ends[-1] + 0.2
    t = np.arange(0, total, hop)

    midi = np.full(t.shape, np.nan)
    for m, s, e in zip(midi_notes, starts, ends):
        sel = (t >= s) & (t < e)
        vib = 0.25 * np.sin(2 * np.pi * 5.5 * t[sel])       # vibrato +-25 cents
        jit = rng.normal(0.0, 0.15, sel.sum())               # noise
        dev = rng.normal(0.0, 0.15)                          # intonation of the note (~15 cents)
        midi[sel] = m + dev + true_offset / 100.0 + vib + jit
    f0 = np.where(np.isfinite(midi), 440.0 * 2.0 ** ((midi - 69.0) / 12.0), 0.0)

    # 10 % outliers in the middle of the singing (random pitches), low confidence
    conf = np.where(f0 > 0, 0.9, 0.0)
    out = (f0 > 0) & (rng.random(t.shape) < 0.10)
    f0[out] = 440.0 * 2.0 ** ((rng.uniform(40, 90, out.sum()) - 69.0) / 12.0)
    conf[out] = 0.3

    segs = list(zip(starts, ends))
    ok = True

    res = estimate_tuning_offset(f0, conf)
    err = abs(res.offset_cents - true_offset)
    print(f"Offset: estimated {res.offset_cents:+.1f} cents (true {true_offset:+.1f}), "
          f"R = {res.strength:.2f}, frames = {res.n_frames}")
    ok &= res.reliable and err < 6.0

    # pitch classes without and with correction
    truth = midi_notes % 12
    c0, _ = segment_pitch_classes(f0, t, segs, 0.0)
    c1, _ = segment_pitch_classes(f0, t, segs, res.offset_cents)
    acc0, acc1 = np.mean(c0 == truth), np.mean(c1 == truth)
    print(f"Pitch classes correct: without correction {acc0:.0%}, with correction {acc1:.0%}")
    ok &= acc1 >= 0.95 and acc1 > acc0 + 0.1

    # transposition: reference file shifted by +2 semitones
    ref = (truth + true_shift) % 12
    sh = estimate_shift_semitones(c1, ref)
    print(f"Shift: estimated {sh.shift_semitones:+d} (true {true_shift:+d}), "
          f"agreement {sh.agreement:.0%} (without shift {sh.agreement_unshifted:.0%})")
    ok &= sh.shift_semitones == true_shift and sh.agreement > 0.9

    # too little data / no grid -> offset 0
    few = estimate_tuning_offset(f0[:20], conf[:20])
    ok &= (not few.reliable) and few.offset_cents == 0.0
    flat = estimate_tuning_offset(440.0 * 2.0 ** (rng.uniform(-6, 6, 5000) / 12.0))
    print(f"No grid: R = {flat.strength:.2f}, reliable = {flat.reliable}")
    ok &= not flat.reliable

    print("SELF TEST:", "passed" if ok else "FAILED")
    return 0 if ok else 1


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true", help="synthetic test")
    ap.add_argument("--f0", help="CSV without header row: f0_hz[,confidence]")
    ap.add_argument("--min-strength", type=float, default=0.1)
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.f0:
        data = np.atleast_2d(np.loadtxt(args.f0, delimiter=","))
        if data.shape[0] == 1 and data.shape[1] > 2:
            data = data.T
        conf = data[:, 1] if data.shape[1] > 1 else None
        res = estimate_tuning_offset(data[:, 0], conf, min_strength=args.min_strength)
        print(f"offset_cents={res.offset_cents:+.1f} strength={res.strength:.3f} "
              f"frames={res.n_frames} reliable={res.reliable}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
