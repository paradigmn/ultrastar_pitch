#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rmvpe_eval.py – chroma accuracy of RMVPE against an Ultrastar notes.txt
=======================================================================

Purpose
-------
Get a feeling for how well the RMVPE weights match the Ultrastar notes, both
on the raw mix and on isolated vocals, side by side.

Procedure
---------
1. Audio -> 16 kHz mono (ffmpeg, any format) -> RMVPE (ONNX) -> F0 every 10 ms
2. notes.txt -> note segments in seconds (only ':' and '*', as in the parser)
3. Per note: median of the F0 semitone pitch -> pitch class (pitch % 12)
4. Comparison with the pitch class of the file, step by step with the
   corrections from tuning_correction.py:

     raw          no correction, timing as in the file
     +tuning      global detuning (cents) estimated from the F0 values
     +transp.     integer shift of the file (semitones), most frequent
                  difference reference - prediction
     +Timing*     additionally a time offset search (see below); the
                  transposition is determined anew for each offset
     control*     the same search with a randomly shuffled reference

The numbers are the shares of notes whose pitch class matches (only notes with
enough voiced frames). This is a note metric and not the same as RCA on frame
level.

About the asterisks (Timing*, control*)
---------------------------------------
Timing* takes the best of many time offsets for each source. This is an
UPPER BOUND: a search over many offsets and 12 transpositions finds something
even in pure chance. The control shows how much the search alone already
yields (reference randomly shuffled, otherwise the same). Only the gap between
Timing* and the control is real agreement.

Time offset
-----------
    offset_ms > 0  means: the audio is later than the notes. This matches the
    sign of `offset_ms` in dev/label_quality.py and would have to be added to
    #GAP. The search covers +-search-ms in steps of step-ms.

Tuning correction
-----------------
By default estimated only from frames inside note segments (in the mix there
are instruments, bass etc. otherwise); --tuning-all uses all voiced frames.

Existing classifier (--pitchnet)
--------------------------------
For comparison, the old classifier of the tool (pitchnet_2020_12_14.onnx) additionally
runs on every audio source, with the same preprocessing as the CLI
(AudioPreprocessor(stride=128), int16 audio, sum of the probabilities per
note). Two variants: PN-<source> without and PN-<source>+PP with the
statistical postprocessing (key correction, CLI default). Stages:

     raw          prediction of the tool, timing as in the file
     +tuning      not applicable (pitchnet has no tuning correction)
     +transp.     global transposition as for RMVPE
     +Timing*     time offset taken from the RMVPE search of the same source
                  and threshold (no search of its own), transposition
                  determined anew

The same notes as for RMVPE at stage Timing* are rated (notes with enough
voiced frames), so the comparison is paired. acc_all shows stage Timing* over
ALL notes (this is how the tool actually works).

Further features
----------------
    - Several sources in the same run: Mix (AUDIO), optionally --vocals
      (isolated vocals as a file), optionally --separate (Demucs model,
      separates in the same run; label "Separated").
    - --thresholds: several voicing thresholds with ONE RMVPE run, because the
      salience is computed only once and decoded several times.
    - --csv: result per note for inspection (which notes are wrong?).
    - Long audio is processed in blocks (20 s, 2 s margin) so that memory
      does not explode.
    - The error distribution shows whether errors are near misses (+-1
      semitone).

Limitations
-----------
    - Own small notes.txt parser (V1, V2, unversioned; decimal comma).
      Not supported: #RELATIVE:YES, BPM changes ('B'). Duets only with
      --include-duets (then both voices are checked against one F0, which is
      only rough).
    - Freestyle and rap notes (F, R, G) are skipped.
    - The Ultrastar notes are handmade: label noise lowers every number.
      The metric measures the agreement with the file, not the truth.
    - RMVPE is monophonic. Backing vocals and harmonies can disturb F0 and
      voicing, in the mix also instruments.
    - The thresholds and defaults are starting values, not tuned on real
      music.

Dependencies
------------
    numpy, scipy (resampling, WAV), onnxruntime, ffmpeg in PATH (any audio
    format; without ffmpeg only .wav files work) and tuning_correction.py
    in the same folder. Only for --separate: ultrastar_pitch/vocal_separator.py.

Usage (from the repo root)
--------------------------
    # raw mix against notes (default model: models/rmvpe.onnx)
    python dev/rmvpe_eval.py dev/testdata/song.mp3 dev/testdata/notes.txt \\
        --model models/rmvpe.onnx

    # compare mix and isolated vocals (e.g. vocals.wav from the cache)
    python dev/rmvpe_eval.py dev/testdata/song.mp3 dev/testdata/notes.txt \\
        --vocals dev/testdata/vocals.wav

    # separate the vocals in the same run (slow) and save them for later
    python dev/rmvpe_eval.py dev/testdata/song.mp3 dev/testdata/notes.txt \\
        --separate models/htdemucs_vocals.onnx --save-vocals vocals.wav

    # self test of the evaluation (synthetic data, no model)
    python dev/rmvpe_eval.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):                    # tuning_correction, ultrastar_pitch/
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from ultrastar_pitch import tuning_correction as tc              # noqa: E402
from ultrastar_pitch import rmvpe as rmvpe_mod                     # noqa: E402

SR = rmvpe_mod.SR                  # 16 kHz, sample rate of the RMVPE input
HOP = rmvpe_mod.HOP                # 10 ms per frame
FRAME_S = rmvpe_mod.FRAME_S
N_CLASS = rmvpe_mod.N_CLASS
CENTS_BASE = rmvpe_mod.CENTS_BASE
CENTS_STEP = rmvpe_mod.CENTS_STEP


class EvalError(Exception):
    """Error with a readable message for the console."""


class DuetError(EvalError):
    """Duet file without --include-duets."""


# --------------------------------------------------------------------------- #
# notes.txt
# --------------------------------------------------------------------------- #
@dataclass
class Notes:
    starts: np.ndarray        # seconds (as in the file, incl. #GAP)
    ends: np.ndarray
    pitches: np.ndarray       # raw values of the file
    voices: np.ndarray
    bpm: float
    gap_ms: float
    beat_ms: float
    v2: bool
    duet: bool
    n_skipped_special: int    # F/R/G notes
    header: dict = None       # all #KEY:value lines (keys in upper case)

    @property
    def ref_classes(self) -> np.ndarray:
        return self.pitches % 12

    @property
    def segments(self) -> list[tuple[float, float]]:
        return list(zip(self.starts.tolist(), self.ends.tolist()))


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def parse_notes(path: str, include_duets: bool = False) -> Notes:
    p = Path(path)
    if not p.is_file():
        raise EvalError(f"notes.txt not found: {path}")
    header: dict[str, str] = {}
    body: list[str] = []
    for line in _read_text(p).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, _, val = line[1:].partition(":")
            header.setdefault(key.strip().upper(), val.strip())
        else:
            body.append(line)
    if "BPM" not in header:
        raise EvalError("#BPM missing in the file")
    try:
        bpm = float(header["BPM"].replace(",", "."))
        gap_ms = float(header.get("GAP", "0").replace(",", ".") or 0)
    except ValueError as e:
        raise EvalError(f"#BPM/#GAP not readable: {e}") from e
    if bpm <= 0:
        raise EvalError("#BPM must be positive")
    if header.get("RELATIVE", "").upper() == "YES":
        raise EvalError("#RELATIVE:YES is not supported")
    version = header.get("VERSION", "").strip()
    try:
        v2 = bool(version) and int(version.split(".")[0]) >= 2
    except ValueError:
        v2 = False
    beat_ms = (60000.0 if v2 else 15000.0) / bpm       # V2: real BPM; otherwise BPM x4

    starts, ends, pitches, voices = [], [], [], []
    voice, duet, special = 1, False, 0
    for line in body:
        c = line[0]
        if c == "E":
            break
        if c == "P":
            duet = True
            try:
                voice = int(line[1:].strip() or 1)
            except ValueError:
                voice = 1
            continue
        if c == "B":
            raise EvalError("BPM changes ('B' in the notes) are not supported")
        if c == "-":
            continue
        if c in "FRG":
            special += 1
            continue
        if c not in ":*":
            continue
        parts = line.split(None, 4)
        try:
            start_beat, dur, pitch = int(parts[1]), int(parts[2]), int(parts[3])
        except (IndexError, ValueError) as e:
            raise EvalError(f"note line not readable: {line[:60]!r}") from e
        t0 = gap_ms / 1000.0 + start_beat * beat_ms / 1000.0
        t1 = t0 + max(dur, 0) * beat_ms / 1000.0
        starts.append(t0), ends.append(t1), pitches.append(pitch), voices.append(voice)
    if duet and not include_duets:
        raise DuetError("duet file: use --include-duets to evaluate anyway (only rough)")
    if not starts:
        raise EvalError("no singable notes (':' / '*') found")
    return Notes(np.array(starts), np.array(ends), np.array(pitches, dtype=int),
                 np.array(voices, dtype=int), bpm, gap_ms, beat_ms, v2, duet, special, header)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def load_audio(path: str, sr: int, channels: int = 1) -> np.ndarray:
    """Any audio -> float32, mono (N,) or stereo (2, N)."""
    if not Path(path).is_file():
        raise EvalError(f"audio file not found: {path}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn",
           "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
        y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    except FileNotFoundError:
        return _load_wav_fallback(path, sr, channels)
    except subprocess.CalledProcessError as e:
        raise EvalError(f"ffmpeg could not read '{path}': "
                        f"{e.stderr.decode(errors='replace')[:300]}") from e
    if y.size == 0:
        raise EvalError(f"'{path}' contains no audio")
    return y.reshape(-1, 2).T.copy() if channels == 2 else y


def load_audio_int16(path: str, sr: int = SR) -> np.ndarray:
    """Mono int16 exactly like ProjectParser (ffmpeg -ac 1 -ar sr pcm_s16le). For pitchnet:
    with f32 output ffmpeg gives noticeably different samples (different resampling/mixing),
    and that alone changes some predictions."""
    if not Path(path).is_file():
        raise EvalError(f"audio file not found: {path}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sr), "-acodec", "pcm_s16le", "-f", "s16le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        return Pitchnet.to_int16(_load_wav_fallback(path, sr, 1))
    except subprocess.CalledProcessError as e:
        raise EvalError(f"ffmpeg could not read '{path}': "
                        f"{e.stderr.decode(errors='replace')[:300]}") from e
    y = np.frombuffer(proc.stdout, dtype="<i2").copy()
    if y.size == 0:
        raise EvalError(f"'{path}' contains no audio")
    return y


def _load_wav_fallback(path: str, sr: int, channels: int) -> np.ndarray:
    if not path.lower().endswith(".wav"):
        raise EvalError("ffmpeg not found; without ffmpeg only .wav files work")
    from math import gcd
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    rate, y = wavfile.read(path)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / float(np.iinfo(y.dtype).max + 1)
    else:
        y = y.astype(np.float32)
    y = y.reshape(len(y), -1)
    y = y.mean(axis=1) if channels == 1 else (np.repeat(y, 2, 1) if y.shape[1] == 1 else y[:, :2]).T
    if rate != sr:
        g = gcd(int(rate), sr)
        y = resample_poly(y, sr // g, int(rate) // g, axis=-1).astype(np.float32)
    return np.ascontiguousarray(y, dtype=np.float32)


# --------------------------------------------------------------------------- #
# RMVPE
# --------------------------------------------------------------------------- #
providers_for = rmvpe_mod.providers_for
decode_salience = rmvpe_mod.decode_salience


class Rmvpe(rmvpe_mod.Rmvpe):
    """RMVPE session of the package (ultrastar_pitch/rmvpe.py) with the error handling of this script."""

    def __init__(self, model_path: str, threads: int = 0, gpu: bool = False):
        if not Path(model_path).is_file():
            raise EvalError(f"RMVPE model not found: {model_path}")
        try:
            super().__init__(model_path, threads, gpu)
        except rmvpe_mod.RmvpeModelError as e:
            raise EvalError(str(e)) from e
        if gpu and self.device != "GPU":
            print("warning: --gpu given, but CUDA is not available; RMVPE runs on the CPU", file=sys.stderr)


def accuracy(pred: np.ndarray, ref: np.ndarray) -> float:
    ok = pred >= 0
    return float(np.mean(pred[ok] == ref[ok])) if ok.any() else float("nan")


def rotate(classes: np.ndarray, shift: int) -> np.ndarray:
    out = classes.copy()
    m = out >= 0
    out[m] = (out[m] + shift) % 12
    return out


def note_frame_mask(times: np.ndarray, segs) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for s, e in segs:
        lo, hi = np.searchsorted(times, s), np.searchsorted(times, e)
        mask[lo:hi] = True
    return mask


def _timing_search_loop(f0, times, segs, ref, offset_cents, range_ms, step_ms, min_frames,
                        plateau=0.005):
    """Reference implementation of timing_search (one Python loop per offset), only used
    by the self test to check that the vectorised version gives identical results."""
    arr = np.asarray(segs, dtype=np.float64)
    offsets, shifts = [], []
    for d in range(-range_ms, range_ms + 1, step_ms):
        shifted = [(a, b) for a, b in (arr + d / 1000.0)]
        cls, _ = tc.segment_pitch_classes(f0, times, shifted, offset_cents, min_frames)
        offsets.append(d)
        shifts.append(tc.estimate_shift_semitones(cls, ref))
    agree = np.array([x.agreement for x in shifts])
    cand = np.flatnonzero(agree >= agree.max() - plateau)
    center = float(np.median(np.array(offsets)[cand]))
    pick = int(cand[np.argmin(np.abs(np.array(offsets)[cand] - center))])
    return offsets[pick], shifts[pick]


segment_classes_offsets = rmvpe_mod.segment_classes_offsets
shifts_for_classes = rmvpe_mod.shifts_for_classes
timing_search = rmvpe_mod.timing_search


@dataclass
class Result:
    label: str
    threshold: float
    n_notes: int
    n_eval: int
    voiced_share: float
    acc_raw: float
    acc_tuning: float
    acc_shift: float
    acc_timing: float
    acc_control: float
    tuning: "tc.TuningResult"
    shift: int
    shift_agreement: float
    offset_ms: int
    final_shift: int
    errors: dict
    acc_weighted: float
    final_classes: np.ndarray
    raw_classes: np.ndarray
    final_medians: np.ndarray
    acc_all: float = float("nan")     # pitchnet only: stage Timing* over all notes


def final_stats(final: np.ndarray, ref: np.ndarray, notes: Notes):
    """Error distribution (shares per semitone distance) and time-weighted accuracy."""
    ok = final >= 0
    dist = np.minimum((final[ok] - ref[ok]) % 12, (ref[ok] - final[ok]) % 12)
    errors = {k: float(np.mean(dist == k)) if ok.any() else float("nan") for k in (0, 1, 2)}
    errors[3] = float(np.mean(dist >= 3)) if ok.any() else float("nan")
    dur = notes.ends - notes.starts
    acc_w = float(np.sum(dur[ok][dist == 0]) / np.sum(dur[ok])) if ok.any() and dur[ok].sum() > 0 else float("nan")
    return errors, acc_w


def analyze(label, sal, notes: Notes, threshold, args) -> Result:
    f0, conf = decode_salience(sal, threshold)
    times = np.arange(len(f0)) * FRAME_S
    segs, ref = notes.segments, notes.ref_classes
    mf = args.min_frames

    # stage 1: raw
    raw, _ = tc.segment_pitch_classes(f0, times, segs, 0.0, mf)
    acc_raw = accuracy(raw, ref)

    # stage 2: tuning correction (preferably from frames inside the notes)
    f0_t = f0
    if not args.tuning_all:
        mask = note_frame_mask(times, segs)
        if int(np.sum((f0 > 0) & mask)) >= 50:
            f0_t = np.where(mask, f0, 0.0)
    tun = tc.estimate_tuning_offset(f0_t, conf, min_strength=args.min_strength)
    cls1, med1 = tc.segment_pitch_classes(f0, times, segs, tun.offset_cents, mf)
    acc_tun = accuracy(cls1, ref)

    # stage 3: global transposition (timing as in the file)
    sh = tc.estimate_shift_semitones(cls1, ref)
    cls2 = rotate(cls1, sh.shift_semitones)
    acc_shift = accuracy(cls2, ref)

    # stage 4: timing search with free transposition (upper bound)
    d, sh4 = timing_search(f0, times, segs, ref, tun.offset_cents,
                           args.search_ms, args.step_ms, mf)
    shifted = [(s + d / 1000.0, e + d / 1000.0) for s, e in segs]
    cls4, med4 = tc.segment_pitch_classes(f0, times, shifted, tun.offset_cents, mf)
    final = rotate(cls4, sh4.shift_semitones)
    acc_timing = accuracy(final, ref)

    # control: the same search against a randomly shuffled reference
    acc_control = float("nan")
    if not args.no_control:
        rng = np.random.default_rng(0)
        _, shc = timing_search(f0, times, segs, rng.permutation(ref), tun.offset_cents,
                               args.search_ms, args.step_ms, mf)
        acc_control = shc.agreement

    # error distribution and time-weighted accuracy of the last stage
    errors, acc_w = final_stats(final, ref, notes)

    voiced = float(np.mean(f0[note_frame_mask(times, segs)] > 0)) if len(f0) else float("nan")
    return Result(label, threshold, len(ref), int(np.sum(cls1 >= 0)), voiced, acc_raw,
                  acc_tun, acc_shift, acc_timing, acc_control, tun, sh.shift_semitones,
                  sh.agreement, d, sh4.shift_semitones, errors, acc_w, final, raw, med4)


# --------------------------------------------------------------------------- #
# Existing classifier (pitchnet) for comparison
# --------------------------------------------------------------------------- #
class Pitchnet:
    """The tool's classifier with the same preprocessing as the CLI
    (ultrastar_pitch.py: AudioPreprocessor(stride=128), sum of the probabilities
    per note, argmax; optionally StochasticPostprocessor)."""

    def __init__(self, model_path: str | None = None, threads: int = 0):
        try:
            from ultrastar_pitch.audio_preprocessor import AudioPreprocessor
            from ultrastar_pitch.stochastic_postprocessor import StochasticPostprocessor
        except ImportError as e:
            raise EvalError("ultrastar_pitch not importable (repo root needed)") from e
        import logging
        import onnxruntime as ort
        # empty/silent segments would otherwise log a warning per note (same behaviour as the tool)
        logging.getLogger().setLevel(logging.ERROR)
        if not model_path:
            model_path = str(HERE.parent / "ultrastar_pitch" / "binaries" / "pitchnet_2020_12_14.onnx")
        if not Path(model_path).is_file():
            raise EvalError(f"pitchnet model not found: {model_path}")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if threads > 0:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.pre = AudioPreprocessor(sr=SR, stride=128)
        self.post = StochasticPostprocessor()

    @staticmethod
    def to_int16(audio: np.ndarray) -> np.ndarray:
        """float audio -> int16 (only fallback/separated vocals; otherwise load_audio_int16)."""
        return np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)

    def predict(self, audio16: np.ndarray, segs) -> np.ndarray:
        """Pitch class per segment (0..11) without postprocessing; segment borders as in
        ProjectParser.process_audio."""
        feats, lens = [], []
        for s, e in segs:
            a = max(0, int(round(s * SR)))
            b = max(a, int(round(e * SR)))
            f = self.pre.transform(audio16[a:b]).astype(np.float32)
            feats.append(f)
            lens.append(len(f))
        prob = self.sess.run(None, {self.inp: np.vstack(feats)})[0]
        out = np.empty(len(lens), dtype=int)
        i0 = 0
        for k, n in enumerate(lens):
            out[k] = int(prob[i0:i0 + n].sum(axis=0).argmax())
            i0 += n
        return out

    def postprocess(self, classes: np.ndarray) -> np.ndarray:
        lst = [int(c) for c in classes]
        key = self.post.detect_key(lst)
        return np.array(self.post.correct_pitches(key, lst), dtype=int)


def analyze_pitchnet(label: str, pn: Pitchnet, audio16: np.ndarray, notes: Notes,
                     rm: Result, cache: dict | None = None) -> list[Result]:
    """pitchnet on the same audio source as the RMVPE result `rm`. Time offset from
    rm.offset_ms, rated on the notes that RMVPE rated at stage Timing*.
    Returns: [PN-<label>, PN-<label>+PP]. `cache` (offset -> classes) avoids duplicate
    predictions with several thresholds."""
    cache = {} if cache is None else cache
    ref = notes.ref_classes
    mask = rm.final_classes >= 0
    d = rm.offset_ms

    def pred(offset):
        if offset not in cache:
            cache[offset] = pn.predict(audio16, [(s + offset / 1000.0, e + offset / 1000.0)
                                                  for s, e in notes.segments])
        return cache[offset]

    base0, base1 = pred(0), pred(d)
    nan = float("nan")
    out = []
    for suffix, pp in (("", False), ("+PP", True)):
        c0 = pn.postprocess(base0) if pp else base0
        c1 = pn.postprocess(base1) if pp else base1
        m0, m1 = np.where(mask, c0, -1), np.where(mask, c1, -1)
        acc_raw = accuracy(m0, ref)
        sh = tc.estimate_shift_semitones(m0, ref)
        acc_shift = accuracy(rotate(m0, sh.shift_semitones), ref)
        sh4 = tc.estimate_shift_semitones(m1, ref)
        final = rotate(m1, sh4.shift_semitones)
        errors, acc_w = final_stats(final, ref, notes)
        out.append(Result(f"PN-{label}{suffix}", rm.threshold, len(ref), int(mask.sum()),
                          rm.voiced_share, acc_raw, nan, acc_shift, accuracy(final, ref), nan,
                          tc.TuningResult(nan, nan, 0, False), sh.shift_semitones, sh.agreement,
                          d, sh4.shift_semitones, errors, acc_w, final, c0,
                          np.full(len(ref), np.nan),
                          acc_all=accuracy(rotate(c1, sh4.shift_semitones), ref)))
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def pct(x: float) -> str:
    return "   -  " if x != x else f"{100 * x:5.1f}%"


def print_results(results: list[Result], args) -> None:
    print()
    print(f"{'Source':<14}{'Thr.':>6}{'Notes':>7}{'rated':>8}{'voiced':>8} |"
          f"{'raw':>7}{'+tuning':>8}{'+transp.':>9} |{'+Timing*':>9}{'control*':>11}{'all*':>8}")
    for r in results:
        print(f"{r.label:<14}{r.threshold:>6.2f}{r.n_notes:>7}{r.n_eval:>8}"
              f"{pct(r.voiced_share):>8} |{pct(r.acc_raw):>7}{pct(r.acc_tuning):>8}"
              f"{pct(r.acc_shift):>9} |{pct(r.acc_timing):>9}{pct(r.acc_control):>11}"
              f"{pct(r.acc_all):>8}")
    print("\nDetails (stage Timing*):")
    for r in results:
        t = r.tuning
        if t.strength != t.strength:
            tun = "n/a"
        elif t.reliable:
            tun = f"{t.offset_cents:+.0f} cents (R {t.strength:.2f})"
        else:
            tun = f"unreliable, 0 (R {t.strength:.2f})"
        e = r.errors
        print(f"  {r.label} @ {r.threshold:.2f}: tuning {tun} | transposition without timing "
              f"{r.shift:+d} ({pct(r.shift_agreement).strip()}), with timing {r.final_shift:+d} | "
              f"time offset {r.offset_ms:+d} ms")
        print(f"      error distance (semitones): 0: {pct(e[0]).strip()}, ±1: {pct(e[1]).strip()}, "
              f"±2: {pct(e[2]).strip()}, ≥3: {pct(e[3]).strip()} | time-weighted correct: "
              f"{pct(r.acc_weighted).strip()}")
    print("\n* Timing = best time offset search (±{0} ms) with free transposition, i.e. an "
          "upper bound.\n  control = the same search with a randomly shuffled reference "
          "(value the search alone yields).\n  Chance level without search: about 8 %."
          .format(args.search_ms))
    print("  time offset > 0: audio is later than the notes (add to #GAP), like "
          "offset_ms in label_quality.py.")
    if any(r.label.startswith("PN-") for r in results):
        print("  PN-* = existing classifier (pitchnet), +PP with postprocessing; Timing* with the "
              "offset of the RMVPE search,\n  rated on the same notes as RMVPE. all* = Timing* over "
              "all notes (including unvoiced ones).")


def write_csv(path: str, notes: Notes, results: list[Result]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "threshold", "start_s", "end_s", "file_pitch", "file_class",
                    "pred_class_raw", "pred_class_final", "midi_median_corrected"])
        for r in results:
            for i in range(len(notes.starts)):
                med = r.final_medians[i]
                w.writerow([r.label, f"{r.threshold:.2f}", f"{notes.starts[i]:.3f}",
                            f"{notes.ends[i]:.3f}", int(notes.pitches[i]), int(notes.pitches[i] % 12),
                            int(r.raw_classes[i]), int(r.final_classes[i]),
                            "" if med != med else f"{med:.2f}"])


# --------------------------------------------------------------------------- #
# Separate vocals (optional)
# --------------------------------------------------------------------------- #
def separate_vocals(audio_path: str, model_path: str, threads: int, gpu: bool = False) -> np.ndarray:
    try:
        from ultrastar_pitch.vocal_separator import VocalSeparator
    except ImportError:
        try:
            from vocal_separator import VocalSeparator           # located next to this file
        except ImportError as e:
            raise EvalError("vocal_separator.py not importable (repo root or "
                            "dev/ folder needed)") from e
    from math import gcd
    from scipy.signal import resample_poly
    wav = load_audio(audio_path, 44100, channels=2)
    sep = VocalSeparator(model_path, num_threads=threads, providers=providers_for(gpu))
    t0 = time.time()

    def progress(done, total):
        print(f"\r  separation: chunk {done}/{total} ({time.time() - t0:.0f} s)",
              end="", file=sys.stderr, flush=True)

    vocals = sep.separate(wav, progress)
    print(file=sys.stderr)
    mono = vocals.mean(axis=0)
    g = gcd(44100, SR)
    return resample_poly(mono, SR // g, 44100 // g).astype(np.float32)


def save_wav16(path: str, audio: np.ndarray) -> None:
    from scipy.io import wavfile
    wavfile.write(path, SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


# --------------------------------------------------------------------------- #
# Self test (synthetic data, no RMVPE model)
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    import tempfile
    rng = np.random.default_rng(1)
    bpm, n_notes = 120.0, 80
    beat_s = 15.0 / bpm                                     # 15000/BPM ms
    true_tuning, true_shift, true_offset = 30.0, 2, 300     # cents, semitones, ms
    pitches = rng.integers(-7, 12, n_notes)
    starts = np.cumsum(rng.integers(3, 6, n_notes))         # Beats
    durs = rng.integers(2, 4, n_notes)
    with tempfile.TemporaryDirectory() as d:
        txt = Path(d) / "notes.txt"
        lines = [f"#BPM:{bpm}".replace(".", ","), "#GAP:500"]
        for s, du, p in zip(starts, durs, pitches):
            lines.append(f": {s} {du} {p} la")
        lines.append("E")
        txt.write_text("\n".join(lines), encoding="utf-8")
        notes = parse_notes(str(txt))
    # synthetic F0: audio 300 ms later, 30 cents too high, file shifted by 2 semitones
    t_end = notes.ends[-1] + 2.0
    times = np.arange(0, t_end, FRAME_S)
    midi = np.full(times.shape, np.nan)
    for s, e, p in zip(notes.starts, notes.ends, notes.pitches):
        sel = (times >= s + true_offset / 1000) & (times < e + true_offset / 1000)
        midi[sel] = 60 + (p - true_shift) + true_tuning / 100 + rng.normal(0, 0.12, sel.sum()) \
            + rng.normal(0, 0.1)
    # outliers (instruments) outside the notes
    out = np.isnan(midi) & (rng.random(times.shape) < 0.3)
    midi[out] = rng.uniform(40, 80, out.sum())
    f0 = np.where(np.isfinite(midi), 440.0 * 2.0 ** ((midi - 69) / 12), 0.0)
    sal = np.zeros((len(times), N_CLASS))
    for i, f in enumerate(f0):                               # salience with peak in the matching bin
        if f > 0:
            b = (1200 * np.log2(f / 10.0) - CENTS_BASE) / CENTS_STEP
            if 0 <= b < N_CLASS:
                j = np.arange(N_CLASS)
                sal[i] = 0.9 * np.exp(-0.5 * ((j - b) / 1.0) ** 2)
    ns = argparse.Namespace(min_frames=3, tuning_all=False, min_strength=0.05, search_ms=500,
                            step_ms=20, no_control=False)
    r = analyze("Test", sal, notes, 0.3, ns)
    f0_t, _ = decode_salience(sal, 0.3)
    same_search = all(
        timing_search(f0_t, times, notes.segments, ref_, off, 500, 20, 3)
        == _timing_search_loop(f0_t, times, notes.segments, ref_, off, 500, 20, 3)
        for ref_ in (notes.ref_classes, rng.permutation(notes.ref_classes)) for off in (0.0, 30.0, -17.0))
    print("vectorised timing search == loop:", same_search)
    print(f"Parser: {len(notes.starts)} notes, first note at {notes.starts[0]:.3f} s "
          f"(expected {0.5 + starts[0] * beat_s:.3f} s)")
    print(f"Tuning {r.tuning.offset_cents:+.0f} cents (true {true_tuning:+.0f}), "
          f"time offset {r.offset_ms:+d} ms (true {true_offset:+d}), "
          f"transposition {r.final_shift:+d} (true {true_shift:+d})")
    print(f"raw {pct(r.acc_raw).strip()} | +tuning {pct(r.acc_tuning).strip()} | "
          f"+transp. {pct(r.acc_shift).strip()} | +Timing {pct(r.acc_timing).strip()} | "
          f"control {pct(r.acc_control).strip()}")
    ok = (abs(notes.starts[0] - (0.5 + starts[0] * beat_s)) < 1e-6
          and abs(r.tuning.offset_cents - true_tuning) < 8
          and abs(r.offset_ms - true_offset) <= 40 and r.final_shift == true_shift
          and r.acc_timing > 0.9 and r.acc_raw < 0.3 and r.acc_control < 0.5 and same_search)
    print("SELF TEST:", "passed" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Main program
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Chroma accuracy of RMVPE against an Ultrastar notes.txt")
    ap.add_argument("audio", nargs="?", help="audio file (any format), usually the mix")
    ap.add_argument("notes", nargs="?", help="notes.txt")
    ap.add_argument("--model", default=str(HERE.parent / "models" / "rmvpe.onnx"),
                    help="RMVPE ONNX (default models/rmvpe.onnx)")
    ap.add_argument("--vocals", help="additional file with isolated vocals")
    ap.add_argument("--separate", metavar="DEMUCS.onnx",
                    help="separate the vocals with a Demucs model (e.g. models/htdemucs_vocals.onnx), slow")
    ap.add_argument("--save-vocals", metavar="WAV", help="save the separated vocals as WAV")
    ap.add_argument("--thresholds", default="0.03,0.15,0.35",
                    help="voicing thresholds, comma-separated (default 0.03,0.15,0.35)")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="min. voiced frames per note (default 3)")
    ap.add_argument("--min-strength", type=float, default=0.05,
                    help="min. strength R of the tuning estimate (default 0.05)")
    ap.add_argument("--tuning-all", action="store_true",
                    help="estimate the tuning from all voiced frames, not only inside notes")
    ap.add_argument("--search-ms", type=int, default=2000, help="time offset search ± ms (2000)")
    ap.add_argument("--step-ms", type=int, default=20, help="step size ms (20)")
    ap.add_argument("--no-control", action="store_true", help="skip the control (random reference)")
    ap.add_argument("--include-duets", action="store_true")
    ap.add_argument("--pitchnet", nargs="?", const="", default=None, metavar="ONNX",
                    help="also run the existing classifier for comparison (without value: "
                         "ultrastar_pitch/binaries/pitchnet_2020_12_14.onnx)")
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime threads (0 = default)")
    ap.add_argument("--gpu", action="store_true",
                    help="run RMVPE (and --separate) on CUDA, fallback to CPU; pitchnet stays on the CPU")
    ap.add_argument("--csv", metavar="FILE", help="result per note as CSV")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.audio or not args.notes:
        ap.print_usage(sys.stderr)
        print("AUDIO and NOTES are required (or --selftest).", file=sys.stderr)
        return 2
    try:
        thresholds = [float(x) for x in args.thresholds.split(",")]
        notes = parse_notes(args.notes, args.include_duets)
        print(f"Notes: {len(notes.starts)} singable ({notes.n_skipped_special} F/R/G skipped), "
              f"BPM {notes.bpm:g}, beat {notes.beat_ms:.1f} ms ({'V2' if notes.v2 else 'V1/unversioned'}), "
              f"GAP {notes.gap_ms:g} ms{', duet' if notes.duet else ''}")
        model = Rmvpe(args.model, args.threads, args.gpu)
        pn = Pitchnet(args.pitchnet or None, args.threads) if args.pitchnet is not None else None

        sources: list[tuple] = [("Mix", load_audio(args.audio, SR), args.audio)]
        if args.vocals:
            sources.append(("Vocals", load_audio(args.vocals, SR), args.vocals))
        if args.separate:
            print("Separating vocals ...", file=sys.stderr)
            sep = separate_vocals(args.audio, args.separate, args.threads, args.gpu)
            if args.save_vocals:
                save_wav16(args.save_vocals, sep)
                print(f"Vocals saved: {args.save_vocals}")
            sources.append(("Separated", sep, None))

        results: list[Result] = []
        pn_results: list[Result] = []
        for label, audio, apath in sources:
            t0 = time.time()
            sal = model.salience(audio)
            print(f"RMVPE {label}: {len(audio) / SR:.0f} s Audio in {time.time() - t0:.1f} s")
            rm_results = [analyze(label, sal, notes, th, args) for th in thresholds]
            results.extend(rm_results)
            if pn is not None:
                t0, cache = time.time(), {}
                a16 = load_audio_int16(apath) if apath else Pitchnet.to_int16(audio)
                for rm in rm_results:
                    pn_results.extend(analyze_pitchnet(label, pn, a16, notes, rm, cache))
                print(f"pitchnet {label}: {time.time() - t0:.1f} s")
        results.extend(pn_results)
        print_results(results, args)
        if args.csv:
            write_csv(args.csv, notes, results)
            print(f"\nCSV: {args.csv}")
    except EvalError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
