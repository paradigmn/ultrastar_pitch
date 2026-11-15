#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file          label_quality.py
@brief         rate how well the notes of ultrastar song files match their audio
@author        paradigm

For every song a timing score and a pitch score (both 0..1) are calculated and written to a csv
file. There is no ground truth, so the notes are compared with an independent analysis of the
audio, which works on full mixes (numpy/scipy only):

    1. harmonic summation gives a pitch salience per frame (20 ms), the share of it, which lies
       on a pitch class (C = 0, like the ultrastar pitch modulo 12), is the evidence for a note
    2. for every time offset (+-max_offset) and every global transposition (24 half steps) the
       mean evidence for the labeled pitch classes inside the notes is calculated: A(offset, shift)
    3. the maximum of A gives the best alignment: offset = timing error, shift = transposition
    4. timing score = evidence for a clear alignment peak * small offset * stable offset over the
       song (a drifting offset indicates a wrong #BPM or a different audio version)
    5. pitch score = evidence at the best offset against the evidence of wrong transpositions
       (0 = chance level, 1 = perfect). pitch_score uses the labels as they are, pitch_score_aligned
       corrects a global transposition (shift_semitones) first.

Columns of the csv:
    offset_ms          audio is later than the notes (add it to #GAP to fix the song), accuracy ~40 ms
    shift_semitones    correction to add to all pitches, if the song is transposed
    drift_ms_per_min   change of the offset over the song (wrong #BPM or other audio version)
    voiced_ratio       share of loud frames inside the notes, low values: the analysis is unreliable

Files with "instrumental" or "karaoke" in their name are skipped (--skip-names), they have no vocals
to compare. --exclude takes a list of further files (see dev/make_exclude_list.py).

Duets (songs with voice changes P1, P2, ...) are skipped (status skipped_duet), because two voices
sing at the same time and a single pitch evidence can't follow both. --include-duets rates them anyway.

The timing evidence is the pitch pattern itself, so with (nearly) random pitches the timing can't
be assessed either. The analysis is much better with a vocal track (#VOCALS or --separate) than
with a full mix; `source` tells which one was used. The scores are meant for ranking and
pre-sorting, not as absolute truth, check the thresholds by listening to some songs.

Usage:
    python dev/label_quality.py /path/to/songs -o dev/results/scores.csv [--workers 8] [--resume]
    python dev/label_quality.py --list songs.txt -o scores2.csv --separate --vocal-model m.onnx
"""
import os

# one thread per process: parallelism comes from the worker processes
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import logging
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sfft

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultrastar_pitch.project_parser import AudioError, ProjectParser, ProjectParserError  # noqa: E402

SAMPLE_RATE = 16000
HOP_S = 0.02  # time resolution of the F0 track
TOLERANCE = 0.75  # semitones around the labeled pitch, which count as the same pitch
MIN_VOICED_FRAMES = 2  # loud frames a note needs to take part
MIN_NOTES = 20  # notes needed for a score
DEFAULT_SKIP_WORDS = "instrumental,karaoke"  # file names with these words are never analysed

FIELDS = [
    "path", "status", "quality_score", "timing_score", "pitch_score", "pitch_score_aligned",
    "offset_ms", "drift_ms_per_min", "local_offset_spread_ms", "valid_groups", "shift_semitones",
    "timing_confidence", "pitch_hit_rate", "pitch_chance_level", "voiced_ratio", "notes_valid",
    "n_notes", "voices", "notes_beyond_audio", "short_notes", "duration_s", "bpm", "version", "source",
    "flags", "seconds", "message",
]


# ------------------------------------------------------------------------------------ evidence
def pitch_class(f0: np.ndarray) -> np.ndarray:
    """ pitch class in semitones in [0, 12), C = 0 (nan for unvoiced frames) """
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.mod(12.0 * np.log2(f0 / 261.6256), 12.0)


def _class_matrix(freqs: np.ndarray) -> np.ndarray:
    """ (bins, 24) 0/1 matrix: F0 bin belongs to the half step class g / 2 (within the tolerance) """
    classes = pitch_class(freqs)
    dist = np.abs(np.mod(classes[:, None] - (np.arange(24) / 2.0)[None, :] + 6.0, 12.0) - 6.0)
    return (dist <= TOLERANCE).astype(np.float32)


class SalienceAnalyzer:
    """ multi pitch evidence for mixes: harmonic summation salience, which share of it lies on
        the pitch class g / 2? (works while accompaniment is playing, a tracker would follow
        the bass or the chords) """

    def __init__(self, sr=SAMPLE_RATE, n_fft=2048, hop=int(SAMPLE_RATE * HOP_S), fmin=65.406,
                 fmax=1046.5, bins_per_semitone=4, harmonics=8, decay=0.8, sharpen=2.0):
        self.sr, self.n_fft, self.hop, self.sharpen = sr, n_fft, hop, sharpen
        n_bins = int(round(12 * math.log2(fmax / fmin) * bins_per_semitone)) + 1
        self.freqs = fmin * 2.0 ** (np.arange(n_bins) / (12.0 * bins_per_semitone))
        pos = np.outer(np.arange(1, harmonics + 1), self.freqs) * n_fft / sr  # fft bin of every harmonic
        self._lo = np.minimum(np.floor(pos).astype(int), n_fft // 2 - 1)
        self._frac = (pos - np.floor(pos)).astype(np.float32)
        self._weight = (decay ** np.arange(harmonics)).astype(np.float32)[:, None]
        self._classes = _class_matrix(self.freqs)
        self._window = np.hanning(n_fft).astype(np.float32)

    def evidence(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ returns (frames, 24) evidence and (frames,) weight; frame l is centered at
            (l * hop + n_fft / 2) / sr """
        y = np.asarray(y, dtype=np.float32)
        if len(y) < self.n_fft:
            return np.zeros((0, 24), np.float32), np.zeros(0, np.float32)
        windows = sliding_window_view(y, self.n_fft)[:: self.hop]
        evid, energy = [], []
        for i in range(0, len(windows), 512):
            frames = windows[i: i + 512] * self._window
            mag = np.abs(sfft.rfft(frames, axis=1)).astype(np.float32)
            energy.append(np.sqrt((frames ** 2).mean(axis=1)))
            # whiten: keep only what stands out of the local spectral envelope (harmonic peaks)
            env = _smooth(mag, 41)
            white = np.maximum(mag / (env + 1e-6) - 1.0, 0.0)
            sal = ((1 - self._frac) * white[:, self._lo] + self._frac * white[:, self._lo + 1]) * self._weight
            sal = sal.sum(axis=1) ** self.sharpen  # (frames, bins)
            sal /= sal.sum(axis=1, keepdims=True) + 1e-9
            evid.append(sal @ self._classes)
        energy = np.concatenate(energy)
        loud = energy > 0.01 * (np.percentile(energy, 95) + 1e-9)  # -40 dB below the loud parts
        return np.concatenate(evid).astype(np.float32), loud.astype(np.float32)


def _smooth(x: np.ndarray, size: int) -> np.ndarray:
    """ moving average along the last axis """
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, size, axis=-1, mode="nearest")


# ------------------------------------------------------------------------------------- scoring
def _peak(values: np.ndarray) -> Tuple[int, float, float, float]:
    """ index of the maximum, its height above the median and robust z-score (nan ignored) """
    valid = values[~np.isnan(values)]
    idx = int(np.nanargmax(values))
    median = float(np.median(valid))
    sigma = 1.4826 * float(np.median(np.abs(valid - median))) + 1e-3
    return idx, float(values[idx]) - median, (float(values[idx]) - median) / sigma, median


def evaluate(starts: np.ndarray, ends: np.ndarray, pitches: np.ndarray, evidence: np.ndarray,
             weight: np.ndarray, t_first_frame: float, max_offset: float = 2.0) -> Dict[str, float]:
    """ compare notes (seconds, pitch classes) with frame evidence: evidence[l, g] in [0, 1] says
        how well frame l supports the pitch class g / 2, weight[l] is 1 for frames which can be
        judged (voiced / loud), t_first_frame is the center time of frame 0 """
    n_frames = len(weight)
    order = np.argsort(starts, kind="stable")
    starts, ends, pitches = starts[order], ends[order], pitches[order].astype(int) % 12
    n_notes = len(starts)
    hit_cs = np.zeros((24, n_frames + 1), dtype=np.float64)
    hit_cs[:, 1:] = np.cumsum(evidence.T.astype(np.float64), axis=1)
    voiced_cs = np.concatenate([[0.0], np.cumsum(weight.astype(np.float64))])
    # the central part of every note, transitions at the borders are unreliable
    dur = ends - starts
    trim = np.where(dur >= 0.1, 0.2 * dur, 0.0)
    first = np.ceil((starts + trim - t_first_frame) / HOP_S).astype(int)
    last = np.maximum(np.floor((ends - trim - t_first_frame) / HOP_S).astype(int) + 1, first + 1)
    n_lags = int(round(max_offset / HOP_S))
    lags = np.arange(-n_lags, n_lags + 1)
    idx_a = np.clip(first[:, None] + lags[None, :], 0, n_frames)
    idx_b = np.clip(last[:, None] + lags[None, :], 0, n_frames)
    n_voiced = voiced_cs[idx_b] - voiced_cs[idx_a]
    valid = n_voiced >= MIN_VOICED_FRAMES
    count = valid.sum(axis=0)
    usable = count >= max(MIN_NOTES, 0.7 * count.max())
    if not usable.any():
        return {"status": "too_few_notes", "notes_valid": int(count.max())}
    agree = np.empty((24, n_notes, len(lags)), dtype=np.float32)
    grid = np.full((24, len(lags)), np.nan, dtype=np.float32)
    for shift in range(24):
        cls = (2 * pitches + shift) % 24
        hits = hit_cs[cls[:, None], idx_b] - hit_cs[cls[:, None], idx_a]
        agree[shift] = np.where(valid, hits / np.maximum(n_voiced, 1), np.nan)
        grid[shift] = np.where(usable, np.where(valid, agree[shift], 0.0).sum(axis=0) / np.maximum(count, 1), np.nan)
    shift_best, lag_best = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
    peak = float(grid[shift_best, lag_best])
    offset = float(lags[lag_best] * HOP_S)

    # timing: how clearly does the alignment stand out against other time offsets?
    _, contrast, z_score, _ = _peak(grid[shift_best])
    confidence = min(float(np.clip((z_score - 4.0) / 6.0, 0, 1)), float(np.clip(contrast / 0.2, 0, 1)))
    # stability of the offset over the song
    n_groups = int(np.clip(n_notes // 40, 1, 8))
    local_offsets, centers_t = [], []
    for group in np.array_split(np.arange(n_notes), n_groups) if n_groups > 1 else []:
        sub = agree[shift_best][group]
        n_ok = np.sum(~np.isnan(sub), axis=0)
        ok = n_ok >= max(8, 0.7 * n_ok.max())
        if ok.sum() < len(lags) // 2:
            continue
        curve = np.where(ok, np.nansum(sub, axis=0) / np.maximum(n_ok, 1), np.nan)
        g_idx, g_contrast, g_z, _ = _peak(curve)
        if g_z >= 3.5 and g_contrast >= 0.1:
            local_offsets.append(lags[g_idx] * HOP_S)
            centers_t.append(float(starts[group].mean()))
    spread, drift = 0.0, 0.0
    if len(local_offsets) >= 2:
        spread = 1.4826 * float(np.median(np.abs(np.array(local_offsets) - offset)))
    if len(local_offsets) >= 3:
        drift = float(np.polyfit(np.array(centers_t) / 60.0, np.array(local_offsets) * 1000.0, 1)[0])
    valid_frac = len(local_offsets) / n_groups if n_groups > 1 else 1.0
    s_off = 1.0 / (1.0 + (abs(offset) / 0.15) ** 2)
    s_stab = 1.0 / (1.0 + (spread / 0.1) ** 2) * (0.5 + 0.5 * valid_frac)
    timing_score = confidence * s_off * s_stab

    # pitch: hit rate at the best offset against wrong transpositions
    column = grid[:, lag_best]
    chance = float(np.median(column))
    score = lambda hit: float(np.clip((hit - chance) / max(1.0 - chance, 1e-6), 0, 1))
    hit_abs = float(column[0])
    voiced_in_notes = float((voiced_cs[idx_b[:, lag_best]] - voiced_cs[idx_a[:, lag_best]]).sum()
                            / max(1, (idx_b[:, lag_best] - idx_a[:, lag_best]).sum()))
    shift_st = (shift_best / 2.0 + 6.0) % 12.0 - 6.0
    return {
        "status": "ok", "timing_score": timing_score, "pitch_score": score(hit_abs),
        "pitch_score_aligned": score(peak), "offset_ms": offset * 1000.0, "drift_ms_per_min": drift,
        "local_offset_spread_ms": spread * 1000.0, "valid_groups": f"{len(local_offsets)}/{n_groups}",
        "shift_semitones": shift_st, "timing_confidence": confidence, "pitch_hit_rate": hit_abs,
        "pitch_chance_level": chance, "voiced_ratio": voiced_in_notes, "notes_valid": int(count[lag_best]),
    }


def make_flags(row: dict, source: Optional[str]) -> str:
    """ short hints why a song got a low score """
    flags = []
    if row.get("status") != "ok":
        return ""
    if row["timing_confidence"] < 0.2:
        flags.append("no_alignment_found")
    if abs(row["offset_ms"]) > 250:
        flags.append("timing_offset")
    if abs(row["drift_ms_per_min"]) > 300 or row["local_offset_spread_ms"] > 150:
        flags.append("timing_drift")
    if abs(row["shift_semitones"]) >= 0.5 and row["pitch_score_aligned"] - row["pitch_score"] > 0.2:
        flags.append("transposed")
    if row["voiced_ratio"] < 0.3:
        flags.append("low_voicing")
    if row.get("voices", 1) > 1:
        flags.append("duet")
    if row["notes_beyond_audio"] > 0.1:
        flags.append("notes_beyond_audio")
    if source not in ("#VOCALS", "vocals.wav", "separated"):
        flags.append("mix_analysis")
    return ";".join(flags)


# --------------------------------------------------------------------------------- song worker
_SEPARATOR = None


def _get_separator(options: dict):
    """ load the vocal separation model once per process """
    global _SEPARATOR
    if options.get("vocal_model") and _SEPARATOR is None:
        from ultrastar_pitch.vocal_separator import VocalSeparator
        _SEPARATOR = VocalSeparator(options["vocal_model"], num_threads=options.get("threads", 0))
    return _SEPARATOR


def analyse_song(job: Tuple[str, dict]) -> dict:
    """ analyse a single song, never raises (problems end up in the status column) """
    path, options = job
    logging.getLogger("ultrastar_pitch").setLevel(logging.ERROR)
    start_time = time.time()
    row = {"path": path, "status": "ok"}
    try:
        parser = ProjectParser()
        parser.load_note_file(path)
        notes = parser.singable
        row.update(n_notes=len(notes), bpm=parser.meta["#BPM"], voices=len(parser.voices),
                   version=".".join(map(str, parser.version)) if parser.version else "unversioned")
        if parser.is_duet and not options.get("include_duets", False):
            row["status"] = "skipped_duet"  # two voices at the same time can't be rated this way
            row["seconds"] = round(time.time() - start_time, 1)
            return row
        use_file = options.get("use_vocals_file", True)
        parser.vocals_sample_rate = options.get("vocals_rate", 16000)
        separator = None
        if options.get("vocal_model") and not parser.has_vocal_audio(use_file):
            separator = _get_separator(options)  # the model is only loaded, if a song needs it
        samples = parser.load_audio(SAMPLE_RATE, separator=separator, use_vocals_file=use_file)
        row.update(source=parser.audio_source, duration_s=round(len(samples) / SAMPLE_RATE, 1))
        starts = np.array([n["t_start"] for n in notes]) / 1000.0
        ends = np.array([n["t_end"] for n in notes]) / 1000.0
        pitches = np.array([n["pitch"] for n in notes])
        duration = len(samples) / SAMPLE_RATE
        row["notes_beyond_audio"] = round(float(np.mean(starts >= duration)), 3)
        row["short_notes"] = round(float(np.mean(ends - starts < 0.06)), 3)
        # only the part of the song with notes (plus the search range) is analysed
        margin = options["max_offset"] + 1.0
        begin = max(0.0, starts.min() - margin)
        stop = min(duration, ends.max() + margin)
        segment = samples[int(begin * SAMPLE_RATE): int(stop * SAMPLE_RATE)].astype(np.float32) / 32768.0
        analyzer = SalienceAnalyzer()
        evidence, weight = analyzer.evidence(segment)
        if len(weight) == 0:
            row["status"] = "audio_too_short"
        else:
            t_first = begin + analyzer.n_fft / 2.0 / SAMPLE_RATE
            result = evaluate(starts, ends, pitches, evidence, weight, t_first, options["max_offset"])
            row.update(result)
            if result["status"] == "ok":
                row["quality_score"] = math.sqrt(result["timing_score"] * result["pitch_score"])
                row["flags"] = make_flags(row, parser.audio_source)
    except ProjectParserError as err:
        row["status"] = ("audio_error:" if isinstance(err, AudioError) else "parse_error:") + type(err).__name__
        row["message"] = str(err)[:200]
    except Exception as err:  # keep the batch running
        row["status"] = "error:" + type(err).__name__
        row["message"] = str(err)[:200]
    row["seconds"] = round(time.time() - start_time, 1)
    return row


# ------------------------------------------------------------------------------------ batch run
def compile_skip_words(value: str) -> List["re.Pattern"]:
    """ "instrumental,karaoke,=inst" -> patterns for the file name (case-insensitive).
        A word matches anywhere in the name, a word with a leading "=" only as a whole word
        (separated by anything but letters and digits), e.g. =inst matches "trixie-inst.txt",
        but not "Against The Wall.txt" """
    patterns = []
    for word in value.split(","):
        word = word.strip().lower()
        if not word:
            continue
        if word.startswith("="):
            patterns.append(re.compile(r"(?<![^\W_])" + re.escape(word[1:]) + r"(?![^\W_])"))
        else:
            patterns.append(re.compile(re.escape(word)))
    return patterns


def is_skipped(path: str, patterns: Sequence["re.Pattern"]) -> bool:
    """ True, if the file name (not the folder) matches one of the skip words """
    name = os.path.basename(path).lower()
    return any(pattern.search(name) for pattern in patterns)


def read_exclude_file(path: str, base: str) -> set:
    """ paths to ignore, one per line (relative ones to base), empty lines and # comments are ignored """
    excluded = set()
    with open(path, encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                excluded.add(os.path.normcase(os.path.abspath(os.path.join(base, line))))
    return excluded


def looks_like_song(path: str) -> bool:
    """ ultrastar note files contain a #BPM header at the beginning """
    try:
        with open(path, "rb") as file:
            return b"#BPM" in file.read(8192).upper()
    except OSError:
        return False


def walk_txt_files(root: str) -> Iterator[str]:
    """ all .txt files below root """
    for folder, _, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith(".txt"):
                yield os.path.abspath(os.path.join(folder, name))


def select_songs(entries: Iterator[str], patterns: Sequence["re.Pattern"], excluded: set) -> Tuple[List[str], Dict[str, int]]:
    """ filter candidate files, returns the songs and how many files were dropped for which reason """
    songs, stats = [], {"by_name": 0, "by_exclude_list": 0, "no_bpm_header": 0}
    for path in entries:
        if is_skipped(path, patterns):
            stats["by_name"] += 1
        elif os.path.normcase(path) in excluded:
            stats["by_exclude_list"] += 1
        elif not looks_like_song(path):
            stats["no_bpm_header"] += 1
        else:
            songs.append(path)
    return songs, stats


def run_jobs(jobs: List[Tuple[str, dict]], workers: int) -> Iterator[dict]:
    """ analyse songs in worker processes, a crashing song doesn't stop the batch """
    chunk = max(16, workers * 16)
    for i in range(0, len(jobs), chunk):
        part, done = jobs[i: i + chunk], set()
        try:
            with ProcessPoolExecutor(workers) as pool:
                futures = {pool.submit(analyse_song, job): job[0] for job in part}
                for future in as_completed(futures):
                    row = future.result()  # raises, if the worker died: the song isn't done then
                    done.add(futures[future])
                    yield row
        except BrokenProcessPool:  # find the song, which kills its worker
            for job in part:
                if job[0] in done:
                    continue
                try:
                    with ProcessPoolExecutor(1) as pool:
                        yield pool.submit(analyse_song, job).result()
                except BrokenProcessPool:
                    yield {"path": job[0], "status": "error:WorkerCrash", "message": "analysis process died"}


def main():
    parser = argparse.ArgumentParser(description="rate timing and pitch integrity of ultrastar songs")
    parser.add_argument("root", nargs="?", help="folder with songs (searched recursively)")
    parser.add_argument("--list", help="text file with one note file per line (relative ones to the folder, "
                                       "if given, else to the current directory) instead of searching the folder")
    parser.add_argument("-o", "--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "scores.csv"))
    parser.add_argument("--workers", type=int,
                        help="analysis processes (default: all cpu cores, but 1 with --separate, where onnxruntime "
                             "already uses all cores and every process would load its own model)")
    parser.add_argument("--resume", action="store_true", help="skip songs already in the output file")
    parser.add_argument("--limit", type=int, help="only analyse the first N songs (trial run)")
    parser.add_argument("--include-duets", action="store_true",
                        help="rate duets too (skipped by default, the scores are unreliable)")
    parser.add_argument("--skip-names", default=DEFAULT_SKIP_WORDS,
                        help="comma separated words, files with one of them in the file name are skipped, "
                             "=word only matches a whole word, e.g. '=inst' (default: %(default)s, empty: none)")
    parser.add_argument("--exclude", help="text file with paths to skip (relative ones to the folder), one per line")
    parser.add_argument("--max-offset", type=float, default=2.0, help="search range for timing errors in s")
    parser.add_argument("--separate", action="store_true", help="isolate vocals first (very slow on cpu)")
    parser.add_argument("--vocal-model", help="onnx model for --separate")
    parser.add_argument("--ignore-vocals-file", action="store_true",
                        help="don't use or write vocals.wav next to the note files")
    parser.add_argument("--vocals-rate", type=int, default=16000,
                        help="sample rate of the saved vocals.wav, mono 16 bit (default: %(default)s, about 8 MB "
                             "per 4 minutes, 44100 needs about 21 MB)")
    args = parser.parse_args()
    if not (args.root or args.list):
        parser.error("give a folder or --list")
    if args.separate and not args.vocal_model:
        parser.error("--separate needs --vocal-model")
    if args.vocals_rate <= 0:
        parser.error("--vocals-rate has to be positive")
    workers = args.workers or (1 if args.separate else os.cpu_count() or 1)
    options = {"max_offset": args.max_offset, "include_duets": args.include_duets,
               "vocal_model": args.vocal_model if args.separate else None,
               "use_vocals_file": not args.ignore_vocals_file, "vocals_rate": args.vocals_rate}
    base = os.path.abspath(args.root or os.getcwd())
    patterns = compile_skip_words(args.skip_names)
    excluded = read_exclude_file(args.exclude, base) if args.exclude else set()
    if args.list:
        with open(args.list, encoding="utf-8", errors="replace") as file:
            entries = (os.path.abspath(os.path.join(base, line.strip())) for line in file if line.strip())
            songs, stats = select_songs(entries, patterns, excluded)
    else:
        songs, stats = select_songs(walk_txt_files(base), patterns, excluded)
    print("not analysed: {by_name} files because of --skip-names, {by_exclude_list} because of --exclude, "
          "{no_bpm_header} without #BPM header".format(**stats), flush=True)
    known = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as file:
            known = {row["path"] for row in csv.DictReader(file)}
    songs = [s for s in songs if s not in known][: args.limit]
    print("{0} songs to analyse ({1} already done)".format(len(songs), len(known)), flush=True)
    new_file = not (args.resume and os.path.exists(args.output))
    started, done = time.time(), 0
    with open(args.output, "a" if not new_file else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        try:
            for row in run_jobs([(s, options) for s in songs], max(1, workers)):
                row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                writer.writerow(row)
                file.flush()
                done += 1
                if done % 20 == 0 or done == len(songs):
                    rate = (time.time() - started) / done
                    print("{0}/{1} songs, {2:.1f} s/song, remaining about {3:.0f} min".format(
                        done, len(songs), rate, rate * (len(songs) - done) / 60), flush=True)
        except KeyboardInterrupt:
            print("\ninterrupted, rerun with --resume to continue", flush=True)


if __name__ == "__main__":
    main()
