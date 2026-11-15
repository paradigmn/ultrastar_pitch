#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_eval.py – end-to-end accuracy of DetectionPipeline (what the tool really writes)
==========================================================================================

Runs the real pipeline on a list of songs in several configurations and compares the
written pitch classes with the note file (exact / within one semitone / soft, after the
global transposition per song, timing of the file). Nothing is
cached: audio decoding, both models and the postprocessing run exactly as in the CLI.

    python dev/pipeline_eval.py dev/lists/rmvpe_batch_songs.txt dev/lists/rmvpe_batch_songs_2021plus.txt \\
        --root . [--limit 40] [--configs pitchnet,ensemble,ensemble+pp,vocals+pp]

Configurations: pitchnet (built-in classifier only, no postprocessing), pitchnet+oldpp (the
tool before 2.0), ensemble (pitchnet + RMVPE on the mix), ensemble+key (+ soft key prior only),
ensemble+pool (+ repetition pooling and soft key prior), ensemble+pp (+ the note context network
with a frame model = the default), vocals+key / vocals+pool / vocals+pp (mix and vocals, needs
#VOCALS or a saved vocals.wav; songs without are skipped for these configurations, no separation
is run). The suffix @file switches the timing correction off (the notes are analysed at the
timing of the file). Paired differences are printed against the first configuration, for every
X+pp against X+pool (or X+key) and for every X against X@file.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import rmvpe_eval as ev                                  # noqa: E402
import rmvpe_eval_batch as eb                            # noqa: E402
from ultrastar_pitch import tuning_correction as tc      # noqa: E402
from ultrastar_pitch import (AudioPreprocessor, DetectionPipeline, PitchClassifier, ProjectParser,  # noqa: E402
                             StochasticPostprocessor)

SIGMA = 1.32
SOFT_W = np.array([math.exp(-(d * d) / (2 * SIGMA * SIGMA)) for d in range(7)])
CONFIGS = ["pitchnet", "pitchnet+oldpp", "ensemble", "ensemble+key", "ensemble+pp", "vocals+key", "vocals+pp"]
LEN_EDGES = [0.06, 0.1, 0.2, 0.4, 0.8]
LEN_NAMES = ["<60", "60-100", "100-200", "200-400", "400-800", ">800"]


def run_config(pipe: DetectionPipeline, config: str, song: str, out: str):
    config, _, timing = config.partition("@")
    pipe.correct_timing = timing != "file"
    pipe.use_rmvpe = config not in ("pitchnet", "pitchnet+oldpp")
    separate = config.startswith("vocals")      # only songs with free vocals reach this point
    postproc = config.endswith(("+pp", "+key", "+pool")) and config != "pitchnet+oldpp"
    pipe.REPETITION_WEIGHT = 0.0 if config.endswith("+key") else DetectionPipeline.REPETITION_WEIGHT
    pipe.use_context_model = config.endswith("+pp")
    if config == "pitchnet+oldpp":
        # the behaviour before this change: hard key snapping of the argmax
        _, pred = pipe.transform(song, out, postproc=False, use_vocals=False)
        pred = np.array(StochasticPostprocessor.correct_pitches(StochasticPostprocessor.detect_key(pred.tolist()),
                                                                pred.tolist()))
        return pred
    return pipe.transform(song, out, postproc=postproc, use_vocals=separate)[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lists", nargs="+")
    ap.add_argument("--root")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--pitchnet-model", default=None,
                    help="onnx of the pitch classifier (default: the packaged pitchnet_2026_11_13; the old model is "
                    "ultrastar_pitch/binaries/pitchnet_2020_12_14.onnx; a frame model is recognised by its input name)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    configs = [c for c in args.configs.split(",") if c]
    songs = []
    for lst in args.lists:
        songs += eb.collect_songs([lst], args.root)
    if args.limit:
        songs = songs[:args.limit]
    pipe = DetectionPipeline(ProjectParser(), AudioPreprocessor(stride=128), PitchClassifier(args.pitchnet_model),
                             StochasticPostprocessor())
    print("pitch classifier:", pipe.pitch_classifier.model_name, "(frame model)" if pipe.pitch_classifier.frame_based else "")
    dist = {c: [] for c in configs}
    durs = {c: [] for c in configs}
    per_song = {c: [] for c in configs}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "notes_new.txt")
        for i, song in enumerate(songs, 1):
            try:
                notes = ev.parse_notes(song)
            except ev.EvalError as e:
                print(f"[{i}/{len(songs)}] skipped: {e}")
                continue
            ref = notes.ref_classes
            line = []
            for c in configs:
                try:
                    pipe.project_parser.load_note_file(song)
                    if c.startswith("vocals") and not pipe.project_parser.has_vocal_audio():
                        line.append(f"{c}: no vocals")
                        continue
                    pred = np.asarray(run_config(pipe, c, song, out))
                except Exception as e:                       # noqa: BLE001
                    line.append(f"{c}: error {type(e).__name__}: {str(e)[:60]}")
                    continue
                sh = tc.estimate_shift_semitones(pred, ref)
                d = ((pred + sh.shift_semitones) % 12 - ref) % 12
                d = np.minimum(d, 12 - d)
                dist[c].append(d)
                durs[c].append(notes.ends - notes.starts)
                per_song[c].append((song, float(np.mean(d == 0))))
                line.append(f"{c} {100 * np.mean(d == 0):.1f}%")
            print(f"[{i}/{len(songs)}] {Path(song).parent.name[:36]}: " + " | ".join(line), flush=True)
    print(f"\n{len(songs)} songs in {(time.time() - t0) / 60:.1f} min")
    print(f"{'configuration':<16}{'songs':>6}{'notes':>8} |{'exact':>8}{'±1':>8}{'soft':>8} |{'song mean':>10}{'>=80%':>7}"
          f" | exact by length (ms) " + " ".join(f"{n:>7}" for n in LEN_NAMES))
    for c in configs:
        if not dist[c]:
            continue
        d = np.concatenate(dist[c])
        b = np.digitize(np.concatenate(durs[c]), LEN_EDGES)
        acc = np.array([a for _, a in per_song[c]])
        print(f"{c:<16}{len(acc):>6}{len(d):>8} |{100 * np.mean(d == 0):7.1f}%{100 * np.mean(d <= 1):7.1f}%"
              f"{100 * np.mean(SOFT_W[d]):7.1f}% |{100 * acc.mean():9.1f}%{100 * np.mean(acc >= 0.8):6.1f}%"
              f" | {'':21}" + " ".join(f"{100 * np.mean(d[b == k] == 0):7.1f}" for k in range(len(LEN_NAMES))))
    # paired differences against the first configuration, and of every X+pp against X+pool (else X+key)
    pairs = [(configs[0], c) for c in configs[1:]]
    for c in configs:
        if c.endswith("+pp"):
            ref = next((c[:-3] + x for x in ("+pool", "+key") if c[:-3] + x in configs), None)
            if ref and ref != configs[0]:
                pairs.append((ref, c))
    pairs += [(c + "@file", c) for c in configs if c + "@file" in configs and configs[0] != c + "@file"]
    for base, c in pairs:
        bm = dict(per_song[base])
        both = [(bm[s], a) for s, a in per_song[c] if s in bm]
        if len(both) < 3:
            continue
        diff = np.array([b - a for a, b in both])
        rng = np.random.default_rng(0)
        m = diff[rng.integers(0, len(diff), (4000, len(diff)))].mean(axis=1)
        print(f"  {c} - {base}: {100 * diff.mean():+.2f} points [{100 * np.percentile(m, 2.5):+.2f} .. "
              f"{100 * np.percentile(m, 97.5):+.2f}] (n={len(diff)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
