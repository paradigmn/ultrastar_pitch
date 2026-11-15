#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          pitchnet_val_compare.py
@brief         paired per-song comparison of trained pitchnet_2026 runs on the validation songs (ONNX, package path)
@author        paradigm

pitchnet_train.py picks the best epoch of one run by the pooled exact accuracy on the validation songs. To choose
between runs whose numbers are close, this script runs the exported model.onnx of every run through the package
code (ultrastar_pitch.pitch_classifier: power_spectrogram, predict_power, frame_scores_per_note), on the mix and
the vocals of every validation song, and on the mix resampled by about +-0.29 semitones (detuned songs, e.g. 432 Hz
tuning is -0.32 semitones; the labels keep their class, the note times are scaled). Per run: exact and within-one
accuracy (pooled over the notes, as in training), the share of the errors that are one semitone off, and the
difference to the first run with a bootstrap interval over the songs (songs resampled, pooled accuracy).

    python dev/pitchnet_val_compare.py base near near_aug > dev/results/pitchnet_runs/val_compare.txt

Also writes dev/results/pitchnet_runs/val_compare.csv (one row per song, run and condition).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dev"))
from ultrastar_pitch.pitch_classifier import PitchClassifier, power_spectrogram  # noqa: E402
from pitchnet_train import CACHE, read_selection  # noqa: E402

RUNS = ROOT / "dev" / "results" / "pitchnet_runs"
# (name, source, up, down): resample_poly(x, up, down) played at 16 kHz shifts the pitch by 12 * log2(down / up)
CONDITIONS = (("mix", "mix", 1, 1), ("vocals", "vocals", 1, 1),
              ("mix +0.29 st", "mix", 1000, 1017), ("mix -0.29 st", "mix", 1017, 1000))


def evaluate(scores: np.ndarray, classes: np.ndarray):
    pred = scores.argmax(axis=1)
    dist = np.abs((pred - classes + 6) % 12 - 6)
    return int((dist == 0).sum()), int((dist <= 1).sum()), len(classes)


def bootstrap(h_a, h_b, n, rng, samples=4000):
    """ difference of the pooled accuracies b - a with a 95 % interval, songs resampled """
    idx = rng.integers(0, len(n), (samples, len(n)))
    d = h_b[idx].sum(axis=1) / n[idx].sum(axis=1) - h_a[idx].sum(axis=1) / n[idx].sum(axis=1)
    return h_b.sum() / n.sum() - h_a.sum() / n.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names in dev/results/pitchnet_runs (the first is the reference)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N validation songs")
    ap.add_argument("--csv", default=str(RUNS / "val_compare.csv"))
    args = ap.parse_args()
    models = {r: PitchClassifier(str(RUNS / r / "model.onnx")) for r in args.runs}
    rows = read_selection("val", args.limit)
    # results[condition][run] = list of (hits, near, notes) per song
    results = {c[0]: {r: [] for r in args.runs} for c in CONDITIONS}
    out_rows = []
    t0 = time.time()
    for i, row in enumerate(rows):
        z = np.load(CACHE / (row["id"] + ".labels.npz"))
        segments, classes = z["segments"], z["classes"]
        audio = {s: np.load(CACHE / "{0}.{1}.npy".format(row["id"], s)).astype(np.float32) / 32768.0 for s in ("mix", "vocals")}
        for name, source, up, down in CONDITIONS:
            x = audio[source] if up == down else resample_poly(audio[source], up, down).astype(np.float32)
            power = power_spectrogram(x)
            seg = segments * (up / down)
            for r, clf in models.items():
                logprob, _ = clf.predict_power(power)
                h, near, n = evaluate(clf.frame_scores_per_note(logprob, seg), classes)
                results[name][r].append((h, near, n))
                out_rows.append({"id": row["id"], "run": r, "condition": name, "exact": h, "within1": near, "notes": n})
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print("[{0}/{1}] {2:.0f} s".format(i + 1, len(rows), time.time() - t0), file=sys.stderr, flush=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as file:
        w = csv.DictWriter(file, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    rng = np.random.default_rng(0)
    ref = args.runs[0]
    print("{0} validation songs, {1} notes; models: {2}".format(
        len(rows), sum(n for _, _, n in results["mix"][ref]), ", ".join(args.runs)))
    for name, *_ in CONDITIONS:
        print("\n" + name)
        print("  {0:10s} {1:>7s} {2:>9s} {3:>13s}   {4}".format("run", "exact", "within 1", "1 st of errs", "exact - " + ref + " [95 % over songs]"))
        a = np.array(results[name][ref], dtype=float)
        for r in args.runs:
            b = np.array(results[name][r], dtype=float)
            h, near, n = b.sum(axis=0)
            line = "  {0:10s} {1:7.2%} {2:9.2%} {3:13.1%}".format(r, h / n, near / n, (near - h) / max(n - h, 1))
            if r != ref:
                d, lo, hi = bootstrap(a[:, 0], b[:, 0], b[:, 2], rng)
                line += "   {0:+.2f} [{1:+.2f} .. {2:+.2f}] points".format(100 * d, 100 * lo, 100 * hi)
            print(line)
    print("\n{0:.0f} s".format(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
