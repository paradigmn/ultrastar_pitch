#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          tuning_survey.py
@brief         how far songs of the database are off the 440 Hz grid (RMVPE tuning estimate of the voice)
@author        paradigm

The frame models of pitchnet_2026 can be trained with or without the tuning augmentation (resampling by up to
+-0.35 semitones). Without it they are better on in-tune songs, with it on detuned ones (dev/pitchnet_val_compare.py).
Which matters more depends on how often songs are detuned, which this script measures on a random sample of the
songs in scores.csv (status ok, without the evaluation songs), with the tuning estimate the pipeline uses for
RMVPE (rmvpe.estimate_tuning: circular mean of the voice f0 inside the notes, cents off the 440 Hz grid). The
sample is not filtered by label quality, unlike the training selection, so it also shows whether that filter
removed detuned songs.

    .venv/bin/python dev/tuning_survey.py --n 400 > dev/results/tuning_survey.txt

Writes dev/results/tuning_survey.csv (path, split in the pitchnet selection or "none", offset, strength, reliable).
"""

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import rmvpe_eval_batch as eb                            # noqa: E402
from ultrastar_pitch import ProjectParser                # noqa: E402
from ultrastar_pitch import rmvpe as rmvpe_mod           # noqa: E402

THRESHOLDS = (10, 15, 20, 25, 30, 40)


def measure(path, rmvpe):
    parser = ProjectParser()
    parser.load_note_file(path)
    mix = parser.load_main_audio(rmvpe_mod.SR)
    sal = rmvpe.salience(mix.astype(np.float32) / 32768.0)
    return rmvpe_mod.estimate_tuning(sal, parser.note_segments(), 0.15)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scores", default=str(ROOT / "dev" / "results" / "scores.csv"))
    ap.add_argument("--csv", default=str(ROOT / "dev" / "results" / "tuning_survey.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    lists = [str(ROOT / "dev" / "lists" / n) for n in ("rmvpe_batch_songs.txt", "rmvpe_batch_songs_2021plus.txt")]
    eval_songs = {str(Path(p).resolve()) for p in eb.collect_songs(lists, str(ROOT))}
    with open(ROOT / "dev" / "lists" / "pitchnet" / "selection.csv", newline="", encoding="utf-8") as file:
        split = {str(Path(r["path"]).resolve()): r["split"] for r in csv.DictReader(file)}
    with open(args.scores, newline="", encoding="utf-8") as file:
        paths = [str(Path(r["path"]).resolve()) for r in csv.DictReader(file) if r["status"] == "ok"]
    paths = sorted(p for p in paths if p not in eval_songs)
    rng = np.random.default_rng(args.seed)
    sample = [paths[i] for i in sorted(rng.choice(len(paths), size=min(args.n, len(paths)), replace=False))]
    rmvpe = rmvpe_mod.Rmvpe(rmvpe_mod.default_model_path())
    t0 = time.time()
    rows = []

    def job(path):
        try:
            return path, measure(path, rmvpe), None
        except Exception as e:                            # noqa: BLE001
            return path, None, "{0}: {1}".format(type(e).__name__, str(e)[:60])

    with ThreadPoolExecutor(args.workers) as pool:
        for i, (path, t, err) in enumerate(pool.map(job, sample), 1):
            if err:
                print("[{0}/{1}] error {2} {3}".format(i, len(sample), Path(path).parent.name[:40], err), file=sys.stderr)
                continue
            rows.append({"path": path, "split": split.get(path, "none"), "offset_cents": round(t.offset_cents, 1),
                         "strength": round(t.strength, 3), "reliable": int(t.reliable)})
            if i % 50 == 0:
                print("[{0}/{1}] {2:.0f} s".format(i, len(sample), time.time() - t0), file=sys.stderr, flush=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as file:
        w = csv.DictWriter(file, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("{0} of {1} songs (status ok, without the evaluation songs) sampled, {2} measured, {3} reliable "
          "tuning estimates, {4:.0f} s".format(len(sample), len(paths), len(rows), sum(r["reliable"] for r in rows),
                                               time.time() - t0))
    groups = {"all": rows, "train/val (quality filtered)": [r for r in rows if r["split"] in ("train", "val")],
              "not selected": [r for r in rows if r["split"] not in ("train", "val")]}
    print("share of the songs with a reliable estimate whose |offset| is at least ... cents (unreliable: offset 0)")
    print("  {0:30s} {1:>6s} {2:>10s}".format("group", "songs", "median |o|") + "".join("{0:>7d}".format(t) for t in THRESHOLDS))
    for name, g in groups.items():
        o = np.abs([r["offset_cents"] for r in g if r["reliable"]])
        if len(o) == 0:
            continue
        print("  {0:30s} {1:6d} {2:10.1f}".format(name, len(o), float(np.median(o))) +
              "".join("{0:6.1f}%".format(100 * np.mean(o >= t)) for t in THRESHOLDS))
    o = np.array([r["offset_cents"] for r in rows if r["reliable"]])
    print("signed offsets: mean {0:+.1f} cents, share below -25: {1:.1f} %, above +25: {2:.1f} %".format(
        o.mean(), 100 * np.mean(o <= -25), 100 * np.mean(o >= 25)))
    hist, edges = np.histogram(o, bins=np.arange(-50, 51, 10))
    print("histogram (10 cent bins from -50): " + " ".join(str(h) for h in hist))
    return 0


if __name__ == "__main__":
    sys.exit(main())
