#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          pipeline_compare.py
@brief         paired per-song comparison of configurations from two (or one) pipeline_eval.py logs
@author        paradigm

pipeline_eval.py compares the configurations of one run with each other. This script pairs the per-song
accuracies of a configuration in one log with a configuration in another log (e.g. the old and the new pitch
model), by song name, and prints the mean difference with a bootstrap interval.

    python dev/pipeline_compare.py dev/results/pipeline_eval_rep.log:ensemble+pp dev/results/pipeline_eval_new.log:ensemble+pp@file
    python dev/pipeline_compare.py LOG_A:CONFIG_A LOG_B:CONFIG_B [--n 4000]
"""

import argparse
import re
import sys

import numpy as np

LINE_RE = re.compile(r"^\[\d+/\d+\] (?P<song>.+?): (?P<rest>.*)$")


def per_song(log: str, config: str) -> dict:
    """ song name -> exact accuracy (0 .. 1) of the configuration in the log """
    out = {}
    pattern = re.compile(r"(?:^|\| )" + re.escape(config) + r" (?P<acc>[\d.]+)%")
    with open(log, encoding="utf-8") as file:
        for line in file:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            a = pattern.search(m.group("rest"))
            if a:
                out[m.group("song")] = float(a.group("acc")) / 100.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="LOG:CONFIG, the baseline")
    ap.add_argument("b", help="LOG:CONFIG, the candidate")
    ap.add_argument("--n", type=int, default=4000, help="bootstrap samples")
    ap.add_argument("--names", action="store_true", help="also print the songs with the largest losses and gains "
                    "(off by default, so that logs and notebooks do not list the songs of the private collection)")
    args = ap.parse_args()
    (log_a, cfg_a), (log_b, cfg_b) = args.a.rsplit(":", 1), args.b.rsplit(":", 1)
    a, b = per_song(log_a, cfg_a), per_song(log_b, cfg_b)
    songs = sorted(set(a) & set(b))
    if len(songs) < 3:
        print("only {0} songs in both ({1} and {2} songs found)".format(len(songs), len(a), len(b)))
        return 1
    x, y = np.array([a[s] for s in songs]), np.array([b[s] for s in songs])
    diff = y - x
    rng = np.random.default_rng(0)
    m = diff[rng.integers(0, len(diff), (args.n, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    print("{0} songs | {1}: {2:.2f} % | {3}: {4:.2f} % | song mean".format(len(songs), cfg_a, 100 * x.mean(), cfg_b, 100 * y.mean()))
    print("{0} - {1}: {2:+.2f} points [{3:+.2f} .. {4:+.2f}], better in {5} songs, worse in {6}".format(
        cfg_b, cfg_a, 100 * diff.mean(), 100 * lo, 100 * hi, int((diff > 0.0005).sum()), int((diff < -0.0005).sum())))
    if args.names:
        worst = sorted(zip(diff, songs))[:3]
        best = sorted(zip(diff, songs))[-3:]
        print("largest losses: " + ", ".join("{0} {1:+.1f}".format(s[:40], 100 * d) for d, s in worst))
        print("largest gains:  " + ", ".join("{0} {1:+.1f}".format(s[:40], 100 * d) for d, s in best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
