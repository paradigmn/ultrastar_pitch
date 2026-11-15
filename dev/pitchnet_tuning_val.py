#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          pitchnet_tuning_val.py
@brief         tuning correction for the frame model (pitchnet_2026) on the validation songs
@author        paradigm

The frame model was trained on songs in tune and loses accuracy on detuned audio (dev/pitchnet_val_compare.py:
-5 points at +29 cents, -9 at -29 cents). Instead of training on detuned audio (which costs accuracy on the songs
in tune), the pipeline can correct the tuning before the model: RMVPE estimates the tuning offset of the voice
(rmvpe.estimate_tuning, already used for the RMVPE chroma) and pitch_classifier.shift_power moves the power
spectrum by it (PitchClassifier.predict_frames(samples, tuning_cents)).

Per validation song (audio and labels of the cache of dev/pitchnet_data.py, RMVPE on the CPU so the GPU stays
free): the RMVPE tuning estimate of the mix (cached in --estimates), then the exact note accuracy of the frame
model (probabilities summed per note, as the tool does)
  * on the mix with the spectrum shifted by fixed offsets (sensitivity of the model to the tuning, -30 .. +30 cents)
  * on the mix, the mix resampled by +-29 cents (detuned songs) and the vocals, without and with the correction
    by the RMVPE estimate (for the resampled mix: the estimate of the original + -29 cents; --check-detuned N
    measures RMVPE on the resampled audio of N songs to verify that)
and compares the policies "never", "always" and "only if |offset| >= T cents" paired per song.

    CUDA_VISIBLE_DEVICES= .venv/bin/python dev/pitchnet_tuning_val.py --workers 2 > dev/results/pitchnet_tuning_val.txt
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CACHE = ROOT / "dev" / "results" / "pitchnet_cache"
SELECTION = ROOT / "dev" / "lists" / "pitchnet" / "selection.csv"
SWEEP = (-30, -20, -10, 0, 10, 20, 30)
DETUNE = (("+29", 1000, 1017, 29.0), ("-29", 1017, 1000, -29.0))     # resample_poly up, down; cents
THRESHOLDS = (0, 5, 10, 15, 20)
_W = {}


def init_worker(model, threads):
    import logging
    logging.disable(logging.WARNING)
    from ultrastar_pitch import rmvpe as rm
    from ultrastar_pitch.pitch_classifier import PitchClassifier
    import onnxruntime as ort
    _W["rmvpe"] = rm.Rmvpe(rm.default_model_path(), threads=threads, gpu=False)
    clf = PitchClassifier(model)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads          # several workers: no session may take all cores
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    clf.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    _W["clf"] = clf
    _W["rm"] = rm


def note_hits(power, segments, classes, cents):
    """ exact hits per note of the frame model on the (shifted) power spectrum """
    from ultrastar_pitch.pitch_classifier import FRAME_FFT_BINS, shift_power
    clf = _W["clf"]
    p = shift_power(power, cents) if cents else power[:, :FRAME_FFT_BINS]
    logprob, _ = clf.predict_power(np.ascontiguousarray(p))
    pred = clf.frame_scores_per_note(logprob, segments).argmax(axis=1)
    return pred == classes


def song(job):
    from scipy.signal import resample_poly
    from ultrastar_pitch.pitch_classifier import FRAME_FFT_BINS_TUNING, power_spectrogram
    sid, estimate, check, sweep = job
    z = np.load(CACHE / (sid + ".labels.npz"))
    segments, classes = z["segments"], z["classes"]
    mix = np.load(CACHE / (sid + ".mix.npy")).astype(np.float32) / 32768.0
    out = {"id": sid, "notes": len(classes)}
    t0 = time.time()
    if estimate is None:
        tun = _W["rm"].estimate_tuning(_W["rmvpe"].salience(mix), segments, 0.15)
        estimate = (float(tun.offset_cents), bool(tun.reliable))
    out["offset"], out["reliable"] = estimate
    off = estimate[0]
    power = power_spectrogram(mix, n_bins=FRAME_FFT_BINS_TUNING)
    out["mix_none"] = note_hits(power, segments, classes, 0.0).sum()
    for s in SWEEP if sweep else ():
        # the model sees the song s cents off
        out["sweep{0:+d}".format(s)] = out["mix_none"] if s == 0 else note_hits(power, segments, classes, -s).sum()
    out["mix_corr"] = note_hits(power, segments, classes, off).sum()
    for name, up, down, cents in DETUNE:
        x = resample_poly(mix, up, down).astype(np.float32)
        seg = segments * (up / down)
        pw = power_spectrogram(x, n_bins=FRAME_FFT_BINS_TUNING)
        out["det{0}_none".format(name)] = note_hits(pw, seg, classes, 0.0).sum()
        out["det{0}_corr".format(name)] = note_hits(pw, seg, classes, off + cents).sum()
        if check:
            tun = _W["rm"].estimate_tuning(_W["rmvpe"].salience(x), seg, 0.15)
            out["det{0}_est".format(name)] = float(tun.offset_cents)
    vocals = np.load(CACHE / (sid + ".vocals.npy")).astype(np.float32) / 32768.0
    pv = power_spectrogram(vocals, n_bins=FRAME_FFT_BINS_TUNING)
    out["voc_none"] = note_hits(pv, segments, classes, 0.0).sum()
    out["voc_corr"] = note_hits(pv, segments, classes, off).sum()
    out["seconds"] = time.time() - t0
    return out


def boot(a, b, n, samples=4000, seed=0):
    """ pooled exact (b - a) in points with a 95 % interval over songs """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), (samples, len(n)))
    d = (b[idx].sum(1) - a[idx].sum(1)) / n[idx].sum(1)
    return 100 * (b.sum() - a.sum()) / n.sum(), 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "pitchnet_2026_11_11.onnx"),
                    help="frame model onnx (default: the first version pitchnet_2026_11_11, as measured in dev/README.md)")
    ap.add_argument("--estimates", default=str(ROOT / "dev" / "results" / "pitchnet_tuning_estimates_val.csv"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8, help="threads of RMVPE and of the frame model per worker")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-detuned", type=int, default=30, help="songs with RMVPE on the resampled audio")
    ap.add_argument("--sweep-songs", type=int, default=80, help="songs with the sensitivity sweep")
    args = ap.parse_args()
    with open(SELECTION, newline="", encoding="utf-8") as file:
        ids = [r["id"] for r in csv.DictReader(file) if r["split"] == "val"]
    ids = ids[:args.limit] if args.limit else ids
    known = {}
    if os.path.exists(args.estimates):
        with open(args.estimates, newline="", encoding="utf-8") as file:
            known = {r["id"]: (float(r["offset_cents"]), r["reliable"] == "1") for r in csv.DictReader(file)}
    jobs = [(sid, known.get(sid), i < args.check_detuned, i < args.sweep_songs) for i, sid in enumerate(ids)]
    print("model {0}, {1} validation songs, {2} tuning estimates cached".format(Path(args.model).name, len(ids),
                                                                           sum(j[1] is not None for j in jobs)))
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(args.model, args.threads)) as pool:
        for i, r in enumerate(pool.map(song, jobs), 1):
            rows.append(r)
            print("[{0}/{1}] {2}: offset {3:+.1f} cents, mix {4}/{5} -> {6}/{5} ({7:.0f} s, total {8:.0f} s)".format(
                i, len(jobs), r["id"], r["offset"], r["mix_none"], r["notes"], r["mix_corr"], r["seconds"],
                time.time() - t0), file=sys.stderr, flush=True)
    with open(args.estimates, "w", newline="", encoding="utf-8") as file:
        w = csv.writer(file)
        w.writerow(["id", "offset_cents", "reliable"])
        for r in rows:
            w.writerow([r["id"], "{0:.2f}".format(r["offset"]), int(r["reliable"])])

    n = np.array([r["notes"] for r in rows], dtype=float)
    col = lambda key: np.array([r[key] for r in rows], dtype=float)   # noqa: E731
    off = col("offset")
    print("{0} songs, {1:.0f} notes, {2:.0f} s".format(len(rows), n.sum(), time.time() - t0))
    a = np.abs(off)
    print("\nRMVPE tuning estimate of the mix: median |offset| {0:.1f} cents, mean offset {1:+.1f}; "
          "|offset| >= 10: {2:.1%}, >= 15: {3:.1%}, >= 25: {4:.1%}; reliable {5:.1%}".format(
              np.median(a), off.mean(), np.mean(a >= 10), np.mean(a >= 15), np.mean(a >= 25), col("reliable").mean()))
    checked = [r for r in rows if "det+29_est" in r]
    if checked:
        for name, _, _, cents in DETUNE:
            d = np.array([r["det{0}_est".format(name)] - r["offset"] - cents for r in checked])
            print("  resampled {0} cents ({1} songs): estimate - (original + {2:+.0f}) = {3:+.1f} cents mean, "
                  "{4:.1f} median abs".format(name, len(checked), cents, d.mean(), np.median(np.abs(d))))

    sw = [r for r in rows if "sweep+0" in r]
    ns = np.array([r["notes"] for r in sw], dtype=float)
    print("\nsensitivity ({0} songs): mix with the spectrum shifted (the model sees the song s cents off), exact %".format(len(sw)))
    base = np.array([r["sweep+0"] for r in sw], dtype=float)
    for s in SWEEP:
        h = np.array([r["sweep{0:+d}".format(s)] for r in sw], dtype=float)
        print("  s = {0:+3d} cents: {1:6.2f} %   vs 0: {2:+.2f} [{3:+.2f} .. {4:+.2f}]".format(
            s, 100 * h.sum() / ns.sum(), *boot(base, h, ns)))

    print("\ncorrection by the RMVPE estimate (policy: correct if |offset| >= T cents), exact %")
    print("  {0:<12} {1:>8}".format("condition", "none") + "".join("{0:>26}".format("T = {0}".format(t)) for t in THRESHOLDS))
    for cond, shift in (("mix", 0.0), ("det+29", 29.0), ("det-29", -29.0), ("voc", 0.0)):
        none, corr = col(cond + "_none"), col(cond + "_corr")
        mag = np.abs(off + shift)          # the offset the pipeline would see on this audio
        line = "  {0:<12} {1:7.2f}%".format(cond, 100 * none.sum() / n.sum())
        for t in THRESHOLDS:
            pol = np.where(mag >= t, corr, none)
            line += "   {0:6.2f}% {1:+.2f} [{2:+.2f}..{3:+.2f}]".format(100 * pol.sum() / n.sum(), *boot(none, pol, n))
        print(line)
    print("\nmix by |offset| (T = 0 against none)")
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 25), (25, 1000)):
        m = (a >= lo) & (a < hi)
        if m.any():
            print("  {0:>3} .. {1:<4} cents: {2:3d} songs, none {3:6.2f} %, corrected {4:6.2f} %".format(
                lo, hi if hi < 1000 else "", int(m.sum()), 100 * col("mix_none")[m].sum() / n[m].sum(),
                100 * col("mix_corr")[m].sum() / n[m].sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
