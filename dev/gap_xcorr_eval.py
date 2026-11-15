#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gap_xcorr_eval.py – find the time offset between notes and audio by cross-correlating voicing
==============================================================================================

Idea: RMVPE's confidence per 10 ms frame says whether someone sings. The notes of the file give
the same as a 0/1 signal (inside a note or not). The peak of the cross-correlation of the two is the
offset by which #GAP is wrong; a weak or ambiguous peak, or different offsets in different parts of
the song, mean that audio and notes don't fit (wrong audio version, wrong #BPM).

Unlike rmvpe.timing_search this needs no pitches, so the tool could use it.

    # 1. per song: RMVPE on the mix (and on the vocals, if free), f0, voicing, chroma frames
    python dev/gap_xcorr_eval.py extract dev/lists/rmvpe_batch_songs.txt dev/lists/rmvpe_batch_songs_2021plus.txt \\
        --root . --gpu
    # 2. variants, agreement with the pitch based offset, mismatch and drift tests
    python dev/gap_xcorr_eval.py evaluate
    # 3. accuracy of the tool (ensemble + postprocessing) at file / xcorr / pitch based timing
    python dev/gap_xcorr_eval.py accuracy --signal bin0.5

Reference ("oracle") offset: rmvpe.timing_search idea with the pitch classes of the file
(agreement after the best transposition, -2..+2 s, 10 ms steps). It needs correct labels, so it is
only used where its agreement is >= 0.4.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import rmvpe_eval_batch as eb                                   # noqa: E402
from ultrastar_pitch import rmvpe as rm                         # noqa: E402
from ultrastar_pitch import ProjectParser                       # noqa: E402

FRAME_S = rm.FRAME_S
CACHE = HERE / "results" / "gap_cache"
SIGNALS = ["cont", "bin0.1", "bin0.3", "bin0.5", "bin0.7"]
ORACLE_MS = np.arange(-2000, 2001, 10)
ORACLE_MIN_AGREE = 0.4
RESULT_PKL = HERE / "results" / "gap_xcorr_accuracy.pkl"
EPS = 1e-9


# --------------------------------------------------------------------------- #
# signals and cross-correlation
# --------------------------------------------------------------------------- #
def note_mask(n_frames: int, segments) -> np.ndarray:
    """ 1 inside the notes, 0 outside, per 10 ms frame """
    times = np.arange(n_frames) * FRAME_S
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    lo, hi = np.searchsorted(times, seg[:, 0]), np.searchsorted(times, seg[:, 1])
    diff = np.zeros(n_frames + 1)
    np.add.at(diff, lo, 1); np.add.at(diff, hi, -1)
    return (np.cumsum(diff)[:-1] > 0).astype(np.float64)


def voicing(conf: np.ndarray, signal: str) -> np.ndarray:
    conf = np.asarray(conf, dtype=np.float64)
    if signal == "cont":
        return conf
    return (conf > float(signal[3:])).astype(np.float64)


def xcorr(v: np.ndarray, n: np.ndarray, max_lag: int, center: bool = True) -> tuple:
    """ normalised c[k] = sum_t v0[t] n0[t - k] for k = -max_lag .. max_lag (frames); a peak at k > 0
        means the audio is later than the notes (same sign as rmvpe.timing_search). center=False:
        no mean removal (for a windowed v, where the global mean would bias the result) """
    v0, n0 = (v - v.mean(), n - n.mean()) if center else (v, n)
    size = 1 << int(np.ceil(np.log2(len(v0) + len(n0) + 1)))
    c = np.fft.irfft(np.fft.rfft(v0, size) * np.conj(np.fft.rfft(n0, size)), size)
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, c[lags % size] / max(np.linalg.norm(v0) * np.linalg.norm(n0), EPS)


def peak(lags, c, exclude_s: float = 0.5) -> dict:
    """ lag (s, parabolic sub-frame), height, and the ratio to the best peak outside +-exclude_s """
    i = int(np.argmax(c))
    frac = 0.0
    if 0 < i < len(c) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        frac = 0.5 * (c[i - 1] - c[i + 1]) / den if den < 0 else 0.0
    far = np.abs(lags - lags[i]) > exclude_s / FRAME_S
    second = float(c[far].max()) if far.any() else 0.0
    return {"lag": (lags[i] + frac) * FRAME_S, "height": float(c[i]), "second": second,
            "ratio": float(c[i] / second) if second > 0 else np.inf}


def offset_by_voicing(conf, segments, signal: str = "bin0.5", range_s: float = 10.0) -> dict:
    v = voicing(conf, signal)
    n = note_mask(len(v), segments)
    lags, c = xcorr(v, n, int(round(range_s / FRAME_S)))
    return peak(lags, c)


def scale_search(conf, segments, signal: str, eps_grid, range_s: float = 10.0) -> tuple:
    """ best tempo factor (1 + eps) and offset: the notes are stretched around the first note, the
        global cross-correlation peak is computed for every eps; returns (eps, peak dict, heights) """
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    t0 = seg[0, 0]
    v = voicing(conf, signal)
    heights, peaks = [], []
    for eps in eps_grid:
        n = note_mask(len(v), t0 + (seg - t0) * (1 + eps))
        lags, c = xcorr(v, n, int(round(range_s / FRAME_S)))
        pk = peak(lags, c)
        heights.append(pk["height"]); peaks.append(pk)
    i = int(np.argmax(heights))
    return float(eps_grid[i]), peaks[i], np.array(heights)


def local_offsets(conf, segments, signal: str, center_s: float, parts: int = 3, range_s: float = 2.0) -> np.ndarray:
    """ offset of each part of the song (notes split into `parts` groups by time), searched around center_s """
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    v = voicing(conf, signal)
    t = np.arange(len(v)) * FRAME_S
    edges = np.linspace(seg[0, 0], seg[-1, 1], parts + 1)
    out = []
    for p in range(parts):
        sel = seg[(seg[:, 0] >= edges[p]) & (seg[:, 0] < edges[p + 1])]
        if len(sel) < 10:
            out.append(np.nan)
            continue
        win = (t >= sel[0, 0] + center_s - range_s - 1) & (t <= sel[-1, 1] + center_s + range_s + 1)
        n = note_mask(len(v), sel)
        lags, c = xcorr(np.where(win, v, 0.0), n, int(round((abs(center_s) + range_s) / FRAME_S)), center=False)
        keep = np.abs(lags * FRAME_S - center_s) <= range_s
        out.append(peak(lags[keep], c[keep])["lag"])
    return np.array(out)


# --------------------------------------------------------------------------- #
# pitch based reference offset
# --------------------------------------------------------------------------- #
def agreement_curve(f0, segments, ref, offset_cents, offsets_ms) -> np.ndarray:
    times = np.arange(len(f0)) * FRAME_S
    cls = rm.segment_classes_offsets(f0, times, segments, np.asarray(offsets_ms) / 1000.0, offset_cents, 3)
    return rm.shifts_for_classes(cls, np.asarray(ref))[1]


def oracle(agree: np.ndarray, plateau: float = 0.005) -> float:
    """ middle of the plateau of the agreement maximum (as rmvpe.timing_search), seconds """
    cand = np.flatnonzero(agree >= agree.max() - plateau)
    center = np.median(ORACLE_MS[cand])
    return float(ORACLE_MS[cand[np.argmin(np.abs(ORACLE_MS[cand] - center))]]) / 1000.0


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
def cache_file(song: str) -> Path:
    return CACHE / (hashlib.sha1(str(Path(song).resolve()).encode()).hexdigest()[:16] + ".pkl")


def analyse_audio(model, samples: np.ndarray, segments) -> dict:
    sal = model.salience(samples.astype(np.float32) / 32768.0)
    f0, conf = rm.decode_salience(sal, 0.15)
    tuning = rm.estimate_tuning(sal, segments, 0.15)
    cls = np.rint(rm.BIN_MIDI - tuning.offset_cents / 100.0).astype(int) % 12
    fold = np.zeros((rm.N_CLASS, 12)); fold[np.arange(rm.N_CLASS), cls] = 1.0
    return {"conf": conf.astype(np.float16), "f0": f0.astype(np.float32), "tuning": float(tuning.offset_cents),
            "chroma": (sal @ fold).astype(np.float32)}


def extract(args) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    songs = []
    for lst in args.lists:
        songs += eb.collect_songs([lst], args.root)
    model = rm.Rmvpe(gpu=args.gpu)
    parser = ProjectParser()
    t0 = time.time()
    for i, song in enumerate(songs, 1):
        target = cache_file(song)
        if target.exists():
            continue
        try:
            parser.load_note_file(song)
            segments = np.array(parser.note_segments(), dtype=np.float64)
            item = {"path": song, "segments": segments, "ref": np.array(parser.dump_pitches(), dtype=np.int8),
                    "duet": parser.is_duet}
            item["mix"] = analyse_audio(model, parser.load_main_audio(rm.SR), segments)
            item["vocals"] = None
            if parser.has_vocal_audio():
                voc = parser.load_audio(rm.SR, use_vocals_file=True)
                if parser.audio_source in ("#VOCALS", "vocals.wav"):
                    item["vocals"] = analyse_audio(model, voc, segments)
            item["agree"] = agreement_curve(item["mix"]["f0"], segments, item["ref"], item["mix"]["tuning"], ORACLE_MS)
        except Exception as err:                                            # noqa: BLE001
            print(f"[{i}/{len(songs)}] skipped {Path(song).parent.name}: {type(err).__name__}: {err}", flush=True)
            continue
        with open(target, "wb") as f:
            pickle.dump(item, f)
        print(f"[{i}/{len(songs)}] {Path(song).parent.name[:40]}  vocals={item['vocals'] is not None}  "
              f"{(time.time() - t0) / i:.1f} s/song", flush=True)
    with open(CACHE / "songs.txt", "w") as f:
        f.write("\n".join(songs) + "\n")
    return 0


def load_items():
    songs = [s for s in (CACHE / "songs.txt").read_text().splitlines() if s]
    items = []
    for song in songs:
        p = cache_file(song)
        if p.exists():
            with open(p, "rb") as f:
                items.append(pickle.load(f))
    return items


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ probability that a matching pair scores higher than a non matching one """
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def evaluate(args) -> int:
    items = load_items()
    n_voc = sum(it["vocals"] is not None for it in items)
    print(f"{len(items)} songs ({n_voc} with free vocals, {sum(it['duet'] for it in items)} duets)\n")
    for it in items:
        it["oracle"] = oracle(it["agree"])
        it["agree_max"] = float(it["agree"].max())
        it["agree0"] = float(it["agree"][ORACLE_MS == 0][0])
    rel = [it for it in items if it["agree_max"] >= ORACLE_MIN_AGREE]
    orc = np.array([it["oracle"] for it in rel])
    print(f"pitch based reference: reliable (agreement >= {ORACLE_MIN_AGREE}) for {len(rel)} songs; "
          f"|offset| <= 20 ms: {np.mean(np.abs(orc) <= 0.02) * 100:.0f} %, > 50 ms: {np.sum(np.abs(orc) > 0.05)}, "
          f"> 100 ms: {np.sum(np.abs(orc) > 0.1)}, > 300 ms: {np.sum(np.abs(orc) > 0.3)} songs")
    print("  offsets > 100 ms: " + ", ".join(f"{1000 * o:+.0f}" for o in sorted(orc[np.abs(orc) > 0.1])) + "\n")

    # 1. variants: agreement of the voicing offset with the pitch based offset
    print(f"{'source':<7}{'signal':<8}{'songs':>6} | |xcorr - pitch|  <=20ms  <=50ms <=100ms  >300ms | "
          f"median height  ratio")
    results = {}
    for source in ("mix", "vocals"):
        for signal in SIGNALS:
            rows = []
            for it in rel:
                if it[source] is None:
                    continue
                pk = offset_by_voicing(it[source]["conf"], it["segments"], signal, args.range)
                rows.append((pk["lag"] - it["oracle"], pk["height"], pk["ratio"]))
            if not rows:
                continue
            d, h, r = (np.array(x) for x in zip(*rows))
            results[(source, signal)] = d
            print(f"{source:<7}{signal:<8}{len(d):>6} | {'':15}{100 * np.mean(np.abs(d) <= 0.02):6.1f}%"
                  f"{100 * np.mean(np.abs(d) <= 0.05):7.1f}%{100 * np.mean(np.abs(d) <= 0.1):7.1f}%"
                  f"{100 * np.mean(np.abs(d) > 0.3):7.1f}% | {np.median(h):13.2f}{np.median(r):7.2f}")
    # same songs only: mix vs vocals
    both = [it for it in rel if it["vocals"] is not None]
    if both:
        print(f"\nsame {len(both)} songs (with vocals), signal {args.signal}:")
        for source in ("mix", "vocals"):
            d = np.array([offset_by_voicing(it[source]["conf"], it["segments"], args.signal, args.range)["lag"]
                          - it["oracle"] for it in both])
            print(f"  {source:<7} <=20ms {100 * np.mean(np.abs(d) <= 0.02):5.1f}%  <=50ms {100 * np.mean(np.abs(d) <= 0.05):5.1f}%"
                  f"  >100ms {100 * np.mean(np.abs(d) > 0.1):5.1f}%  median |d| {1000 * np.median(np.abs(d)):.0f} ms")

    sig = args.signal
    pk_all = [offset_by_voicing(it["mix"]["conf"], it["segments"], sig, args.range) for it in items]
    lag = np.array([p["lag"] for p in pk_all])
    height = np.array([p["height"] for p in pk_all])
    ratio = np.array([p["ratio"] for p in pk_all])

    # 2. the songs where the two methods disagree
    print(f"\nmix, {sig}: songs with |xcorr - pitch| > 50 ms (reliable reference):")
    for it, p in zip(items, pk_all):
        if it["agree_max"] >= ORACLE_MIN_AGREE and abs(p["lag"] - it["oracle"]) > 0.05:
            print(f"  {Path(it['path']).parent.name[:44]:<44} xcorr {1000 * p['lag']:+6.0f} ms  pitch {1000 * it['oracle']:+6.0f} ms"
                  f"  height {p['height']:.2f} ratio {p['ratio']:.2f}  agreement at 0 / pitch: {it['agree0']:.2f} / {it['agree_max']:.2f}")
    print(f"songs with large voicing offset |xcorr| > 100 ms (all {len(items)} songs): {int(np.sum(np.abs(lag) > 0.1))}")

    # 3. wrong audio: notes of song i with the audio of song i+k
    neg = []
    for k in (1, 2, 3):
        for i, it in enumerate(items):
            other = items[(i + k * 37) % len(items)]
            neg.append(offset_by_voicing(other["mix"]["conf"], it["segments"], sig, args.range))
    neg_h = np.array([p["height"] for p in neg])
    neg_r = np.array([p["ratio"] for p in neg])
    print(f"\nwrong audio (notes of one song, mix of another, {len(neg)} pairs):")
    print(f"  peak height  matching: median {np.median(height):.2f}, 5 % quantile {np.percentile(height, 5):.2f}, min {height.min():.2f}"
          f" | wrong: median {np.median(neg_h):.2f}, 95 % quantile {np.percentile(neg_h, 95):.2f}, max {neg_h.max():.2f}"
          f" | AUC {auc(height, neg_h):.3f}")
    print(f"  peak ratio   matching: median {np.median(ratio):.2f}, 5 % quantile {np.percentile(ratio, 5):.2f}"
          f" | wrong: median {np.median(neg_r):.2f}, 95 % quantile {np.percentile(neg_r, 95):.2f} | AUC {auc(ratio, neg_r):.3f}")
    for thr in (0.15, 0.2, 0.25, 0.3):
        print(f"  height < {thr:.2f}: {100 * np.mean(height < thr):5.1f} % of the matching songs flagged, "
              f"{100 * np.mean(neg_h < thr):5.1f} % of the wrong pairs detected")

    # 4. drift: offsets of the three thirds of the song, real files and simulated wrong #BPM
    def spread(it, segments, center):
        lo = local_offsets(it["mix"]["conf"], segments, sig, center)
        return float(np.nanmax(lo) - np.nanmin(lo)) if np.isfinite(lo).sum() >= 2 else np.nan
    real = np.array([spread(it, it["segments"], l) for it, l in zip(items, lag)])
    print(f"\ndrift (offset of the last minus the first third etc., max - min of three parts):")
    print(f"  files as they are: median {1000 * np.nanmedian(real):.0f} ms, 90 % quantile {1000 * np.nanpercentile(real, 90):.0f} ms, "
          f"> 100 ms: {int(np.nansum(real > 0.1))}, > 300 ms: {int(np.nansum(real > 0.3))} songs")
    for eps in (0.002, 0.005, 0.01, 0.02):
        sim = []
        for it in items:
            seg = it["segments"]
            t0 = seg[0, 0]
            scaled = t0 + (seg - t0) * (1 + eps)
            pk = offset_by_voicing(it["mix"]["conf"], scaled, sig, args.range)
            sim.append((spread(it, scaled, pk["lag"]), pk["height"]))
        s, h = (np.array(x) for x in zip(*sim))
        print(f"  #BPM off by {100 * eps:.1f} %: spread median {1000 * np.nanmedian(s):.0f} ms, > 100 ms in "
              f"{100 * np.nanmean(s > 0.1):.0f} %, > 300 ms in {100 * np.nanmean(s > 0.3):.0f} % of the songs; "
              f"peak height median {np.median(h):.2f}")
    print("  songs with spread > 300 ms as they are:")
    for it, s, l in zip(items, real, lag):
        if s > 0.3:
            lo = local_offsets(it["mix"]["conf"], it["segments"], sig, l)
            print(f"    {Path(it['path']).parent.name[:44]:<44} parts " + " ".join(f"{1000 * x:+6.0f}" for x in lo)
                  + f" ms, pitch agreement {it['agree_max']:.2f}")
    return 0


# --------------------------------------------------------------------------- #
# accuracy of the tool at three timings
# --------------------------------------------------------------------------- #
def accuracy(args) -> int:
    from ultrastar_pitch import AudioPreprocessor, DetectionPipeline, PitchClassifier, StochasticPostprocessor
    from ultrastar_pitch import repetition_pooling as rp
    from ultrastar_pitch import tuning_correction as tc
    logging.basicConfig(level=logging.ERROR)
    items = load_items()
    pipe = DetectionPipeline(ProjectParser(), AudioPreprocessor(stride=128), PitchClassifier(), StochasticPostprocessor())
    parser = pipe.project_parser
    names = ["file", "xcorr", f"xcorr if |o|>{int(1000 * args.min_shift)}ms, h>={args.min_height}", "pitch (oracle)"]
    res = {n: [] for n in names}
    per_song = []
    for i, it in enumerate(items, 1):
        parser.load_note_file(it["path"])
        mix = parser.load_main_audio(rm.SR)
        ctx = parser.note_context()
        seg = it["segments"]
        pk = offset_by_voicing(it["mix"]["conf"], seg, args.signal, args.range)
        xo = round(pk["lag"], 2)
        guarded = abs(xo) > args.min_shift and pk["height"] >= args.min_height
        offs = {"file": 0.0, names[1]: xo, names[2]: xo if guarded else 0.0,
                names[3]: oracle(it["agree"]) if it["agree"].max() >= ORACLE_MIN_AGREE else 0.0}
        cs = np.vstack([np.zeros((1, 12)), np.cumsum(it["mix"]["chroma"].astype(np.float64), axis=0)])
        times = np.arange(len(it["mix"]["chroma"])) * FRAME_S
        cache = {}
        line = []
        for name, off in offs.items():
            if off not in cache:
                pn = pipe.pitchnet_scores(mix, parser.note_bounds(rm.SR, off))
                lo = np.searchsorted(times, seg[:, 0] + off)
                hi = np.maximum(np.searchsorted(times, seg[:, 1] + off), lo)
                ch = cs[hi] - cs[lo]
                ch = ch / np.maximum(ch.sum(axis=1, keepdims=True), 1e-6)
                scores = 0.6 * np.log(ch + 1e-6) + 0.4 * pn
                scores = rp.pool_repetitions(scores, **ctx)
                pred = StochasticPostprocessor.correct_scores(scores)
                sh = tc.estimate_shift_semitones(pred, it["ref"].astype(int))
                d = ((pred + sh.shift_semitones) % 12 - it["ref"]) % 12
                cache[off] = np.minimum(d, 12 - d)
            res[name].append(cache[off])
            line.append(f"{1000 * off:+.0f}ms {100 * np.mean(cache[off] == 0):.1f}%")
        # for the threshold choice (thresholds): exact distances at file timing and at the voicing offset
        per_song.append({"path": it["path"], "lag": xo, "height": pk["height"], "d_file": cache[0.0], "d_xcorr": cache[xo],
                         "dur": seg[:, 1] - seg[:, 0]})
        print(f"[{i}/{len(items)}] {Path(it['path']).parent.name[:36]}: " + " | ".join(line), flush=True)
    with open(RESULT_PKL, "wb") as f:
        pickle.dump({"signal": args.signal, "songs": per_song}, f)
    print(f"\n{len(items)} songs, mix, ensemble + repetition pooling + key prior, signal {args.signal}")
    base = np.array([np.mean(d == 0) for d in res["file"]])
    for name in names:
        d = np.concatenate(res[name])
        acc = np.array([np.mean(x == 0) for x in res[name]])
        diff = acc - base
        rng = np.random.default_rng(0)
        m = diff[rng.integers(0, len(diff), (4000, len(diff)))].mean(axis=1)
        print(f"  {name:<32} exact {100 * np.mean(d == 0):6.2f}%  ±1 {100 * np.mean(d <= 1):6.2f}% | vs file "
              f"{100 * diff.mean():+5.2f} [{100 * np.percentile(m, 2.5):+5.2f} .. {100 * np.percentile(m, 97.5):+5.2f}]"
              f"  songs better/worse by > 1 pt: {int(np.sum(diff > 0.01))}/{int(np.sum(diff < -0.01))}")
    return 0


def thresholds(args) -> int:
    """ choose the minimum shift and the minimum peak height on one song list, test on the other """
    with open(RESULT_PKL, "rb") as f:
        res = pickle.load(f)["songs"]
    runs = {}
    for k, lst in enumerate(args.lists):
        for song in eb.collect_songs([lst], args.root):
            runs[str(Path(song).resolve())] = k
    run = np.array([runs.get(str(Path(r["path"]).resolve()), -1) for r in res])
    acc_f = np.array([np.mean(r["d_file"] == 0) for r in res])
    acc_x = np.array([np.mean(r["d_xcorr"] == 0) for r in res])
    lag = np.abs(np.array([r["lag"] for r in res]))
    height = np.array([r["height"] for r in res])
    shifts = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
    heights = [0.1, 0.15, 0.2, 0.25, 0.3]

    def gain(sel, ms, mh):
        use = (lag > ms) & (height >= mh)
        return np.where(use, acc_x, acc_f)[sel] - acc_f[sel]

    print(f"{len(res)} songs, runs: " + ", ".join(f"{k}: {int(np.sum(run == k))}" for k in range(len(args.lists))))
    print("mean gain per song (points) for min shift (rows) x min height (columns), all songs:")
    print(f"{'':>8}" + "".join(f"{h:>8}" for h in heights))
    for ms in shifts:
        print(f"{1000 * ms:6.0f}ms" + "".join(f"{100 * gain(run >= 0, ms, mh).mean():8.2f}" for mh in heights))
    cross = np.zeros(len(res))
    for k in range(len(args.lists)):
        train = (run >= 0) & (run != k)
        best = max(((ms, mh) for ms in shifts for mh in heights), key=lambda p: gain(train, *p).mean())
        test = run == k
        cross[test] = gain(test, *best)
        print(f"run {k}: chosen on the other list: min shift {1000 * best[0]:.0f} ms, min height {best[1]} -> "
              f"gain on this list {100 * cross[test].mean():+.2f} points")
    sel = run >= 0
    m = cross[sel][np.random.default_rng(0).integers(0, sel.sum(), (4000, sel.sum()))].mean(axis=1)
    print(f"cross-run gain over all songs: {100 * cross[sel].mean():+.2f} [{100 * np.percentile(m, 2.5):+.2f} .. "
          f"{100 * np.percentile(m, 97.5):+.2f}], songs worse by > 1 pt: {int(np.sum(cross[sel] < -0.01))}")
    for thr in (0.08, 0.1, 0.12, 0.15):
        print(f"  peak height < {thr}: {int(np.sum(height < thr))} of {len(res)} matching songs")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("extract"); a.add_argument("lists", nargs="+"); a.add_argument("--root")
    a.add_argument("--gpu", action="store_true")
    for name in ("evaluate", "accuracy"):
        b = sub.add_parser(name)
        b.add_argument("--signal", default="bin0.5", choices=SIGNALS)
        b.add_argument("--range", type=float, default=10.0, help="search range of the offset, seconds")
        if name == "accuracy":
            b.add_argument("--min-shift", type=float, default=0.03)
            b.add_argument("--min-height", type=float, default=0.2)
    t = sub.add_parser("thresholds"); t.add_argument("lists", nargs=2); t.add_argument("--root")
    args = ap.parse_args(argv)
    return {"extract": extract, "evaluate": evaluate, "accuracy": accuracy, "thresholds": thresholds}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
