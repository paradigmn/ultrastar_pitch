#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          pipeline_weights_val.py
@brief         the ensemble weights of DetectionPipeline for a pitch model, tuned on the validation songs of pitchnet_2026
@author        paradigm

DetectionPipeline combines per note: scores = w * RMVPE + (1 - w) * pitchnet (log domain, RMVPE_WEIGHT), the
vocals with (1 - v) * mix + v * vocals (VOCALS_WEIGHT), then repetition pooling and the soft key prior with
weight k (StochasticPostprocessor.PRIOR_WEIGHT). These weights were set with the old pitchnet; this script
measures them for a new model on the validation songs of dev/lists/pitchnet/selection.csv (never on the 197
evaluation songs, which stay the test set).

extract: runs the steps of DetectionPipeline.transform once per song (note file, audio, RMVPE salience, timing
check with correction, then the per note scores of RMVPE, of the old pitchnet ("old") and of every frame model
given with --model NAME=PATH, on the mix and on the vocals) and caches them with the note context and the
reference classes:

    .venv/bin/python dev/pipeline_weights_val.py extract --model base=dev/results/pitchnet_runs/base/model.onnx \
        --model near_aug=dev/results/pitchnet_runs/near_aug/model.onnx

evaluate: combines the cached scores for a grid of weights exactly like the pipeline (per model) and scores the
result like pipeline_eval.py (exact pitch class after the global transposition per song, pooled over the notes;
intervals: songs resampled):

    .venv/bin/python dev/pipeline_weights_val.py evaluate > dev/results/pipeline_weights_val.txt
"""

import argparse
import csv
import itertools
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import rmvpe_eval as ev                                  # noqa: E402
from ultrastar_pitch import tuning_correction as tc      # noqa: E402
from ultrastar_pitch import (AudioPreprocessor, DetectionPipeline, PitchClassifier, ProjectParser,  # noqa: E402
                             StochasticPostprocessor)
from ultrastar_pitch import repetition_pooling, timing_check  # noqa: E402
from ultrastar_pitch import rmvpe as rmvpe_mod           # noqa: E402

SELECTION = ROOT / "dev" / "lists" / "pitchnet" / "selection.csv"
CACHE = ROOT / "dev" / "results" / "pipeline_weights_cache"
RMVPE_WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
VOCALS_WEIGHTS = (0.25, 0.35, 0.5)
PRIOR_WEIGHTS = (0.0, 0.15, 0.25, 0.4)


def val_songs(limit=0, split="val"):
    with open(SELECTION, newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["split"] == split]
    return rows[:limit] if limit else rows


def extract_rows(args) -> list:
    """ the songs of --songs: val / train (the training songs of pitchnet_2026 in a fixed random order) / list
        (the path lists of --list, ids t000, t001, ...); then --shard i/n and --limit """
    if args.songs == "list":
        import rmvpe_eval_batch as eb
        paths = eb.collect_songs(args.list, args.root)
        rows = [{"id": "t{0:03d}".format(i), "path": path} for i, path in enumerate(paths)]
    else:
        rows = val_songs(0, args.songs)
        if args.songs == "train":
            rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        rows = rows[i::n]
    return rows[:args.limit] if args.limit else rows


def save_npz(out: Path, data: dict) -> None:
    """ write the cache file under a temporary name first, so an interrupted run leaves no broken file """
    part = out.with_name(out.name + ".part")
    with open(part, "wb") as file:
        np.savez_compressed(file, **data)
    os.replace(part, out)


def update_song(pipe, rmvpe, models, args, out: Path, song: str):
    """ --update: add the scores of the models (and with --tuned the tuning estimate and the corrected scores)
        that the cached song lacks, with the cached timing correction; RMVPE runs only for a missing tuning
        estimate, or for the vocals when --sources asks for them and the cache has the mix only (then RMVPE,
        the tuning estimate and all models are added for the vocals). Returns a log message, None if nothing
        was missing """
    data = dict(np.load(out))
    sources = [s for s in ("mix", "vocals") if "rmvpe_" + s in data]
    new = [s for s in ("vocals",) if s in args.sources.split(",") and "rmvpe_" + s not in data]
    todo = {s: [n for n, clf in models.items() if n + "_" + s not in data or
                (args.tuned and clf.frame_based and n + "_tuned_" + s not in data)] for s in sources}
    need = {s: args.tuned and "tuning_" + s not in data for s in sources}
    if not any(todo.values()) and not any(need.values()) and not new:
        return None
    parser = pipe.project_parser
    parser.load_note_file(song)
    applied = float(data["applied_s"])
    segments = parser.note_segments()
    if len(segments) != len(data["ref"]):
        raise ValueError("{0} notes in the parser, {1} cached".format(len(segments), len(data["ref"])))
    if applied:
        segments = [(s + applied, e + applied) for s, e in segments]
    added = []
    for source in sources + new:
        if source not in new and not todo[source] and not need[source]:
            continue
        if source == "mix":
            samples = parser.load_main_audio(rmvpe_mod.SR)
        else:
            if source in new and not parser.has_vocal_audio():
                continue
            samples = parser.load_audio(rmvpe_mod.SR, separator=None, use_vocals_file=True)
            if source in new and parser.audio_source not in ("#VOCALS", "vocals.wav", "separated"):
                continue
        names = todo.get(source, [])
        if source in new:      # the whole source, as extract computes it
            salience = pipe.salience(rmvpe, samples)
            tuning = pipe.estimate_tuning(salience, segments)
            data["tuning_" + source] = np.array([tuning.offset_cents, float(tuning.reliable)])
            data["rmvpe_" + source] = pipe.rmvpe_scores(rmvpe, samples, segments, salience, tuning)
            added += ["rmvpe_" + source, "tuning_" + source]
            names = list(models)
        elif need[source]:
            tuning = pipe.estimate_tuning(pipe.salience(rmvpe, samples), segments)
            data["tuning_" + source] = np.array([tuning.offset_cents, float(tuning.reliable)])
            added.append("tuning_" + source)
        for name in names:
            clf = models[name]
            pipe.pitch_classifier = clf
            if name + "_" + source not in data:
                if clf.frame_based:
                    data[name + "_" + source] = pipe.frame_model_scores(samples, segments)
                else:
                    bounds = parser.note_bounds(rmvpe_mod.SR, applied) if applied else parser.note_bounds(rmvpe_mod.SR)
                    data[name + "_" + source] = pipe.pitchnet_scores(samples, bounds)
                added.append(name + "_" + source)
            if args.tuned and clf.frame_based and name + "_tuned_" + source not in data:
                data[name + "_tuned_" + source] = pipe.frame_model_scores(samples, segments, float(data["tuning_" + source][0]))
                added.append(name + "_tuned_" + source)
    if not added:
        return None
    save_npz(out, data)
    return "added " + ", ".join(added)


def extract(args) -> int:
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    models = {"old": PitchClassifier(str(HERE.parent / "ultrastar_pitch" / "binaries" / "pitchnet_2020_12_14.onnx"))}
    for item in args.model:
        name, _, path = item.partition("=")
        if not path or name in models or name == "rmvpe":
            raise SystemExit("--model NAME=PATH, unique names other than old / rmvpe: " + item)
        models[name] = PitchClassifier(path)
        assert models[name].frame_based, path + " is not a frame model"
    if args.rmvpe_threads:
        import onnxruntime as ort
        for clf in models.values():                # the same thread limit for the pitch models
            so = ort.SessionOptions()
            so.intra_op_num_threads = args.rmvpe_threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CPUExecutionProvider"] if args.cpu else clf.sess.get_providers()
            clf.sess = ort.InferenceSession(clf.sess._model_path, so, providers=providers)
    pipe = DetectionPipeline(ProjectParser(), AudioPreprocessor(stride=128), models["old"], StochasticPostprocessor())
    if args.no_old:
        del models["old"]
    if args.rmvpe_threads or args.cpu:
        # an own RMVPE session: CPU only (the GPU may be busy with a training) and a limited number of threads
        pipe._rmvpe_path = pipe.rmvpe_model or rmvpe_mod.default_model_path()
        pipe._rmvpe = rmvpe_mod.Rmvpe(pipe._rmvpe_path, threads=args.rmvpe_threads, gpu=not args.cpu)
    rmvpe = pipe._get_rmvpe()
    assert rmvpe is not None, "RMVPE model not found"
    parser = pipe.project_parser
    rows = extract_rows(args)
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        out = cache / (row["id"] + ".npz")
        if out.exists() and args.update:
            try:
                msg = update_song(pipe, rmvpe, models, args, out, row["path"])
            except Exception as e:                            # noqa: BLE001
                msg = "error {0}: {1}".format(type(e).__name__, str(e)[:80])
            if msg:
                print("[{0}/{1}] {2} {3}: {4} ({5:.0f} s)".format(i, len(rows), row["id"], Path(row["path"]).parent.name[:40],
                                                               msg, time.time() - t0), flush=True)
            continue
        if args.update or (out.exists() and not args.overwrite):
            continue                                          # --update touches only the cached songs
        song = row["path"]
        try:
            notes = ev.parse_notes(song)
            parser.load_note_file(song)
            segments = parser.note_segments()
            bounds = parser.note_bounds(rmvpe_mod.SR)
            if len(segments) != len(notes.ref_classes):
                raise ValueError("{0} notes in the parser, {1} in rmvpe_eval".format(len(segments), len(notes.ref_classes)))
            mix = parser.load_main_audio(rmvpe_mod.SR)
            salience = pipe.salience(rmvpe, mix)
            check = timing_check.check_timing(salience.max(axis=1), segments, correct=True)
            if check.applied_s:
                segments = [(s + check.applied_s, e + check.applied_s) for s, e in segments]
                bounds = parser.note_bounds(rmvpe_mod.SR, check.applied_s)
            data = {"ref": notes.ref_classes.astype(np.int16), "durations": notes.ends - notes.starts,
                    "starts": notes.starts, "applied_s": check.applied_s, "path": song}
            sources = {"mix": (mix, salience)}
            if "vocals" in args.sources.split(",") and parser.has_vocal_audio():
                vocals = parser.load_audio(rmvpe_mod.SR, separator=None, use_vocals_file=True)
                if parser.audio_source in ("#VOCALS", "vocals.wav", "separated"):
                    sources["vocals"] = (vocals, None)
            for source, (samples, sal) in sources.items():
                if sal is None:
                    sal = pipe.salience(rmvpe, samples)
                tuning = pipe.estimate_tuning(sal, segments)
                data["tuning_" + source] = np.array([tuning.offset_cents, float(tuning.reliable)])
                data["rmvpe_" + source] = pipe.rmvpe_scores(rmvpe, samples, segments, sal, tuning)
                for name, clf in models.items():
                    pipe.pitch_classifier = clf
                    if clf.frame_based:
                        data[name + "_" + source] = pipe.frame_model_scores(samples, segments)
                        if args.tuned:   # the frame model with the spectrum corrected by the tuning estimate
                            data[name + "_tuned_" + source] = pipe.frame_model_scores(samples, segments, tuning.offset_cents)
                    else:
                        data[name + "_" + source] = pipe.pitchnet_scores(samples, bounds)
            for key, value in parser.note_context().items():
                data["ctx_" + key] = np.array(value)
            save_npz(out, data)
            msg = "{0} notes, sources {1}, timing {2:+.0f} ms".format(len(segments), "+".join(sources), 1000 * check.applied_s)
        except Exception as e:                                # noqa: BLE001
            msg = "error {0}: {1}".format(type(e).__name__, str(e)[:80])
        print("[{0}/{1}] {2} {3}: {4} ({5:.0f} s)".format(i, len(rows), row["id"], Path(song).parent.name[:40], msg,
                                                       time.time() - t0), flush=True)
    return 0


def pooled_scores(z, model, w, v, use_vocals):
    """ the combined and pooled scores of the pipeline for these weights (before the key prior) """
    scores = w * z["rmvpe_mix"] + (1 - w) * z[model + "_mix"]
    if use_vocals:
        scores = (1 - v) * scores + v * (w * z["rmvpe_vocals"] + (1 - w) * z[model + "_vocals"])
    context = {key[4:]: z[key].tolist() for key in z if key.startswith("ctx_")}
    return repetition_pooling.pool_repetitions(scores, weight=DetectionPipeline.REPETITION_WEIGHT, **context)


def key_prior(scores, k):
    """ the pitch classes after the soft key prior of weight k """
    return scores.argmax(axis=1) if k == 0 else StochasticPostprocessor.correct_scores(scores, weight=k)


def accuracy(pred, ref):
    sh = tc.estimate_shift_semitones(pred, ref)
    d = ((pred + sh.shift_semitones) % 12 - ref) % 12
    d = np.minimum(d, 12 - d)
    return int((d == 0).sum()), int((d <= 1).sum()), len(d)


def evaluate(args) -> int:
    files = sorted(CACHE.glob("*.npz"))
    songs = [dict(np.load(f, allow_pickle=False)) for f in files]
    songs = [z for z in songs if "rmvpe_vocals" in z]   # same songs for mix and vocals configurations
    models = sorted({k[:-4] for k in songs[0] if k.endswith("_mix") and k != "rmvpe_mix" and not k.startswith("tuning_")},
                    key=lambda m: (m != "old", m))
    if args.models:
        models = [m for m in models if m in args.models.split(",")]
    w_grid, v_grid, k_grid = (tuple(float(x) for x in g.split(",")) if g else d for g, d in
                              ((args.w, RMVPE_WEIGHTS), (args.v, VOCALS_WEIGHTS), (args.k, PRIOR_WEIGHTS)))
    songs = [z for z in songs if all(m + "_vocals" in z for m in models)]
    print("{0} validation songs with mix and vocals ({1} cached), {2} notes; models: {3}".format(
        len(songs), len(files), sum(len(z["ref"]) for z in songs), ", ".join(models)))
    results = {}
    t0 = time.time()
    for model, w, use_vocals in itertools.product(models, w_grid, (False, True)):
        for v in v_grid if use_vocals else (None,):
            pooled_all = [pooled_scores(z, model, w, v, use_vocals) for z in songs]
            for k in k_grid:
                results[(model, w, v, k, use_vocals)] = np.array(
                    [accuracy(key_prior(sc, k), z["ref"]) for sc, z in zip(pooled_all, songs)], dtype=float)
    print("grid of {0} settings in {1:.0f} s".format(len(results), time.time() - t0))

    def pooled(key):
        h, near, n = results[key].sum(axis=0)
        return h / n, near / n

    rng = np.random.default_rng(0)

    def diff(key_a, key_b, samples=4000):
        a, b = results[key_a], results[key_b]
        idx = rng.integers(0, len(a), (samples, len(a)))
        d = b[idx, 0].sum(axis=1) / b[idx, 2].sum(axis=1) - a[idx, 0].sum(axis=1) / a[idx, 2].sum(axis=1)
        return [100 * x for x in (pooled(key_b)[0] - pooled(key_a)[0], np.percentile(d, 2.5), np.percentile(d, 97.5))]

    cur_w, cur_v, cur_k = DetectionPipeline.RMVPE_WEIGHT, DetectionPipeline.VOCALS_WEIGHT, StochasticPostprocessor.PRIOR_WEIGHT
    print("current weights: RMVPE w {0}, vocals v {1}, key prior k {2}".format(cur_w, cur_v, cur_k))
    for use_vocals in (False, True):
        label = "mix + vocals (vocals+pp)" if use_vocals else "mix (ensemble+pp)"
        table_v = cur_v if cur_v in v_grid else v_grid[-1]
        print("\n=== " + label + ": exact %, rows RMVPE weight w, columns key prior k" + (
            ", vocals weight v = {0}".format(table_v) if use_vocals else ""))
        for model in models:
            print("  {0:16s}".format(model) + "".join("   k={0:<5}".format(k) for k in k_grid))
            for w in w_grid:
                cells = [pooled((model, w, table_v if use_vocals else None, k, use_vocals))[0] for k in k_grid]
                print("    w={0:<4}        ".format(w) + "".join("{0:9.2f} ".format(100 * c) for c in cells))
        cur = {m: (m, cur_w, cur_v if use_vocals else None, cur_k, use_vocals) for m in models}
        if "old" not in models or not all(key in results for key in cur.values()):
            # without the old model or the current weights in the grid: only the best weights per model
            for m in models:
                b = max([key for key in results if key[0] == m and key[4] == use_vocals], key=lambda key: pooled(key)[0])
                print("  best {0}: w {1}, v {2}, k {3}: exact {4:.2%}, within 1 {5:.2%}".format(m, b[1], b[2], b[3], *pooled(b)))
                if use_vocals:
                    print("    v at w {0}, k {1}: ".format(b[1], b[3]) + ", ".join(
                        "{0}: {1:.2%}".format(v, pooled((m, b[1], v, b[3], True))[0]) for v in v_grid))
            continue
        best = {m: max([key for key in results if key[0] == m and key[4] == use_vocals], key=lambda key: pooled(key)[0])
                for m in models}
        print("  {0:16s} {1:>18s} {2:>40s} {3:>28s}".format("model", "current weights", "best weights (w, v, k): exact / within 1",
                                                       "best - old at current"))
        for m in models:
            b = best[m]
            line = "  {0:16s} {1:8.2%} / {2:7.2%}   w {3}, v {4}, k {5}: {6:7.2%} / {7:7.2%}".format(
                m, *pooled(cur[m]), b[1], b[2], b[3], *pooled(b))
            line += "   {0:+.2f} [{1:+.2f} .. {2:+.2f}]".format(*diff(cur["old"], b))
            print(line)
        for m in models[1:]:
            print("  {0} - old, both at current weights: {1:+.2f} [{2:+.2f} .. {3:+.2f}] points; best - current weights "
                  "of {0}: {4:+.2f} [{5:+.2f} .. {6:+.2f}]".format(m, *diff(cur["old"], cur[m]), *diff(cur[m], best[m])))
        for i, a in enumerate(models[1:]):
            for b in models[2 + i:]:
                print("  {0} - {1}, both at their best weights: {2:+.2f} [{3:+.2f} .. {4:+.2f}] points".format(
                    b, a, *diff(best[a], best[b])))
        if use_vocals and "old" in models:
            for m in models[1:]:
                print("  vocals weight v for {0} at its best w, k: ".format(m) + ", ".join(
                    "v={0}: {1:.2%}".format(v, pooled((m, best[m][1], v, best[m][3], True))[0]) for v in VOCALS_WEIGHTS))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--model", action="append", default=[], help="NAME=PATH of a frame model onnx (repeatable)")
    ex.add_argument("--limit", type=int, default=0)
    ex.add_argument("--overwrite", action="store_true")
    ex.add_argument("--update", action="store_true", help="cached songs: add the missing models (and with --tuned "
                    "the tuning estimate and the corrected scores; the vocals, when --sources has them and the cache "
                    "only the mix); RMVPE only for a missing tuning estimate or source; songs not in the cache are skipped")
    ex.add_argument("--songs", default="val", choices=["val", "train", "list"],
                    help="val (default), train (training songs of pitchnet_2026, fixed random order) or list (--list)")
    ex.add_argument("--list", action="append", default=[], help="song path list (with --songs list, repeatable)")
    ex.add_argument("--root", default=None, help="root of the paths in --list")
    ex.add_argument("--cache", default=str(CACHE), help="output folder (default: the validation cache)")
    ex.add_argument("--shard", default="", help="i/n: every n-th song starting at i (parallel processes)")
    ex.add_argument("--no-old", action="store_true", help="without the old pitchnet")
    ex.add_argument("--sources", default="mix,vocals", help="mix or mix,vocals (default)")
    ex.add_argument("--rmvpe-threads", type=int, default=0, help="threads of RMVPE and the pitch models (default: all)")
    ex.add_argument("--cpu", action="store_true", help="RMVPE on the CPU")
    ex.add_argument("--tuned", action="store_true", help="also the frame models with the tuning correction "
                    "(NAME_tuned_SOURCE, always corrected; tuning_SOURCE = [offset cents, reliable])")
    ev_ = sub.add_parser("evaluate")
    ev_.add_argument("--models", default="", help="comma separated (default: all cached)")
    ev_.add_argument("--w", default="", help="RMVPE weights, comma separated (default {0})".format(RMVPE_WEIGHTS))
    ev_.add_argument("--v", default="", help="vocals weights (default {0})".format(VOCALS_WEIGHTS))
    ev_.add_argument("--k", default="", help="key prior weights (default {0})".format(PRIOR_WEIGHTS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    return extract(args) if args.cmd == "extract" else evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
