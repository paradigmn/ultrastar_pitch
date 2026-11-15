#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rmvpe_eval_batch.py – rmvpe_eval.py over many songs, with a summary
===================================================================

Purpose
-------
Turn the single-song check (rmvpe_eval.py) into a picture across many songs: how well
does RMVPE match the UltraStar notes, does it break down for certain genres or
quality levels, and is vocal separation worth it compared to the raw mix?

Input
-----
Any number of positional arguments:
    - folder               all notes files in it, recursively (.txt with #BPM)
    - text file of paths   one notes.txt per line ('#' = comment), relative
                           paths are resolved against --root (default: folder of the list)
    - single notes.txt     (recognized by a #BPM line)
Filters as in label_quality.py: --exclude FILE (path list), --skip-names
(default instrumental,karaoke), duets are skipped (--include-duets).
Song lists live in dev/lists/ (e.g. rmvpe_batch_songs.txt, exclude.txt).

Audio per song (from the headers of the notes.txt, relative to the file's folder)
---------------------------------------------------------------------------------
    Mix        #AUDIO, otherwise (V1/unversioned) #MP3
    Vocals     in this order: file from #VOCALS, vocals.wav in the folder
               (if vocals.wav.json matches the mix audio or is missing), matching
               vocals_<name>.wav, file in --vocals-cache-dir
    Separated  only with --separate MODEL: missing vocals are separated and stored in
               --vocals-cache-dir (slow; one Demucs model per process)
Absolute paths and '..' in audio headers are rejected. The cache lookup follows
the rules of ProjectParser.find_vocals_file (vocals.wav or vocals_<audioname>.wav;
with vocals.wav.json only if audio name and size match) and was checked against
the parser on 97 songs (same result).

Selection
---------
--sample N              N random songs (--seed)
--stratify genre        spread evenly over #GENRE (round-robin draw)
--stratify quality      spread evenly over quartiles of quality_score (needs
                        --quality-csv from label_quality.py)

Output
------
--out CSV (default rmvpe_batch.csv; results are usually kept in dev/results/): one row
per song x source x threshold. It is appended as songs finish, --resume skips songs
already in it (error rows only without --retry-errors). Columns: path, genre, status,
source, threshold, bpm, notes, evaluated, audio_s, voiced_share, acc_raw, acc_tuning,
acc_shift, acc_timing, acc_control, tuning_cents, tuning_R, shift, shift_agreement,
offset_ms, shift_timing, err0..err3 (number of notes per semitone distance, stage
Timing*), acc_weighted, n_b0..n_b5 / ok_b0..ok_b5 (notes and hits per note length),
secs, acc_all (PN-* only). --notes-csv FILE additionally writes every single note.

Existing classifier (--pitchnet [ONNX])
---------------------------------------
Two more rows per audio source: PN-<source> (pitchnet without postprocessing) and
PN-<source>+PP (with statistical postprocessing, the CLI default). Preprocessing,
audio decoding (int16 via ffmpeg) and segment boundaries as in the tool; checked
against DetectionPipeline.transform (identical predictions). The time offset for
Timing* comes from the RMVPE search of the same source and threshold, evaluated on
the same notes as RMVPE (paired). acc_all = Timing* over all notes. Details:
rmvpe_eval.py.

Summary (at the end or with --summary-only CSV)
-----------------------------------------------
    1. Songs: ok / skipped / error (with reasons)
    2. Accuracy per source and threshold: mean, median, 10th percentile over songs,
       pooled over notes, share of songs >= 80 %
    3. Paired comparisons per song (Mix vs. Vocals/Separated, RMVPE vs.
       pitchnet, with/without postprocessing): mean difference with 95 % bootstrap
       interval (songs as the sample), better/worse counts; stages raw,
       +transp., +Timing* and tolerant (exact+±1, soft) per song
    4. Which correction matters: songs with transposition != 0, with a timing gain
       > 5 points, with |time offset| > 100 ms, with |tuning| > 15 cents
    5. Accuracy by note length (pooled)
    5b. Error distance and tolerant accuracy (exact+±1, up to ±2, 'soft' with
        Gaussian weights, --soft-sigma), also per quality_score group
    6. Accuracy per genre (>= 3 songs)
    7. The 10 weakest songs (primary source)
    8. With --quality-csv: rank correlation with quality_score/pitch_score, comparison
       of time offset and transposition with label_quality.py

Reading notes
-------------
    - Time offset > 0: the audio is later than the notes (like offset_ms in
      label_quality.py). The sign of shift_semitones there is not verified,
      so "equal" and "reversed sign" are counted separately.
    - Timing* is an upper bound (search over many offsets). Both stages are fine for
      comparisons between sources; "+transp." involves no search.
    - Accuracy is per note (pitch class matches), not per frame.
    - The confidence interval treats songs as an independent sample. With few
      songs it is wide, which is intended.

Parallel
--------
--workers N starts N processes (each with its own RMVPE, --threads per process). With
N = 1 everything runs in the main process. Each process holds the model and activations
in RAM, with --separate also the Demucs model (304 MB plus activations).
--gpu runs RMVPE (and --separate) on CUDA; the results are identical to the CPU
(checked on 24 songs). Measured on the workstation (Xeon Silver 4110, RTX 5000), 24
songs, Mix only, with --pitchnet: CPU 4 workers x 4 threads 198 s; GPU 1 worker 123 s,
2: 72 s, 4 (x 2 threads): 46 s, 8 (x 2 threads): 39 s at about 660 MiB GPU memory per
worker. With the GPU, ffmpeg and the time offset search (CPU) are the bottleneck, not
RMVPE.

Dependencies: numpy, scipy (Spearman), onnxruntime, ffmpeg; rmvpe_eval.py and
tuning_correction.py in the same folder. --separate: vocal_separator.py. Models live
in models/ (default --model: models/rmvpe.onnx).

Usage (from the repo root)
--------------------------
    python dev/rmvpe_eval_batch.py dev/lists/rmvpe_batch_songs.txt --root . --pitchnet --gpu \\
        --workers 8 --threads 2 --quality-csv dev/results/scores.csv --out dev/results/rmvpe_batch.csv
    python dev/rmvpe_eval_batch.py index.txt --root ~/songs --sample 40 --stratify genre \\
        --workers 2 --threads 4 --out dev/results/rmvpe_batch.csv
    python dev/rmvpe_eval_batch.py ~/songs/Rock --sample 20 --no-vocals
    python dev/rmvpe_eval_batch.py --summary-only dev/results/rmvpe_batch.csv \\
        --quality-csv dev/results/scores.csv
    python dev/rmvpe_eval_batch.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import rmvpe_eval as ev                     # noqa: E402

BUCKET_EDGES_MS = [0, 60, 100, 200, 400, 800, float("inf")]
BUCKET_LABELS = ["<60", "60-100", "100-200", "200-400", "400-800", ">800"]
N_BUCKETS = len(BUCKET_LABELS)

COLUMNS = (["path", "genre", "status", "source", "threshold", "bpm", "notes", "evaluated",
            "audio_s", "voiced_share", "acc_raw", "acc_tuning", "acc_shift", "acc_timing",
            "acc_control", "tuning_cents", "tuning_R", "shift", "shift_agreement",
            "offset_ms", "shift_timing", "err0", "err1", "err2", "err3", "acc_weighted"]
           + [f"n_b{i}" for i in range(N_BUCKETS)] + [f"ok_b{i}" for i in range(N_BUCKETS)]
           + ["secs", "acc_all"])

# order of the sources in the summary (PN-* = existing classifier)
SOURCE_ORDER = ["Mix", "Vocals", "Separated", "PN-Mix", "PN-Mix+PP", "PN-Vocals", "PN-Vocals+PP",
                "PN-Separated", "PN-Separated+PP"]
# source labels of CSVs written by older versions of this script
LEGACY_SOURCES = {"Getrennt": "Separated", "PN-Getrennt": "PN-Separated", "PN-Getrennt+PP": "PN-Separated+PP"}
LEGACY_STATUS = {"skipped:duett": "skipped:duet"}
LEGACY_GENRES = {"unbekannt": "unknown"}
# paired comparisons (base, source): difference = source - base
PAIRS = [("Mix", "Vocals"), ("Mix", "Separated"),
         ("Mix", "PN-Mix"), ("Mix", "PN-Mix+PP"), ("PN-Mix", "PN-Mix+PP"),
         ("Vocals", "PN-Vocals"), ("Vocals", "PN-Vocals+PP"),
         ("PN-Mix", "PN-Vocals"), ("PN-Mix+PP", "PN-Vocals+PP"),
         ("PN-Mix", "PN-Separated"), ("PN-Mix+PP", "PN-Separated+PP")]


# --------------------------------------------------------------------------- #
# Find, filter and select songs
# --------------------------------------------------------------------------- #
def looks_like_notes(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4096).decode("utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "#bpm" in head


def collect_songs(inputs: list[str], root: str | None) -> list[str]:
    songs: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.rglob("*.txt")):
                if looks_like_notes(f):
                    songs.append(str(f))
        elif p.is_file():
            if looks_like_notes(p):
                songs.append(str(p))
            else:                                    # path list
                base = Path(root) if root else p.parent
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    q = Path(line)
                    songs.append(str(q if q.is_absolute() else base / q))
        else:
            raise ev.EvalError(f"input not found: {item}")
    seen, out = set(), []
    for s in songs:
        key = os.path.normpath(s)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def read_header(path: str) -> dict:
    header: dict[str, str] = {}
    try:
        for line in ev._read_text(Path(path)).splitlines():
            line = line.strip()
            if line.startswith("#"):
                k, _, v = line[1:].partition(":")
                header.setdefault(k.strip().upper(), v.strip())
            elif line and line[0] in ":*-EFRGP":
                break
    except OSError:
        pass
    return header


def genre_of(header: dict) -> str:
    g = header.get("GENRE", "").strip().lower()
    g = re.split(r"[,;/]", g)[0].strip() if g else ""
    return g or "unknown"


def select_songs(songs, args, quality) -> list[str]:
    if args.exclude:
        skip = {os.path.normpath(x.strip()) for x in
                Path(args.exclude).read_text(encoding="utf-8", errors="replace").splitlines()
                if x.strip() and not x.strip().startswith("#")}
        songs = [s for s in songs if s not in skip]
    words = [w.strip().lower() for w in args.skip_names.split(",") if w.strip()]
    songs = [s for s in songs if not any(w in Path(s).name.lower() for w in words)]
    if args.sample and args.sample < len(songs):
        rng = np.random.default_rng(args.seed)
        if args.stratify:
            groups: dict[str, list[str]] = defaultdict(list)
            if args.stratify == "genre":
                for s in songs:
                    groups[genre_of(read_header(s))].append(s)
            else:
                scores = {s: lookup_quality(quality, s, args.root).get("quality_score") for s in songs}
                known = np.array([v for v in scores.values() if v is not None])
                if known.size < 8:
                    raise ev.EvalError("--stratify quality needs --quality-csv with matching paths")
                qs = np.quantile(known, [0.25, 0.5, 0.75])
                for s, v in scores.items():
                    groups["unknown" if v is None else f"Q{int(np.searchsorted(qs, v)) + 1}"].append(s)
            for g in groups.values():
                rng.shuffle(g)
            names = sorted(groups)
            picked: list[str] = []
            while len(picked) < args.sample and any(groups[n] for n in names):
                for n in names:
                    if groups[n] and len(picked) < args.sample:
                        picked.append(groups[n].pop())
            songs = picked
        else:
            songs = [songs[i] for i in sorted(rng.choice(len(songs), args.sample, replace=False))]
    if args.limit:
        songs = songs[:args.limit]
    return songs


# --------------------------------------------------------------------------- #
# Audio per song
# --------------------------------------------------------------------------- #
def find_file(folder: Path, name: str) -> Path | None:
    if not name:
        return None
    if os.path.isabs(name) or ".." in Path(name).parts:
        raise ev.EvalError(f"unsafe audio path in header: {name!r}")
    target = folder / name
    if target.is_file():
        return target
    if target.parent.is_dir():                       # tolerate case differences
        for c in target.parent.iterdir():
            if c.name.lower() == Path(name).name.lower() and c.is_file():
                return c
    return None


def find_mix(notes: ev.Notes, folder: Path) -> Path:
    h = notes.header or {}
    names = [h.get("AUDIO", "")] + ([] if notes.v2 else [h.get("MP3", "")])
    names = [n for n in names if n]
    if not names:
        raise ev.EvalError("no #AUDIO/#MP3 in header")
    for n in names:
        f = find_file(folder, n)
        if f:
            return f
    raise ev.EvalError(f"audio file missing: {names[0]}")


def find_vocals(notes: ev.Notes, folder: Path, mix: Path, cache_dir: str | None,
                notes_path: str) -> Path | None:
    h = notes.header or {}
    f = find_file(folder, h.get("VOCALS", "")) if h.get("VOCALS") else None
    if f:
        return f
    # same rules as ProjectParser.find_vocals_file: vocals.wav or vocals_<audio>.wav;
    # with .json only if audio name and size match; vocals.wav without .json counts as hand-made
    try:
        names = {c.name.lower(): c for c in folder.iterdir()}
    except OSError:
        names = {}
    for cand in ("vocals.wav", f"vocals_{mix.stem.lower()}.wav"):
        f = names.get(cand)
        if f is None or not f.is_file():
            continue
        meta = names.get(cand + ".json")
        if meta is None:
            if cand == "vocals.wav":
                return f
            continue
        if _vocals_origin_ok(meta, mix):
            return f
    if cache_dir:
        c = cached_vocals_path(cache_dir, notes_path)
        if c.is_file():
            return c
    return None


def _vocals_origin_ok(meta: Path, mix: Path) -> bool:
    """vocals.wav.json names this mix audio (name, case-insensitive, and size)."""
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
        return (str(info.get("audio", "")).lower() == mix.name.lower()
                and info.get("size") == mix.stat().st_size)
    except (OSError, ValueError, AttributeError):
        return False


def cached_vocals_path(cache_dir: str, notes_path: str) -> Path:
    h = hashlib.sha1(os.path.abspath(notes_path).encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / f"{h}.wav"


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
_STATE: dict = {}


def worker_init(cfg: dict) -> None:
    _STATE["cfg"] = cfg
    _STATE["model"] = ev.Rmvpe(cfg["model"], cfg["threads"], cfg.get("gpu", False))
    _STATE["separator"] = None
    _STATE["pitchnet"] = (ev.Pitchnet(cfg["pitchnet"] or None, cfg["threads"])
                          if cfg.get("pitchnet") is not None else None)


def _separator():
    cfg = _STATE["cfg"]
    if _STATE["separator"] is None:
        try:
            from ultrastar_pitch.vocal_separator import VocalSeparator
        except ImportError:
            from vocal_separator import VocalSeparator
        _STATE["separator"] = VocalSeparator(cfg["separate"], num_threads=cfg["threads"],
                                             providers=ev.providers_for(cfg.get("gpu", False)))
    return _STATE["separator"]


def separate_to_cache(mix: Path, target: Path) -> Path:
    from math import gcd
    from scipy.signal import resample_poly
    wav = ev.load_audio(str(mix), 44100, channels=2)
    vocals = _separator().separate(wav).mean(axis=0)
    g = gcd(44100, ev.SR)
    audio = resample_poly(vocals, ev.SR // g, 44100 // g).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    ev.save_wav16(str(target), audio)
    return target


def _f(x: float, fmt: str = ".4f") -> str:
    return "" if x != x else format(x, fmt)


def make_row(path, genre, label, thr, r: "ev.Result", notes: ev.Notes, audio_s, secs) -> dict:
    ref = notes.ref_classes
    ok = r.final_classes >= 0
    dist = np.minimum((r.final_classes[ok] - ref[ok]) % 12, (ref[ok] - r.final_classes[ok]) % 12)
    len_ms = (notes.ends - notes.starts)[ok] * 1000.0
    row = {"path": path, "genre": genre, "status": "ok", "source": label, "threshold": f"{thr:.2f}",
           "bpm": f"{notes.bpm:g}", "notes": r.n_notes, "evaluated": int(ok.sum()),
           "audio_s": f"{audio_s:.1f}", "voiced_share": f"{r.voiced_share:.4f}",
           "acc_raw": _f(r.acc_raw), "acc_tuning": _f(r.acc_tuning),
           "acc_shift": _f(r.acc_shift), "acc_timing": _f(r.acc_timing),
           "acc_control": _f(r.acc_control), "acc_all": _f(r.acc_all),
           "tuning_cents": _f(r.tuning.offset_cents, ".1f"), "tuning_R": _f(r.tuning.strength, ".3f"),
           "shift": r.shift, "shift_agreement": f"{r.shift_agreement:.4f}",
           "offset_ms": r.offset_ms, "shift_timing": r.final_shift,
           "err0": int(np.sum(dist == 0)), "err1": int(np.sum(dist == 1)),
           "err2": int(np.sum(dist == 2)), "err3": int(np.sum(dist >= 3)),
           "acc_weighted": _f(r.acc_weighted),
           "secs": f"{secs:.1f}"}
    for i in range(N_BUCKETS):
        sel = (len_ms >= BUCKET_EDGES_MS[i]) & (len_ms < BUCKET_EDGES_MS[i + 1])
        row[f"n_b{i}"] = int(sel.sum())
        row[f"ok_b{i}"] = int(np.sum(dist[sel] == 0))
    return row


def process_song(path: str) -> dict:
    cfg = _STATE["cfg"]
    t0 = time.time()
    out = {"path": path, "genre": "", "status": "ok", "rows": [], "note_rows": []}
    try:
        notes = ev.parse_notes(path, cfg["include_duets"])
        folder = Path(path).parent
        out["genre"] = genre_of(notes.header or {})
        mix = find_mix(notes, folder)
        sources = [("Mix", mix)]
        if not cfg["no_vocals"]:
            vp = find_vocals(notes, folder, mix, cfg["vocals_cache_dir"], path)
            if vp:
                sources.append(("Vocals", vp))
            elif cfg["separate"]:
                target = cached_vocals_path(cfg["vocals_cache_dir"], path)
                sources.append(("Separated", separate_to_cache(mix, target)))
        an = argparse.Namespace(**cfg["analysis"])
        model, pn = _STATE["model"], _STATE["pitchnet"]
        for label, apath in sources:
            audio = ev.load_audio(str(apath), ev.SR)
            sal = model.salience(audio)
            results = [ev.analyze(label, sal, notes, thr, an) for thr in cfg["thresholds"]]
            if pn is not None:
                a16, cache = ev.load_audio_int16(str(apath)), {}
                results += [x for rm in results[:len(cfg["thresholds"])]
                            for x in ev.analyze_pitchnet(label, pn, a16, notes, rm, cache)]
            for r in results:
                thr = r.threshold
                out["rows"].append(make_row(path, out["genre"], r.label, thr, r, notes,
                                            len(audio) / ev.SR, time.time() - t0))
                if cfg["want_notes"]:
                    for i in range(len(notes.starts)):
                        out["note_rows"].append([path, r.label, f"{thr:.2f}", f"{notes.starts[i]:.3f}",
                                                 f"{notes.ends[i]:.3f}", int(notes.pitches[i]),
                                                 int(notes.pitches[i] % 12), int(r.raw_classes[i]),
                                                 int(r.final_classes[i])])
    except ev.DuetError:
        out["status"] = "skipped:duet"
    except ev.EvalError as e:
        out["status"] = f"error:{str(e)[:120]}"
    except Exception as e:                                   # noqa: BLE001
        out["status"] = f"error:{type(e).__name__}: {str(e)[:100]}"
    out["secs"] = time.time() - t0
    return out


# --------------------------------------------------------------------------- #
# Quality CSV (label_quality.py)
# --------------------------------------------------------------------------- #
def load_quality(path: str | None, key: str | None):
    if not path:
        return None
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = rows[0].keys()
    key = key or next((c for c in ("path", "file", "notes", "notes_path", "txt", "filename")
                       if c in cols), None)
    if key is None or key not in cols:
        raise ev.EvalError(f"path column not found in {path} (available: {list(cols)}); "
                           "specify it with --quality-key")

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    by_full, by_tail = {}, {}
    for r in rows:
        rec = {k: num(v) for k, v in r.items() if k != key}
        p = os.path.normpath(r[key])
        by_full[p] = rec
        by_tail[Path(p).parts[-3:]] = rec
    return {"full": by_full, "tail": by_tail}


def lookup_quality(q, song_path: str, root: str | None) -> dict:
    if not q:
        return {}
    p = os.path.normpath(song_path)
    cands = [p, os.path.normpath(os.path.abspath(p))]
    if root:
        cands.append(os.path.normpath(os.path.relpath(p, root)))
    for c in cands:
        if c in q["full"]:
            return q["full"][c]
    return q["tail"].get(Path(p).parts[-3:], {})


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_results(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["source"] = LEGACY_SOURCES.get(r.get("source") or "", r.get("source"))
        r["status"] = LEGACY_STATUS.get(r.get("status") or "", r.get("status"))
        r["genre"] = LEGACY_GENRES.get(r.get("genre") or "", r.get("genre"))
    return rows


def pooled(rows, key_ok="err0") -> float:
    ok = sum(fnum(r["err0"]) for r in rows)
    tot = sum(fnum(r[k]) for r in rows for k in ("err0", "err1", "err2", "err3"))
    return ok / tot if tot else float("nan")


W = 15        # width of the source column


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x[~np.isnan(x)])) if np.any(~np.isnan(x)) else float("nan")


def p(x) -> str:
    return "   -  " if x != x else f"{100 * x:5.1f}%"


def bootstrap_ci(d: np.ndarray, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def err_stats(rows: list[dict], sigma: float):
    """Pooled error distances (stage Timing*) and tolerant accuracies.
    Returns: exact, ±1, ±2, >=3, tol1 (exact+±1), tol2, soft (Gaussian weights)."""
    e = [sum(fnum(r.get(f"err{i}")) for r in rows if r.get(f"err{i}") not in (None, "")) for i in range(4)]
    n = sum(e)
    if n == 0:
        return (float("nan"),) * 7
    w = [math.exp(-(d * d) / (2.0 * sigma * sigma)) for d in (1, 2, 3)]
    soft = (e[0] + w[0] * e[1] + w[1] * e[2] + w[2] * e[3]) / n
    return (e[0] / n, e[1] / n, e[2] / n, e[3] / n, (e[0] + e[1]) / n,
            (e[0] + e[1] + e[2]) / n, soft)


def summarize(rows: list[dict], quality, root, sigma: float = 1.32) -> None:
    songs_status: dict[str, str] = {}
    for r in rows:
        songs_status.setdefault(r["path"], r["status"])
    cnt = Counter(("ok" if s == "ok" else s.split(":")[0] + ":" + s.split(":", 1)[1][:60]
                   if s.startswith("error") else s) for s in songs_status.values())
    print(f"\n=== Summary: {len(songs_status)} songs ===")
    print("Status: " + ", ".join(f"{k}: {v}" for k, v in cnt.most_common()))
    good = [r for r in rows if r["status"] == "ok" and r["source"]]
    if not good:
        print("No evaluable songs.")
        return
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in good:
        groups[(r["source"], r["threshold"])].append(r)
    order = sorted(groups, key=lambda k: (SOURCE_ORDER.index(k[0]) if k[0] in SOURCE_ORDER else 99,
                                          k[0], k[1]))

    print("\n-- Accuracy per source/threshold (mean over songs; 'pooled' over all notes) --")
    print(f"{'Source':<{W}}{'Thr.':>6}{'Songs':>6}{'Notes':>7} |{'raw':>7}{'+tuning':>8}{'+transp.':>9}"
          f"{'+Timing*':>9} |{'Median+T':>9}{'P10+T':>8}{'>=80%':>7} |{'pooled*':>9}{'all*':>8}")
    for k in order:
        g = groups[k]
        a = lambda col: np.array([fnum(r.get(col)) for r in g])       # noqa: E731
        sh = a("acc_shift")
        print(f"{k[0]:<{W}}{k[1]:>6}{len(g):>6}{int(sum(a('evaluated'))):>7} |"
              f"{p(_nanmean(a('acc_raw'))):>7}{p(_nanmean(a('acc_tuning'))):>8}"
              f"{p(_nanmean(sh)):>9}{p(_nanmean(a('acc_timing'))):>9} |"
              f"{p(np.nanmedian(sh)):>9}{p(np.nanpercentile(sh, 10)):>8}"
              f"{p(np.mean(sh >= 0.8)):>7} |{p(pooled(g)):>9}{p(_nanmean(a('acc_all'))):>8}")
    print("   (pooled* = hits/notes at stage Timing*; all* = PN-* only: Timing* over all notes,\n"
          "    including those RMVPE leaves unrated for lack of voiced frames; PN-*: '+tuning' does not apply,\n"
          "    Timing* uses the offset from the RMVPE search of the same source)")

    # paired comparisons
    print("\n-- Paired comparisons per song (difference = source - base, same threshold) --")
    print(f"   raw/+transp./+Timing* as above; exact+±1 and soft (σ={sigma:g}) per song at stage Timing*")
    stages = (("raw", lambda r: fnum(r.get("acc_raw"))),
              ("+transp.", lambda r: fnum(r.get("acc_shift"))),
              ("+Timing*", lambda r: fnum(r.get("acc_timing"))),
              ("exact+±1", lambda r: err_stats([r], sigma)[4]),
              ("soft", lambda r: err_stats([r], sigma)[6]))
    any_pair = False
    for base, other in PAIRS:
        for thr in sorted({k[1] for k in groups}):
            if (base, thr) not in groups or (other, thr) not in groups:
                continue
            any_pair = True
            bm = {r["path"]: r for r in groups[(base, thr)]}
            pairs = [(bm[r["path"]], r) for r in groups[(other, thr)] if r["path"] in bm]
            if len(pairs) < 3:
                print(f"  {other} - {base} @ {thr}: only {len(pairs)} songs with both sources")
                continue
            print(f"  {other} - {base} @ {thr}:")
            for stage, fn in stages:
                d = np.array([fn(b) - fn(a_) for a_, b in pairs])
                d = d[~np.isnan(d)]
                if len(d) < 3:
                    continue
                lo, hi = bootstrap_ci(d)
                print(f"    {stage:<9} n={len(d):>3}: mean {100 * d.mean():+5.1f} pt "
                      f"[95%: {100 * lo:+5.1f} .. {100 * hi:+5.1f}] | better (>2 pt) {int(np.sum(d > 0.02))}, "
                      f"worse {int(np.sum(d < -0.02))}, equal {int(np.sum(np.abs(d) <= 0.02))}")
    if any_pair:
        print("   If the interval contains 0, no difference is established. Base and source from the same\n"
              "   audio file (e.g. Mix vs. PN-Mix) are evaluated on the same notes, Mix vs. Vocals are not.")
    else:
        print("   (no second source)")

    prim = order[0]
    g = groups[prim]
    n = len(g)
    shifts = [int(fnum(r["shift"])) for r in g]
    gain = [fnum(r["acc_timing"]) - fnum(r["acc_shift"]) for r in g]
    print(f"\n-- What drives the accuracy? ({prim[0]} @ {prim[1]}, {n} songs) --")
    print(f"  Transposition != 0: {sum(s != 0 for s in shifts)} songs | Timing* gains > 5 pt: "
          f"{sum(x > 0.05 for x in gain)} | |time offset| > 100 ms: "
          f"{sum(abs(fnum(r['offset_ms'])) > 100 for r in g)} | |tuning| > 15 cents: "
          f"{sum(abs(fnum(r['tuning_cents'])) > 15 for r in g)}")

    print("\n-- Accuracy by note length (pooled, stage Timing*) --")
    print(f"{'Source':<{W}}{'Thr.':>6} | " + " ".join(f"{lab:>14}" for lab in BUCKET_LABELS))
    for k in order:
        cells = []
        for i in range(N_BUCKETS):
            nn = sum(fnum(r[f"n_b{i}"]) for r in groups[k])
            okk = sum(fnum(r[f"ok_b{i}"]) for r in groups[k])
            cells.append(f"{p(okk / nn if nn else float('nan'))} ({int(nn):>5})")
        print(f"{k[0]:<{W}}{k[1]:>6} | " + " ".join(f"{c:>14}" for c in cells))

    w1, w2, w3 = (math.exp(-(d * d) / (2.0 * sigma * sigma)) for d in (1, 2, 3))
    chance = sum(math.exp(-(min(d, 12 - d) ** 2) / (2.0 * sigma * sigma)) for d in range(12)) / 12
    print(f"\n-- Error distance and tolerant accuracy (pooled, stage Timing*) --")
    print(f"   Gaussian weights (σ={sigma:g} semitones): distance 0 = 1.00, ±1 = {w1:.2f}, ±2 = {w2:.2f}, "
          f"≥3 <= {w3:.2f}; chance level 'soft' = {chance:.0%} (exact: 8 %, ±1: 25 %)")
    print(f"{'Source':<{W}}{'Thr.':>6} |{'exact':>7}{'±1':>7}{'±2':>7}{'>=3':>7} |{'exact+±1':>9}"
          f"{'to ±2':>8}{'soft':>8}")
    for k in order:
        ex, e1, e2, e3, t1, t2, so = err_stats(groups[k], sigma)
        print(f"{k[0]:<{W}}{k[1]:>6} |{p(ex):>7}{p(e1):>7}{p(e2):>7}{p(e3):>7} |{p(t1):>9}{p(t2):>8}{p(so):>8}")

    by_genre: dict[str, list[dict]] = defaultdict(list)
    for r in g:
        by_genre[r["genre"] or "unknown"].append(r)
    big = {k: v for k, v in by_genre.items() if len(v) >= 3}
    if not big:
        print(f"\n-- Genre: no group with >= 3 songs ({len(by_genre)} distinct #GENRE values "
              f"across {len(g)} songs), so no breakdown --")
    if big:
        print(f"\n-- Accuracy per genre ({prim[0]} @ {prim[1]}, +transp., >= 3 songs) --")
        for name, v in sorted(big.items(), key=lambda kv: -len(kv[1])):
            accs = np.array([fnum(r["acc_shift"]) for r in v])
            print(f"  {name[:24]:<24} {len(v):>3} songs: mean {p(np.nanmean(accs))}, "
                  f"median {p(np.nanmedian(accs))}, min {p(np.nanmin(accs))}")

    if quality:
        qrows = []
        for r in g:
            qv = lookup_quality(quality, r["path"], root).get("quality_score")
            if qv is not None:
                qrows.append((qv, r))
        if len(qrows) >= 8:
            qrows.sort(key=lambda x: x[0])
            print(f"\n-- Accuracy by quality_score (label_quality.py), four equal-size groups "
                  f"(groups from {prim[0]} @ {prim[1]}, same songs for every source where available) --")
            print(f"{'Group':<17}{'Source':<{W}}{'Songs':>6}{'q range':>14} |{'raw':>7}{'+transp.':>9}"
                  f"{'+Timing*':>9}{'Median T*':>10}{'>=80% T*':>9} |{'exact+±1':>9}{'soft':>8}")
            names = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
            srcs = [k for k in order if k[1] == prim[1]]
            for name, chunk in zip(names, np.array_split(np.arange(len(qrows)), 4)):
                qgrp = [qrows[i] for i in chunk]
                for k in srcs:
                    by_path = {r["path"]: r for r in groups[k]}
                    grp = [by_path[r["path"]] for _, r in qgrp if r["path"] in by_path]
                    if not grp:
                        continue
                    ra = np.array([fnum(r["acc_raw"]) for r in grp])
                    sh = np.array([fnum(r["acc_shift"]) for r in grp])
                    tm = np.array([fnum(r["acc_timing"]) for r in grp])
                    est = err_stats(grp, sigma)
                    print(f"{name if k == srcs[0] else '':<17}{k[0]:<{W}}{len(grp):>6}"
                          f"{qgrp[0][0]:>7.2f}-{qgrp[-1][0]:<6.2f} |{p(_nanmean(ra)):>7}{p(_nanmean(sh)):>9}"
                          f"{p(_nanmean(tm)):>9}{p(np.nanmedian(tm)):>10}{p(np.mean(tm >= 0.8)):>9} |"
                          f"{p(est[4]):>9}{p(est[6]):>8}")
            print("   Beware of the selection effect: quality_score comes from the same audio and pitch evidence as this\n"
                  "   accuracy. Q4 are songs where label_quality.py also finds good agreement; this is not an\n"
                  "   independent estimate for 'clean labels' and rather biased upward. Q1 is dominated by\n"
                  "   label/timing errors. Only listening checks give an independent assessment.")

    print(f"\n-- The 10 weakest songs ({prim[0]} @ {prim[1]}, by +transp.) --")
    for r in sorted(g, key=lambda r: fnum(r["acc_shift"]))[:10]:
        q = lookup_quality(quality, r["path"], root)
        qs = f" q={q['quality_score']:.2f}" if q.get("quality_score") is not None else ""
        print(f"  {p(fnum(r['acc_shift']))} (Timing* {p(fnum(r['acc_timing']))}) offset "
              f"{int(fnum(r['offset_ms'])):+5d} ms, shift {int(fnum(r['shift'])):+d}, tuning "
              f"{fnum(r['tuning_cents']):+4.0f} cents{qs}  {Path(r['path']).parent.name[:40]}/"
              f"{Path(r['path']).name[:30]}")

    if quality:
        print(f"\n-- Comparison with label_quality.py ({prim[0]} @ {prim[1]}) --")
        from scipy.stats import spearmanr
        for col in ("quality_score", "pitch_score", "pitch_score_aligned", "timing_score"):
            xs, ys = [], []
            for r in g:
                v = lookup_quality(quality, r["path"], root).get(col)
                if v is not None:
                    xs.append(v), ys.append(fnum(r["acc_shift"]))
            if len(xs) >= 5 and np.std(xs) > 0 and np.std(ys) > 0:
                rho = spearmanr(xs, ys).statistic
                print(f"  Spearman {col:<20} vs. RMVPE accuracy: {rho:+.2f} (n={len(xs)})")
        diffs, same, opp, other, nsh = [], 0, 0, 0, 0
        for r in g:
            q = lookup_quality(quality, r["path"], root)
            if q.get("offset_ms") is not None:
                diffs.append(abs(fnum(r["offset_ms"]) - q["offset_ms"]))
            if q.get("shift_semitones") is not None:
                nsh += 1
                a_, b_ = int(fnum(r["shift"])) % 12, int(q["shift_semitones"]) % 12
                same += a_ == b_
                opp += a_ == (-b_) % 12 and a_ != b_
                other += a_ != b_ and a_ != (-b_) % 12
        if diffs:
            print(f"  Time offset: median |Δ| {np.median(diffs):.0f} ms, {np.mean(np.array(diffs) <= 60):.0%} "
                  f"of songs within 60 ms (n={len(diffs)})")
        if nsh:
            print(f"  Transposition: equal {same}, reversed sign {opp}, different {other} (n={nsh})")


# --------------------------------------------------------------------------- #
# Self test (synthetic songs with a fake tracker)
# --------------------------------------------------------------------------- #
class _FakeRmvpe:
    """FFT pitch estimator in RMVPE format [T, 360] (self test only)."""

    def __init__(self, *a, **k):
        pass

    def salience(self, audio, **kw):
        n = len(audio)
        win = np.hanning(1024)
        frames = n // ev.HOP + 1
        padded = np.pad(audio, (512, 512), mode="reflect")
        sal = np.zeros((frames, ev.N_CLASS), dtype=np.float32)
        j = np.arange(ev.N_CLASS)
        freqs = np.fft.rfftfreq(8192, 1 / ev.SR)
        lo, hi = np.searchsorted(freqs, 70), np.searchsorted(freqs, 1000)
        for i in range(frames):
            seg = padded[i * ev.HOP:i * ev.HOP + 1024] * win
            mag = np.abs(np.fft.rfft(seg, 8192))[lo:hi]
            k = int(mag.argmax())
            if mag[k] < 2.0 or k in (0, len(mag) - 1):
                continue
            a, b, c = np.log(mag[k - 1] + 1e-9), np.log(mag[k] + 1e-9), np.log(mag[k + 1] + 1e-9)
            f = freqs[lo + k] + (0.5 * (a - c) / (a - 2 * b + c)) * (freqs[1] - freqs[0])
            bin_ = (1200 * np.log2(f / 10.0) - ev.CENTS_BASE) / ev.CENTS_STEP
            sal[i] = 0.9 * np.exp(-0.5 * ((j - bin_) / 1.0) ** 2)
        return sal


def _selftest() -> int:
    import tempfile
    from scipy.io import wavfile
    rng = np.random.default_rng(3)
    sr = 22050
    root = Path(tempfile.mkdtemp())

    def make_song(name, *, shift=0, tuning=0.0, offset_ms=0, genre="Pop", v2=False, noise=0.0,
                  vocals_cache=False, duet=False, missing_audio=False, n_notes=26, bpm=120.0,
                  spacing=(5, 8), dur=(3, 6), interferer=False):
        d = root / name
        d.mkdir()
        beat_s = (60.0 if v2 else 15.0) / bpm
        starts = np.cumsum(rng.integers(spacing[0], spacing[1], n_notes)) - 4
        durs = rng.integers(dur[0], dur[1], n_notes)
        pitches = rng.integers(-6, 8, n_notes)
        total = (starts[-1] + durs[-1]) * beat_s + 2.0
        t = np.arange(int(total * sr)) / sr
        voice = np.zeros_like(t)
        for s, du, pi in zip(starts, durs, pitches):
            a, b = int((1.0 + s * beat_s + offset_ms / 1000) * sr), int((1.0 + (s + du) * beat_s + offset_ms / 1000) * sr)
            f = 440 * 2 ** ((60 + pi - shift + tuning / 100 - 69) / 12)
            voice[a:b] = 0.4 * np.sin(2 * np.pi * f * t[a:b])
        mix = voice + noise * rng.standard_normal(len(t))
        if interferer:                                   # loud sustained tone, like an instrument
            mix = mix + 0.8 * np.sin(2 * np.pi * 311.13 * t)
        wavfile.write(str(d / "song.wav"), sr, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
        if vocals_cache:
            wavfile.write(str(d / "vocals.wav"), sr, (np.clip(voice, -1, 1) * 32767).astype(np.int16))
        head = ["#VERSION:2.0.0"] if v2 else []
        head += [f"#BPM:{bpm}", "#GAP:1000", f"#GENRE:{genre}"]
        head += ["#AUDIO:song.wav"] if (v2 or name.endswith("a")) else ["#MP3:song.wav"]
        if missing_audio:
            head = [h.replace("song.wav", "missing.wav") for h in head]
        body = []
        if duet:
            body.append("P1")
        for s, du, pi in zip(starts, durs, pitches):
            body.append(f": {s} {du} {pi} la")
        if duet:
            body += ["P2", ": 0 2 1 la"]
        (d / "notes.txt").write_text("\n".join(head + body + ["E"]), encoding="utf-8")
        return d / "notes.txt"

    songs = [
        make_song("s1a", shift=2, tuning=25, genre="Rock", noise=0.05),
        make_song("s2", genre="Rock", interferer=True, vocals_cache=True),
        make_song("s3", genre="Pop", offset_ms=400, v2=True, vocals_cache=True, bpm=480.0,
                  spacing=(3, 5), dur=(2, 4)),
        make_song("s4", genre="Pop"),
        make_song("s5", genre="Hip Hop"),
        make_song("duet", duet=True),
        make_song("broken", missing_audio=True),
    ]
    sys.modules[__name__].ev.Rmvpe = _FakeRmvpe
    out_csv = root / "res.csv"
    argv = [str(root), "--out", str(out_csv), "--thresholds", "0.3", "--search-ms", "400",
            "--workers", "1", "--model", "fake", "--pitchnet"]
    rc = main(argv)
    rows = read_results(str(out_csv))
    st = {r["path"]: r["status"] for r in rows}
    ok_rows = [r for r in rows if r["status"] == "ok"]
    by = {(Path(r["path"]).parent.name, r["source"]): r for r in ok_rows}
    checks = {
        "7 songs collected": len(st) == 7,
        "duet skipped": any(v == "skipped:duet" for v in st.values()),
        "missing audio reported as error": any(v.startswith("error:audio file missing") for v in st.values()),
        "5 songs ok": sum(v == "ok" for v in st.values()) == 5,
        "Vocals source for s2": ("s2", "Vocals") in by and ("s1a", "Vocals") not in by,
        "shift s1a = +2": int(by[("s1a", "Mix")]["shift"]) == 2,
        "tuning s1a ~ +25": abs(float(by[("s1a", "Mix")]["tuning_cents"]) - 25) < 10,
        "offset s3 ~ +400": abs(float(by[("s3", "Mix")]["offset_ms"]) - 400) <= 60,
        "V2 timing s3 good": float(by[("s3", "Mix")]["acc_timing"]) > 0.85,
        "Vocals better than noisy mix (s2)":
            float(by[("s2", "Vocals")]["acc_shift"]) > float(by[("s2", "Mix")]["acc_shift"]) + 0.2,
        "pitchnet rows per source": all((n, src) in by for n in ("s1a", "s2", "s3", "s4", "s5")
                                         for src in ("PN-Mix", "PN-Mix+PP"))
            and ("s2", "PN-Vocals+PP") in by and ("s1a", "PN-Vocals") not in by,
        "pitchnet paired (same notes, same offset)": all(
            by[(n, "PN-Mix")]["evaluated"] == by[(n, "Mix")]["evaluated"]
            and by[(n, "PN-Mix+PP")]["offset_ms"] == by[(n, "Mix")]["offset_ms"]
            for n in ("s1a", "s2", "s3", "s4", "s5")),
        "pitchnet values valid": all(0.0 <= float(r["acc_timing"]) <= 1.0 and r["acc_all"] != ""
                                     and r["acc_tuning"] == "" for (n, src), r in by.items()
                                     if src.startswith("PN-")),
        "RMVPE rows without acc_all": all(r["acc_all"] == "" for (n, src), r in by.items()
                                         if not src.startswith("PN-")),
    }
    for n in ("s1a", "s3", "s4"):
        print(f"  Info {n}: RMVPE {by[(n, 'Mix')]['acc_timing']}, pitchnet {by[(n, 'PN-Mix')]['acc_timing']}, "
              f"pitchnet+PP {by[(n, 'PN-Mix+PP')]['acc_timing']} (Timing*, synthetic sine tones)")
    # resume: nothing left to do, no duplicate rows
    main(argv + ["--resume"])
    checks["--resume without duplicates"] = len(read_results(str(out_csv))) == len(rows)
    for k, v in checks.items():
        print(f"  {'OK  ' if v else 'FAIL'}  {k}")
    ok = rc == 0 and all(checks.values())
    print("SELFTEST:", "passed" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Main program
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="rmvpe_eval.py over many songs")
    ap.add_argument("inputs", nargs="*", help="folders, path lists or notes.txt files")
    ap.add_argument("--root", help="base folder for relative paths in lists")
    ap.add_argument("--exclude", help="file with paths to leave out")
    ap.add_argument("--skip-names", default="instrumental,karaoke",
                    help="skip file names containing these words (default instrumental,karaoke)")
    ap.add_argument("--include-duets", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="N random songs")
    ap.add_argument("--stratify", choices=["genre", "quality"], help="spread the sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N (after selection)")
    ap.add_argument("--model", default=str(HERE.parent / "models" / "rmvpe.onnx"))
    ap.add_argument("--thresholds", default="0.15", help="voicing thresholds (default 0.15)")
    ap.add_argument("--no-vocals", action="store_true", help="evaluate the mix only")
    ap.add_argument("--pitchnet", nargs="?", const="", default=None, metavar="ONNX",
                    help="also evaluate the existing classifier as extra sources PN-<source> and "
                         "PN-<source>+PP (without a value: model from ultrastar_pitch/binaries)")
    ap.add_argument("--separate", metavar="DEMUCS.onnx", help="separate missing vocals locally")
    ap.add_argument("--vocals-cache-dir", help="storage for separated vocals (required for --separate)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime threads per process")
    ap.add_argument("--gpu", action="store_true",
                    help="RMVPE and --separate on CUDA (falls back to CPU); every worker holds its own "
                         "model in GPU memory")
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--min-strength", type=float, default=0.05)
    ap.add_argument("--tuning-all", action="store_true")
    ap.add_argument("--search-ms", type=int, default=2000)
    ap.add_argument("--step-ms", type=int, default=20)
    ap.add_argument("--control", action="store_true", help="also compute the random control (slower)")
    ap.add_argument("--out", default="rmvpe_batch.csv")
    ap.add_argument("--notes-csv", help="additionally write every note to this CSV")
    ap.add_argument("--resume", action="store_true", help="skip songs already in --out")
    ap.add_argument("--retry-errors", action="store_true", help="with --resume, recompute error rows")
    ap.add_argument("--summary-only", metavar="CSV", help="only summarize an existing CSV")
    ap.add_argument("--soft-sigma", type=float, default=1.32,
                    help="σ of the Gaussian weights for the 'soft' accuracy, in semitones "
                         "(default 1.32: distance 1 counts 0.75)")
    ap.add_argument("--quality-csv", help="CSV from label_quality.py")
    ap.add_argument("--quality-key", help="path column of the quality CSV")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    try:
        quality = load_quality(args.quality_csv, args.quality_key)
        if args.summary_only:
            summarize(read_results(args.summary_only), quality, args.root, args.soft_sigma)
            return 0
        if not args.inputs:
            ap.print_usage(sys.stderr)
            print("Inputs required (or --summary-only / --selftest).", file=sys.stderr)
            return 2
        if args.separate and not args.vocals_cache_dir:
            raise ev.EvalError("--separate needs --vocals-cache-dir (the vocals are stored there)")
        songs = select_songs(collect_songs(args.inputs, args.root), args, quality)

        done: set[str] = set()
        out_path = Path(args.out)
        if args.resume and out_path.is_file():
            for r in read_results(str(out_path)):
                if r["status"] == "ok" or r["status"].startswith("skipped") or not args.retry_errors:
                    done.add(os.path.normpath(r["path"]))
        todo = [s for s in songs if os.path.normpath(s) not in done]
        print(f"{len(songs)} songs selected, {len(songs) - len(todo)} already done, {len(todo)} pending")

        cfg = {"model": args.model, "threads": args.threads, "include_duets": args.include_duets,
               "no_vocals": args.no_vocals, "separate": args.separate,
               "vocals_cache_dir": args.vocals_cache_dir, "pitchnet": args.pitchnet, "gpu": args.gpu,
               "thresholds": [float(x) for x in args.thresholds.split(",")],
               "want_notes": bool(args.notes_csv),
               "analysis": {"min_frames": args.min_frames, "tuning_all": args.tuning_all,
                            "min_strength": args.min_strength, "search_ms": args.search_ms,
                            "step_ms": args.step_ms, "no_control": not args.control}}
        new_file = not (args.resume and out_path.is_file())
        if not new_file:
            with open(out_path, newline="", encoding="utf-8") as f:
                head = next(csv.reader(f), [])
            if head != COLUMNS:
                raise ev.EvalError(f"{out_path} has different columns (older version); --resume only works with "
                                   "a CSV of this version, otherwise use a new --out file")
        fout = open(out_path, "w" if new_file else "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(fout, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        fnotes = writer_n = None
        if args.notes_csv:
            fnotes = open(args.notes_csv, "w" if new_file else "a", newline="", encoding="utf-8")
            writer_n = csv.writer(fnotes)
            if new_file:
                writer_n.writerow(["path", "source", "threshold", "start_s", "end_s", "file_pitch",
                                   "file_class", "pred_class_raw", "pred_class_final"])

        def handle(res, idx):
            if res["rows"]:
                for row in res["rows"]:
                    writer.writerow(row)
                a = {r["source"]: r for r in res["rows"] if r["threshold"] == res["rows"][0]["threshold"]}
                txt = " ".join(f"{k} {100 * float(v['acc_shift']):.1f}%" for k, v in a.items())
            else:
                writer.writerow({"path": res["path"], "genre": res["genre"], "status": res["status"]})
                txt = res["status"]
            fout.flush()
            if writer_n:
                writer_n.writerows(res["note_rows"])
                fnotes.flush()
            print(f"[{idx}/{len(todo)}] {Path(res['path']).parent.name[:40]}: {txt} ({res['secs']:.0f} s)",
                  flush=True)

        t0 = time.time()
        if todo:
            if args.workers <= 1:
                worker_init(cfg)
                for i, s in enumerate(todo, 1):
                    handle(process_song(s), i)
            else:
                import multiprocessing as mp
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                         initializer=worker_init, initargs=(cfg,)) as pool:
                    futs = [pool.submit(process_song, s) for s in todo]
                    for i, fu in enumerate(as_completed(futs), 1):
                        handle(fu.result(), i)
        fout.close()
        if fnotes:
            fnotes.close()
        print(f"Done in {(time.time() - t0) / 60:.1f} min -> {args.out}")
        summarize(read_results(str(out_path)), quality, args.root, args.soft_sigma)
    except ev.EvalError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
