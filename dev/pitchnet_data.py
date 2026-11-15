#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          pitchnet_data.py
@brief         select, clean, split and cache the training data of pitchnet_2026 (dev/pitchnet_2026.ipynb)
@author        paradigm

plan     scores.csv of label_quality.py -> dev/lists/pitchnet/{train,val,test}.txt, selection.csv, plan.txt
         - songs with status ok, quality_score >= --min-quality, timing_confidence >= --min-timing-conf and
           |drift_ms_per_min| <= --max-drift (a drifting offset means a wrong #BPM, the labels drift too)
         - the evaluation songs (dev/lists/rmvpe_batch_songs*.txt) and everything that may be the same
           song (same folder, same artist/title, same audio, similar lyrics: the duplicate groups of
           select_training_data.py) are excluded, so nothing of the test songs leaks into the training
         - split train / val by the hash of the duplicate group (versions of one song stay together),
           test = the evaluation songs, which are never touched by the training
         Only the csv and the note files are read, no audio.
extract  for every song of the lists: the mix as the tool decodes it (ffmpeg, 16 kHz, mono, int16), the
         saved vocals.wav (16 kHz mono int16, written by the tool's separation) and per 10 ms frame the
         pitch class label (-1 = no label) and the voicing target, into dev/results/pitchnet_cache/:
           <id>.mix.npy, <id>.vocals.npy  (int16, same length)     <id>.labels.npz
         The labels are cleaned: the note times are shifted by offset_ms of label_quality.py (the audio
         is later than the notes by that much), a confident transposition (shift_semitones with a clear
         gain of pitch_score_aligned over pitch_score) is applied to the pitch classes, the first and
         last --trim ms of every note are not labeled (transitions), frames outside the audio are dropped.
         Voicing: 1 inside the notes, 0 farther than --unvoiced-gap s from every note, -1 in between.
show     one cached song: coverage and a sanity check of the alignment (the vocals are louder inside the
         labeled frames than 200 ms before or after, otherwise the offset correction points the wrong way)

    python dev/pitchnet_data.py plan dev/results/scores.csv --min-quality 0.5
    python dev/pitchnet_data.py extract --workers 8 [--limit 2] [--split val]
    python dev/pitchnet_data.py show 00042
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
import wave
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dev"))
from ultrastar_pitch.project_parser import ProjectParser, ProjectParserError  # noqa: E402
import select_training_data as std  # noqa: E402  (duplicate groups, split hash)

LISTS = ROOT / "dev" / "lists" / "pitchnet"
CACHE = ROOT / "dev" / "results" / "pitchnet_cache"
EVAL_LISTS = (ROOT / "dev" / "lists" / "rmvpe_batch_songs.txt", ROOT / "dev" / "lists" / "rmvpe_batch_songs_2021plus.txt")
SR = 16000
HOP = 160                       # 10 ms, the frame grid of the model (and of RMVPE / timing_check)
FRAME_S = HOP / SR

logging.getLogger("ultrastar_pitch").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def number(row: dict, key: str) -> Optional[float]:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def norm_title(header: Dict[str, str]) -> str:
    """ artist - title, lower case, brackets and punctuation removed """
    s = (header.get("#ARTIST", "") + " - " + header.get("#TITLE", "")).lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def read_header(path: str) -> Dict[str, str]:
    header = {}
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return header
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            header.setdefault("#" + key.strip().upper(), value.strip())
        elif line and line[0] in ":*-EFRGP":
            break
    return header


def norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def read_list(path: Path) -> List[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(str((ROOT / line) if not os.path.isabs(line) else Path(line)))
    return out


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(args) -> int:
    with open(args.scores, newline="", encoding="utf-8") as file:
        all_rows = list(csv.DictReader(file))
    rows = [r for r in all_rows if r.get("status") in std.CONTEXT_STATUS]
    by_path = {norm_path(r["path"]): i for i, r in enumerate(rows)}
    log = []

    def say(text=""):
        print(text)
        log.append(text)

    say("{0} rows in {1}, {2} with status ok".format(len(all_rows), args.scores, sum(r["status"] == "ok" for r in rows)))

    # --- duplicate groups over all usable songs (the same logic as select_training_data.py)
    t0 = time.time()
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        prints = list(pool.map(std.fingerprint, [(r["path"], not args.no_audio) for r in rows], chunksize=32))
    groups, _ = std.find_groups(rows, prints, args.lyrics_threshold)
    members = {}
    for i in range(len(rows)):
        members.setdefault(groups.find(i), []).append(i)
    group_name = {i: min(rows[k]["path"] for k in members[groups.find(i)]) for i in range(len(rows))}
    say("duplicate groups: {0} groups of {1} songs, {2} groups with more than one song ({3:.0f} s)".format(
        len(members), len(rows), sum(len(m) > 1 for m in members.values()), time.time() - t0))

    # --- evaluation songs: themselves, their groups, the same folder and the same artist/title
    eval_paths = [p for lst in EVAL_LISTS for p in read_list(lst)]
    excluded = set()
    eval_titles = {norm_title(read_header(p)) for p in eval_paths}
    eval_folders = {norm_path(os.path.dirname(p)) for p in eval_paths}
    reasons = Counter()
    for i, r in enumerate(rows):
        p = norm_path(r["path"])
        if p in {norm_path(e) for e in eval_paths}:
            excluded.add(i); reasons["evaluation song"] += 1
        elif norm_path(os.path.dirname(r["path"])) in eval_folders:
            excluded.add(i); reasons["same folder"] += 1
        elif norm_title(read_header(r["path"])) in eval_titles:
            excluded.add(i); reasons["same artist/title"] += 1
    for i in list(excluded):
        for k in members[groups.find(i)]:
            if k not in excluded:
                excluded.add(k); reasons["duplicate group of an evaluation song"] += 1
    missing = [p for p in eval_paths if norm_path(p) not in by_path]
    say("evaluation songs: {0} ({1} not in the csv, excluded by folder/title only); excluded from the training: {2}".format(
        len(eval_paths), len(missing), ", ".join("{0} {1}".format(v, k) for k, v in reasons.most_common())))

    # --- selection
    dropped = Counter()
    selected = []
    for i, r in enumerate(rows):
        if r["status"] != "ok":
            dropped["status " + r["status"]] += 1; continue
        if i in excluded:
            dropped["evaluation song or related"] += 1; continue
        q, conf, drift = number(r, "quality_score"), number(r, "timing_confidence"), number(r, "drift_ms_per_min")
        if q is None or q < args.min_quality:
            dropped["quality_score < {0}".format(args.min_quality)] += 1; continue
        if conf is None or conf < args.min_timing_conf:
            dropped["timing_confidence < {0}".format(args.min_timing_conf)] += 1; continue
        if drift is None or abs(drift) > args.max_drift:
            dropped["|drift| > {0} ms/min".format(args.max_drift)] += 1; continue
        if (number(r, "notes_beyond_audio") or 0) > 0:
            dropped["notes beyond the audio"] += 1; continue
        selected.append(i)
    say("selected {0} songs; dropped: {1}".format(len(selected), ", ".join("{0} {1}".format(v, k) for k, v in dropped.most_common())))

    # --- label corrections and split
    LISTS.mkdir(parents=True, exist_ok=True)
    fields = ["id", "path", "split", "group", "quality_score", "timing_score", "pitch_score", "pitch_score_aligned",
              "timing_confidence", "offset_ms", "drift_ms_per_min", "shift_semitones", "shift_applied", "n_notes", "duration_s"]
    out_rows = []
    splits = Counter()
    shifted = 0
    notes = Counter()
    for n, i in enumerate(sorted(selected, key=lambda k: rows[k]["path"])):
        r = rows[i]
        shift = number(r, "shift_semitones") or 0.0
        gain = (number(r, "pitch_score_aligned") or 0.0) - (number(r, "pitch_score") or 0.0)
        apply = int(abs(shift) >= 0.5 and gain >= args.shift_min_gain)
        shifted += apply
        split = std.split_of(group_name[i], args.val_fraction, 0.0)
        splits[split] += 1
        notes[split] += int(number(r, "n_notes") or 0)
        out_rows.append({"id": "{0:05d}".format(n), "path": r["path"], "split": split, "group": group_name[i],
                         "shift_applied": int(round(shift)) if apply else 0,
                         **{k: r.get(k, "") for k in fields if k in r}})
    with open(LISTS / "selection.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    for split in ("train", "val"):
        (LISTS / (split + ".txt")).write_text(
            "".join(r["path"] + "\n" for r in out_rows if r["split"] == split), encoding="utf-8")
    (LISTS / "test.txt").write_text("".join(p + "\n" for p in eval_paths), encoding="utf-8")
    say("transposition applied to {0} songs (|shift| >= 0.5 semitones and pitch_score_aligned - pitch_score >= {1})".format(
        shifted, args.shift_min_gain))
    offsets = np.array([number(r, "offset_ms") or 0.0 for r in out_rows])
    say("offset correction: |offset| > 50 ms in {0} songs, > 100 ms in {1}, max {2:.0f} ms".format(
        int((np.abs(offsets) > 50).sum()), int((np.abs(offsets) > 100).sum()), float(np.abs(offsets).max()) if len(offsets) else 0))
    for split in ("train", "val"):
        say("{0}: {1} songs, {2} notes".format(split, splits[split], notes[split]))
    say("test: {0} evaluation songs (dev/lists/rmvpe_batch_songs*.txt), not used for any decision".format(len(eval_paths)))
    size = sum(float(r.get("duration_s") or 0) for r in out_rows) * SR * 2 * 2 / 1e9
    say("cache size for mix + vocals as int16: about {0:.0f} GB".format(size))
    (LISTS / "plan.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("-> {0}".format(LISTS))
    return 0


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
def read_vocals(path: str, parser: ProjectParser, n_samples: int) -> np.ndarray:
    """ vocals.wav as 16 kHz mono int16 of exactly n_samples (padded / cut) """
    with wave.open(path, "rb") as wav:
        if wav.getframerate() == SR and wav.getnchannels() == 1 and wav.getsampwidth() == 2:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        else:
            samples = parser._convert_audio(path, SR)  # other format: let ffmpeg convert it
    out = np.zeros(n_samples, dtype=np.int16)
    n = min(n_samples, len(samples))
    out[:n] = samples[:n]
    return out


def frame_labels(segments: np.ndarray, classes: np.ndarray, n_frames: int, trim_s: float,
                 unvoiced_gap_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ (label per frame -1..11, voicing target per frame -1/0/1, note index per labeled frame) for
        frames centred at t * FRAME_S; a note is labeled between start + trim and end - trim (trim at
        most a quarter of the note) """
    labels = np.full(n_frames, -1, dtype=np.int8)
    note_of = np.full(n_frames, -1, dtype=np.int32)
    voiced = np.zeros(n_frames, dtype=np.int8)
    near = np.zeros(n_frames, dtype=bool)
    centres = np.arange(n_frames) * FRAME_S
    for n, ((start, end), cls) in enumerate(zip(segments, classes)):
        trim = min(trim_s, 0.25 * (end - start))
        lo, hi = np.searchsorted(centres, start + trim), np.searchsorted(centres, end - trim)
        labels[lo:hi] = cls
        note_of[lo:hi] = n
        voiced[np.searchsorted(centres, start):np.searchsorted(centres, end)] = 1
        near[np.searchsorted(centres, start - unvoiced_gap_s):np.searchsorted(centres, end + unvoiced_gap_s)] = True
    voicing = np.where(voiced == 1, 1, np.where(near, -1, 0)).astype(np.int8)
    return labels, voicing, note_of


def extract_song(job: dict) -> dict:
    """ one song -> cache files; returns a summary """
    row, trim_s, unvoiced_gap_s = job["row"], job["trim"], job["unvoiced_gap"]
    sid = row["id"]
    out = {"id": sid, "path": row["path"]}
    parser = ProjectParser()
    try:
        parser.load_note_file(row["path"])
        vocals_path = parser.find_vocals_file()
        if vocals_path is None:
            raise ProjectParserError("no vocals.wav next to the note file")
        mix = parser.load_main_audio(SR)
        vocals = read_vocals(vocals_path, parser, len(mix))
    except (ProjectParserError, OSError, wave.Error) as err:
        out["error"] = str(err)
        return out
    offset_s = (number(row, "offset_ms") or 0.0) / 1000.0
    shift = int(row.get("shift_applied") or 0)
    segments = np.asarray(parser.note_segments(), dtype=np.float64) + offset_s
    classes = (np.asarray(parser.dump_pitches()) + shift) % 12
    n_frames = len(mix) // HOP + 1
    inside = (segments[:, 0] >= 0) & (segments[:, 1] <= len(mix) / SR)
    labels, voicing, note_of = frame_labels(segments[inside], classes[inside], n_frames, trim_s, unvoiced_gap_s)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / (sid + ".mix.npy"), mix.astype(np.int16))
    np.save(CACHE / (sid + ".vocals.npy"), vocals)
    np.savez(CACHE / (sid + ".labels.npz"), labels=labels, voicing=voicing, note_of=note_of,
             segments=segments[inside].astype(np.float32), classes=classes[inside].astype(np.int8),
             offset_s=offset_s, shift=shift, sr=SR, hop=HOP)
    out.update(seconds=len(mix) / SR, notes=int(inside.sum()), notes_dropped=int((~inside).sum()),
               labeled_frames=int((labels >= 0).sum()), voiced=int((voicing == 1).sum()), unvoiced=int((voicing == 0).sum()),
               vocals_source=os.path.basename(vocals_path))
    return out


def extract(args) -> int:
    with open(LISTS / "selection.csv", newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if args.split in ("all", r["split"])]
    if args.limit:
        rows = rows[:args.limit]
    todo = [r for r in rows if args.force or not (CACHE / (r["id"] + ".labels.npz")).exists()]
    print("{0} songs ({1}), {2} to extract -> {3}".format(len(rows), args.split, len(todo), CACHE), flush=True)
    jobs = [{"row": r, "trim": args.trim / 1000.0, "unvoiced_gap": args.unvoiced_gap} for r in todo]
    log = open(CACHE.parent / "pitchnet_extract.log", "a", encoding="utf-8")
    CACHE.mkdir(parents=True, exist_ok=True)
    done = errors = 0
    t0 = time.time()
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        futures = [pool.submit(extract_song, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if "error" in res:
                errors += 1
                line = "{0} ERROR {1}: {2}".format(res["id"], res["path"], res["error"])
            else:
                done += 1
                line = "{id} {seconds:.0f}s notes {notes} (dropped {notes_dropped}) labeled frames {labeled_frames} voiced {voiced} unvoiced {unvoiced} {vocals_source}".format(**res)
            log.write(line + "\n"); log.flush()
            if n % 50 == 0 or n == len(jobs):
                print("[{0}/{1}] {2:.0f} min, {3} errors".format(n, len(jobs), (time.time() - t0) / 60, errors), flush=True)
    print("done: {0} songs, {1} errors (dev/results/pitchnet_extract.log)".format(done, errors))
    return 0


# --------------------------------------------------------------------------- #
# show
# --------------------------------------------------------------------------- #
def show(args) -> int:
    sid = args.id
    mix = np.load(CACHE / (sid + ".mix.npy"), mmap_mode="r")
    vocals = np.load(CACHE / (sid + ".vocals.npy"), mmap_mode="r")
    z = np.load(CACHE / (sid + ".labels.npz"))
    labels, voicing = z["labels"], z["voicing"]
    print("{0}: {1:.0f} s, {2} notes, offset {3:+.0f} ms, shift {4:+d}".format(
        sid, len(mix) / SR, len(z["classes"]), 1000 * float(z["offset_s"]), int(z["shift"])))
    print("frames: {0} labeled ({1:.1f} %), voicing 1: {2}, 0: {3}, ignored: {4}".format(
        int((labels >= 0).sum()), 100 * (labels >= 0).mean(), int((voicing == 1).sum()), int((voicing == 0).sum()), int((voicing == -1).sum())))
    print("classes:", dict(sorted(Counter(labels[labels >= 0].tolist()).items())))
    # alignment check: rms of the vocals per frame, inside the labeled frames vs shifted by +-200 ms
    n = len(labels)
    v = np.asarray(vocals[:n * HOP], dtype=np.float64).reshape(-1, HOP) if len(vocals) >= n * HOP else None
    if v is None:
        v = np.zeros((n, HOP))
        v.flat[:len(vocals)] = vocals
    rms = np.sqrt((v ** 2).mean(axis=1))
    mask = labels >= 0
    print("vocals rms inside the labeled frames by shift of the labels (the peak should be at or near 0):")
    best = None
    for shift_ms in list(range(-400, -100, 100)) + list(range(-100, 101, 10)) + list(range(200, 401, 100)):
        value = rms[np.roll(mask, shift_ms // 10)].mean()
        best = (value, shift_ms) if best is None or value > best[0] else best
        print("  {0:+5d} ms: {1:8.1f}{2}".format(shift_ms, value, "  <- labeled position" if shift_ms == 0 else ""))
    print("  peak at {0:+d} ms".format(best[1]))
    print("  outside (voicing 0): {0:8.1f}".format(rms[voicing == 0].mean() if (voicing == 0).any() else float("nan")))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("scores")
    p.add_argument("--min-quality", type=float, default=0.5)
    p.add_argument("--min-timing-conf", type=float, default=0.5)
    p.add_argument("--max-drift", type=float, default=20.0, help="ms per minute")
    p.add_argument("--shift-min-gain", type=float, default=0.1, help="pitch_score_aligned - pitch_score needed to apply the transposition")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--lyrics-threshold", type=float, default=0.8)
    p.add_argument("--no-audio", action="store_true", help="duplicate search without the audio hash")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.set_defaults(func=plan)
    e = sub.add_parser("extract")
    e.add_argument("--split", default="all", choices=["all", "train", "val"])
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--trim", type=float, default=20.0, help="ms not labeled at the start and the end of a note")
    e.add_argument("--unvoiced-gap", type=float, default=0.3, help="s away from every note for a voicing target 0")
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    e.add_argument("--force", action="store_true")
    e.set_defaults(func=extract)
    s = sub.add_parser("show")
    s.add_argument("id")
    s.set_defaults(func=show)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
