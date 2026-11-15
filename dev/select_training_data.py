#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file          select_training_data.py
@brief         select training songs from the scores of label_quality.py and list possible duplicates
@author        paradigm

    python dev/select_training_data.py scores.csv -o training_data --min-quality 0.4

Writes into the output folder:
    training.txt     note files above the thresholds (one path per line)
    training.csv     the same with scores, group_id and split
    duplicates.csv   groups of songs, which are possibly the same song, with their scores
    train.txt, val.txt, test.txt   only with --val-fraction / --test-fraction

Possible duplicates are found by four signals, each one links songs to a group:
    same_folder   several note files in one folder (versions of one song)
    same_title    same artist and title after normalisation (case, accents, brackets, feat.)
    same_audio    identical audio file (size + hash of the first and last 64 kB)
    same_lyrics   similar lyrics (estimated jaccard similarity of 8 character shingles)
The signals only suggest, which songs to look at. Groups can be bigger than real duplicates,
because links are chained (A~B and B~C put A and C into one group). Only groups with at least
one selected song are listed, unless --all-groups is given. In every group the song with the
best quality_score is marked as suggested_keep. Remove songs from the training set by writing
their paths into a text file and pass it with --exclude (or use --dedupe to keep only the
suggested_keep song of every group).

group_id of training.csv is the duplicate group (or the song itself), the split never separates a
group, so versions of one song don't leak between train, val and test.
"""
import argparse
import csv
import hashlib
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_quality import read_exclude_file  # noqa: E402  (also puts the repo on the path)
from ultrastar_pitch.project_parser import ProjectParser  # noqa: E402

SKETCH_SIZE = 256  # hashes per song for the lyrics similarity
SHINGLE = 8  # characters per shingle
MIN_SHINGLES = 50  # lyrics shorter than this (like "la la la") say nothing about the song
MAX_POSTINGS = 300  # hashes shared by more songs are generic ("lalalala") and ignored
MIN_SHARED = 5  # shared sketch hashes needed to check a pair of songs
CONTEXT_STATUS = ("ok", "skipped_duet")  # songs, which take part in the duplicate search

DUPLICATE_FIELDS = ["group_id", "group_size", "reasons", "path", "status", "in_training", "suggested_keep",
                    "quality_score", "timing_score", "pitch_score", "pitch_score_aligned", "source",
                    "lyric_similarity", "artist", "title"]
TRAINING_FIELDS = ["path", "group_id", "split", "quality_score", "timing_score", "pitch_score",
                   "pitch_score_aligned", "offset_ms", "shift_semitones", "source", "flags"]


# ------------------------------------------------------------------------------------ fingerprints
def alnum(text: str) -> str:
    """ lower case letters and digits only, accents removed """
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if char.isalnum() and not unicodedata.combining(char))


def song_key(artist: str, title: str) -> str:
    """ artist and title without brackets ([VIDEO], (feat. x), (cover)), "feat." parts, case, accents """
    clean = lambda text: alnum(re.split(r"\b(?:feat|ft|featuring)\b", re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text), flags=re.I)[0])
    artist_key, title_key = clean(artist), clean(title)
    return artist_key + "|" + title_key if artist_key and len(title_key) >= 2 else ""


def lyric_sketch(text: str) -> Optional[np.ndarray]:
    """ the smallest hashes of all shingles of the lyrics (bottom-k sketch), None for tiny lyrics """
    text = alnum(text)
    if len(text) < SHINGLE + MIN_SHINGLES:
        return None
    chars = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    hashes = np.zeros(len(chars) - SHINGLE + 1, dtype=np.uint64)
    for offset in range(SHINGLE):
        hashes = hashes * np.uint64(1000003) + chars[offset: len(chars) - SHINGLE + 1 + offset]
    hashes ^= hashes >> np.uint64(33)  # spread the bits
    hashes *= np.uint64(0xFF51AFD7ED558CCD)
    hashes ^= hashes >> np.uint64(33)
    hashes = np.unique(hashes)
    return hashes[:SKETCH_SIZE] if len(hashes) >= MIN_SHINGLES else None


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """ estimated jaccard similarity of two sketches """
    union = np.union1d(a, b)[:SKETCH_SIZE]
    return float(np.mean(np.isin(union, a) & np.isin(union, b)))


def audio_signature(path: str) -> Optional[str]:
    """ size and hash of the first and last 64 kB of a file """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as file:
            head = file.read(65536)
            file.seek(max(0, size - 65536))
            tail = file.read(65536)
    except OSError:
        return None
    return "{0}:{1}".format(size, hashlib.sha1(head + tail).hexdigest())


def fingerprint(job: Tuple[str, bool]) -> dict:
    """ everything needed to compare a song with others, never raises """
    path, with_audio = job
    result = {"path": path, "artist": "", "title": "", "key": "", "sketch": None, "audio": None}
    try:
        parser = ProjectParser()
        parser.load_note_file(path)
        result["artist"], result["title"] = str(parser.meta.get("#ARTIST", "")), str(parser.meta.get("#TITLE", ""))
        result["key"] = song_key(result["artist"], result["title"])
        # the text of every note: everything behind the pitch
        text = "".join(parser.file_buffer[note["line"]][note["span"][1]:] for note in parser.singable)
        result["sketch"] = lyric_sketch(text)
        if with_audio:
            result["audio"] = audio_signature(parser.find_audio_file()[1])
    except Exception:  # unreadable songs simply have no fingerprint
        pass
    return result


# -------------------------------------------------------------------------------------- grouping
def similar_pairs(sketches: Dict[int, np.ndarray], threshold: float) -> Dict[Tuple[int, int], float]:
    """ pairs of songs with a lyrics similarity >= threshold (candidates come from shared hashes) """
    if not sketches:
        return {}
    ids = np.concatenate([np.full(len(s), i, dtype=np.int64) for i, s in sketches.items()])
    values = np.concatenate(list(sketches.values()))
    order = np.argsort(values, kind="stable")
    values, ids = values[order], ids[order]
    starts = np.flatnonzero(np.concatenate([[True], values[1:] != values[:-1]]))
    ends = np.concatenate([starts[1:], [len(values)]])
    shared = Counter()
    for begin, end in zip(starts, ends):
        if 2 <= end - begin <= MAX_POSTINGS:
            shared.update(combinations(sorted(ids[begin:end].tolist()), 2))
    return {pair: sim for pair, count in shared.items() if count >= MIN_SHARED
            for sim in [similarity(sketches[pair[0]], sketches[pair[1]])] if sim >= threshold}


class Groups:
    """ union find with a record of the signals, which linked the songs """

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.edges = []  # type: List[Tuple[int, int, str]]

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def link(self, i: int, j: int, reason: str):
        self.edges.append((i, j, reason))
        self.parent[self.find(i)] = self.find(j)


def find_groups(rows: List[dict], prints: List[dict], lyrics_threshold: float) -> Tuple[Groups, Dict[int, float]]:
    """ link all songs, which look like the same song, returns the groups and the best lyrics
        similarity of every song """
    groups = Groups(len(rows))
    for reason, keys in (("same_folder", [os.path.dirname(r["path"]) for r in rows]),
                         ("same_title", [p["key"] for p in prints]),
                         ("same_audio", [p["audio"] or "" for p in prints])):
        members = defaultdict(list)
        for i, key in enumerate(keys):
            if key:
                members[key].append(i)
        for indices in members.values():
            for other in indices[1:]:
                groups.link(indices[0], other, reason)
    best = {}  # type: Dict[int, float]
    sketches = {i: p["sketch"] for i, p in enumerate(prints) if p["sketch"] is not None}
    for (i, j), sim in similar_pairs(sketches, lyrics_threshold).items():
        groups.link(i, j, "same_lyrics")
        best[i], best[j] = max(best.get(i, 0.0), sim), max(best.get(j, 0.0), sim)
    return groups, best


# ----------------------------------------------------------------------------------------- main
def number(row: dict, key: str) -> Optional[float]:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def split_of(group_key: str, val: float, test: float) -> str:
    """ stable split of a group, based on a hash of its name """
    position = int(hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:8], 16) / 16.0 ** 8
    return "test" if position < test else "val" if position < test + val else "train"


def main():
    parser = argparse.ArgumentParser(description="select training songs and list possible duplicates")
    parser.add_argument("scores", help="csv file of label_quality.py")
    parser.add_argument("-o", "--output", default="training_data", help="output folder")
    parser.add_argument("--min-quality", type=float, default=0.4, help="threshold for quality_score (default: %(default)s)")
    parser.add_argument("--min-timing", type=float, default=0.0, help="threshold for timing_score")
    parser.add_argument("--min-pitch", type=float, default=0.0, help="threshold for the pitch score")
    parser.add_argument("--pitch-column", default="pitch_score", choices=["pitch_score", "pitch_score_aligned"],
                        help="pitch_score_aligned also accepts transposed songs")
    parser.add_argument("--drop-flags", default="", help="comma separated flags of the scores csv, which exclude a song")
    parser.add_argument("--exclude", help="text file with paths to remove from the training set (one per line)")
    parser.add_argument("--dedupe", action="store_true", help="keep only the suggested_keep song of every group")
    parser.add_argument("--lyrics-threshold", type=float, default=0.8, help="similarity of lyrics, which links songs")
    parser.add_argument("--no-audio", action="store_true", help="don't compare the audio files (saves reading them)")
    parser.add_argument("--all-groups", action="store_true", help="list groups without a selected song too")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if args.val_fraction < 0 or args.test_fraction < 0 or args.val_fraction + args.test_fraction >= 1:
        parser.error("val and test fraction have to be >= 0 and sum up to less than 1")

    with open(args.scores, newline="", encoding="utf-8") as file:
        all_rows = list(csv.DictReader(file))
    rows = [r for r in all_rows if r.get("status") in CONTEXT_STATUS]
    print("{0} rows, {1} usable ({2})".format(len(all_rows), len(rows), ", ".join(
        "{0}: {1}".format(k, v) for k, v in Counter(r.get("status", "") for r in all_rows).most_common(6))), flush=True)

    # --- selection by thresholds
    drop_flags = {flag.strip() for flag in args.drop_flags.split(",") if flag.strip()}
    removed = read_exclude_file(args.exclude, os.getcwd()) if args.exclude else set()
    quality = [number(r, "quality_score") for r in rows if r["status"] == "ok"]
    quality = [q for q in quality if q is not None]
    if quality:
        print("quality_score of the rated songs: " + ", ".join(
            "{0}% {1:.2f}".format(p, np.percentile(quality, p)) for p in (10, 25, 50, 75, 90)))
        print("songs with quality_score >= " + ", ".join(
            "{0:.1f}: {1}".format(t, sum(q >= t for q in quality)) for t in np.arange(0.1, 1.0, 0.1)), flush=True)
    selected = set()
    for i, r in enumerate(rows):
        if r["status"] != "ok":
            continue
        q, t, p = number(r, "quality_score"), number(r, "timing_score"), number(r, args.pitch_column)
        if None in (q, t, p) or q < args.min_quality or t < args.min_timing or p < args.min_pitch:
            continue
        if drop_flags & set(r.get("flags", "").split(";")):
            continue
        if os.path.normcase(os.path.abspath(r["path"])) in removed:
            continue
        selected.add(i)

    # --- duplicates
    jobs = [(r["path"], not args.no_audio) for r in rows]
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        prints = list(pool.map(fingerprint, jobs, chunksize=32))
    groups, lyric_best = find_groups(rows, prints, args.lyrics_threshold)
    members = defaultdict(list)
    for i in range(len(rows)):
        members[groups.find(i)].append(i)
    reasons = defaultdict(set)
    for i, j, reason in groups.edges:
        reasons[groups.find(i)].add(reason)

    def quality_key(i: int):
        return (-(number(rows[i], "quality_score") or -1.0), -(number(rows[i], "timing_score") or -1.0), rows[i]["path"])

    keep = {}  # best selected song of every group
    for root, indices in members.items():
        chosen = sorted((i for i in indices if i in selected), key=quality_key)
        if chosen:
            keep[root] = chosen[0]
    if args.dedupe:
        selected = {i for i in selected if keep[groups.find(i)] == i}

    # --- ids: stable, based on the sorted first path of every group
    ordered = sorted(members.values(), key=lambda indices: min(rows[i]["path"] for i in indices))
    group_id = {}
    for number_, indices in enumerate(ordered, 1):
        for i in indices:
            group_id[i] = "g{0:05d}".format(number_)
    group_name = {i: min(rows[k]["path"] for k in members[groups.find(i)]) for i in range(len(rows))}

    os.makedirs(args.output, exist_ok=True)
    chosen_rows = sorted(selected, key=lambda i: rows[i]["path"])
    splits = {i: split_of(group_name[i], args.val_fraction, args.test_fraction) for i in chosen_rows}
    with open(os.path.join(args.output, "training.txt"), "w", encoding="utf-8", newline="\n") as file:
        file.writelines(rows[i]["path"] + "\n" for i in chosen_rows)
    with open(os.path.join(args.output, "training.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, TRAINING_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for i in chosen_rows:
            writer.writerow(dict(rows[i], group_id=group_id[i], split=splits[i]))
    if args.val_fraction or args.test_fraction:
        for name in ("train", "val", "test"):
            with open(os.path.join(args.output, name + ".txt"), "w", encoding="utf-8", newline="\n") as file:
                file.writelines(rows[i]["path"] + "\n" for i in chosen_rows if splits[i] == name)

    listed = 0
    with open(os.path.join(args.output, "duplicates.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, DUPLICATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for indices in ordered:
            root = groups.find(indices[0])
            if len(indices) < 2 or not (args.all_groups or any(i in selected for i in indices)):
                continue
            listed += 1
            for i in sorted(indices, key=quality_key):
                writer.writerow(dict(
                    rows[i], group_id=group_id[i], group_size=len(indices), reasons=",".join(sorted(reasons[root])),
                    in_training="yes" if i in selected else "", suggested_keep="yes" if keep.get(root) == i else "",
                    lyric_similarity=round(lyric_best[i], 2) if i in lyric_best else "",
                    artist=prints[i]["artist"], title=prints[i]["title"]))
    n_groups = len(ordered)
    n_dup_songs = sum(len(g) for g in ordered if len(g) > 1)
    print("{0} songs selected in {1} groups -> {2}".format(len(selected), len({group_id[i] for i in selected}), args.output))
    print("{0} of {1} songs are in groups with more than one song, {2} groups listed in duplicates.csv".format(
        n_dup_songs, len(rows), listed))
    print("reasons: " + ", ".join("{0} {1}".format(r, c) for r, c in Counter(
        reason for root in members for reason in reasons[root] if len(members[root]) > 1).most_common()) + " (groups)")
    if n_groups and args.val_fraction + args.test_fraction:
        print("split: " + ", ".join("{0} {1}".format(k, v) for k, v in sorted(Counter(splits.values()).items())))


if __name__ == "__main__":
    main()
