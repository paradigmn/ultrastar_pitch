#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          note_hygiene_eval.py
@brief         measure the checks of ultrastar_pitch/note_hygiene.py on a song database and verify its fixes
@author        paradigm

Only the note files are read (no audio). For every note file:
  - which checks fire (share of the files per group of folders, line breaks per checked line break);
  - how much every fix changes on its own (changed lines);
  - verification: without new pitches the tool writes the file unchanged; with keep_octave an
    unchanged pitch class leaves the note untouched and a new one lands within 6 semitones of the
    old note; after all fixes no fixable issue is left, a second run changes nothing, and the fixed
    file is read again with the same notes, pitches and start times (end times only change where
    an overlapping note was shortened).

    python dev/note_hygiene_eval.py songs --group "A+B=folderA,folderB" --group C=folderC \\
        --workers 8 | tee dev/results/note_hygiene_eval.txt
"""

import argparse
import collections
import logging
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultrastar_pitch import note_hygiene  # noqa: E402
from ultrastar_pitch.project_parser import ProjectParser, ProjectParserError  # noqa: E402

# the parser warns about duplicate headers etc. in thousands of files
logging.getLogger("ultrastar_pitch").setLevel(logging.ERROR)


def checked_breaks(lines, body_start):
    """ line breaks whose position the style check looks at (pause of 2 beats or more, inside the pause) """
    blocks, _, _ = note_hygiene._scan(lines, body_start)
    count = 0
    for block in blocks:
        prev, brk = None, None
        for el in block.elements:
            if el.kind == "break":
                if brk is None and prev is not None:
                    brk = el
                continue
            if prev is not None and brk is not None and brk.beat is not None:
                if el.start - prev.end >= 2 and prev.end <= brk.beat <= el.start:
                    count += 1
            prev, brk = el, None
    return count


def analyse(path: str) -> dict:
    res = {"path": path, "fail": []}
    parser = ProjectParser()
    try:
        parser.load_note_file(path)
    except ProjectParserError as err:
        res["skip"] = type(err).__name__
        return res
    lines0 = parser.render_lines()
    body, ms = parser.body_start, parser.ms_per_beat
    utf8 = parser.encoding == "utf-8"
    if utf8 and lines0 != parser.file_buffer:
        res["fail"].append("render without new pitches changes the file")
    issues = note_hygiene.check_song(parser, style=True)
    res["kinds"] = collections.Counter(i.kind for i in issues)
    res["breaks_checked"] = checked_breaks(parser.file_buffer, body)
    # every fix on its own: changed lines
    res["changed"] = {}
    for fix in note_hygiene.FIXES:
        fixed, done = note_hygiene.fix_lines(lines0, body, ms, (fix,))
        res["changed"][fix] = sum(1 for a, b in zip(fixed, lines0) if a != b) + abs(len(fixed) - len(lines0))
    # all fixes: nothing fixable left, idempotent, readable with the same notes
    fixed, done = note_hygiene.fix_lines(lines0, body, ms, note_hygiene.FIXES)
    res["done"] = done
    left = [i for i in note_hygiene.check_lines(fixed, body, ms, style=True) if i.fix is not None]
    if left:
        res["fail"].append("fixable issues left: " + ", ".join(sorted({i.kind for i in left})))
        res["left"] = collections.Counter(i.kind for i in left)
    again, done2 = note_hygiene.fix_lines(fixed, body, ms, note_hygiene.FIXES)
    if again != fixed or done2:
        res["fail"].append("second fix run changes: {0}".format(done2))
    with tempfile.TemporaryDirectory(prefix="hygiene_") as tmp:
        out = os.path.join(tmp, "notes.txt")
        with open(out, "w", encoding="utf-8", newline="\n") as file:
            file.writelines(fixed)
        check = ProjectParser()
        try:
            check.load_note_file(out)
        except ProjectParserError as err:
            res["fail"].append("fixed file not readable: {0}".format(err))
            check = None
    if check is not None:
        a, b = parser.singable, check.singable
        if len(a) != len(b):
            res["fail"].append("number of notes {0} -> {1}".format(len(a), len(b)))
        else:
            if any(x["pitch_file"] != y["pitch_file"] for x, y in zip(a, b)):
                res["fail"].append("pitches changed")
            if any(abs(x["t_start"] - y["t_start"]) > 1e-6 for x, y in zip(a, b)):
                res["fail"].append("start times changed")
            shortened = sum(1 for x, y in zip(a, b) if y["t_end"] < x["t_end"] - 1e-6)
            longer = sum(1 for x, y in zip(a, b) if y["t_end"] > x["t_end"] + 1e-6)
            if longer or shortened > done.get("overlapping notes shortened", 0):
                res["fail"].append("end times changed beyond the shortened notes")
            if any(x["text"].strip() != y["text"].strip() for x, y in zip(a, b)):
                res["fail"].append("syllables changed")
    # octave: unchanged pitch classes leave the file untouched, new ones within 6 semitones
    parser.update_pitches(parser.dump_pitches(), keep_octave=True)
    if utf8 and parser.render_lines() != parser.file_buffer:
        res["fail"].append("keep_octave with the same pitch classes changes the file")
    new = [(p + 5) % 12 for p in parser.dump_pitches()]
    parser.update_pitches(new, keep_octave=True)
    if any(s["pitch_out"] % 12 != n or abs(s["pitch_out"] - s["pitch_file"]) > 6 for s, n in zip(parser.singable, new)):
        res["fail"].append("keep_octave out of range")
    res["outside_octave"] = sum(1 for s in parser.singable if not 0 <= s["pitch_file"] <= 11)
    res["notes"] = len(parser.singable)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="song folder, searched recursively for *.txt")
    ap.add_argument("--group", action="append", default=[],
                    help="NAME=folder1,folder2: first level folders reported together (the rest: other)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    root = Path(args.root)
    groups = {}
    for spec in args.group:
        name, _, folders = spec.partition("=")
        for folder in folders.split(","):
            groups[folder.strip()] = name.strip()
    files = sorted(str(p) for p in root.glob("**/*.txt") if p.name.lower() != "index.txt")
    with multiprocessing.Pool(args.workers) as pool:
        results = pool.map(analyse, files, chunksize=32)

    order = [name for name in dict.fromkeys(groups.values())] + ["other"]
    n = collections.Counter()
    skipped = collections.defaultdict(collections.Counter)
    has = collections.defaultdict(collections.Counter)
    breaks = collections.Counter()
    break_issues = collections.Counter()
    changed = collections.defaultdict(collections.Counter)
    done_total = collections.defaultdict(collections.Counter)
    fails = collections.Counter()
    examples = collections.defaultdict(list)
    notes = outside = 0
    for res in results:
        group = groups.get(Path(res["path"]).relative_to(root).parts[0], "other")
        if "skip" in res:
            skipped[group][res["skip"]] += 1
            continue
        n[group] += 1
        for kind in res["kinds"]:
            has[group][kind] += 1
        breaks[group] += res["breaks_checked"]
        break_issues[group] += res["kinds"].get("break_position", 0)
        for fix, count in res["changed"].items():
            changed[group][fix] += count > 0
        for what, count in res["done"].items():
            done_total[group][what] += count
        for fail in res["fail"]:
            key = fail.split(":")[0]
            fails[key] += 1
            if len(examples[key]) < 3:
                examples[key].append("{0}: {1}".format(res["path"], fail))
        notes += res["notes"]
        outside += res["outside_octave"]

    def row(label, values, fmt="{0:12.1f}%"):
        print("{0:58s}".format(label[:58]) + "".join(fmt.format(v) for v in values))

    print("note files: " + ", ".join("{0} {1}".format(g, n[g]) for g in order)
          + " | not readable: " + ", ".join("{0} {1}".format(g, dict(skipped[g])) for g in order if skipped[g]))
    print("{0:58s}".format("share of the files with ...") + "".join("{0:>13s}".format(g) for g in order))
    for kind, (severity, text) in note_hygiene.KINDS.items():
        row("[{0}] {1}".format(severity, kind), [100.0 * has[g][kind] / max(n[g], 1) for g in order])
    row("line breaks not at barbeque's position (per checked one)",
        [100.0 * break_issues[g] / max(breaks[g], 1) for g in order])
    print()
    print("{0:58s}".format("files changed by the fix alone") + "".join("{0:>13s}".format(g) for g in order))
    for fix in note_hygiene.FIXES:
        row(fix, [100.0 * changed[g][fix] / max(n[g], 1) for g in order])
    print()
    print("{0:58s}".format("all fixes: changes per file") + "".join("{0:>13s}".format(g) for g in order))
    whats = sorted({w for g in done_total for w in done_total[g]})
    for what in whats:
        row(what, [done_total[g][what] / max(n[g], 1) for g in order], "{0:13.2f}")
    print()
    print("notes outside 0 .. 11 (written as 0 .. 11 before keep_octave): {0} of {1} = {2:.1f} %".format(
        outside, notes, 100.0 * outside / max(notes, 1)))
    total = sum(n.values())
    print("verification on {0} files: {1}".format(total, "ok" if not fails else "FAILED"))
    for key, count in fails.most_common():
        print("  {0}: {1} files".format(key, count))
        for example in examples[key]:
            print("    " + example)


if __name__ == "__main__":
    main()
