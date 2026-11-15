#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          remaining_errors_val.py
@brief         what is left in the errors of the default pipeline: information in some model, or label errors?
@author        paradigm

On the 249 validation songs (cached scores, no model runs except the small note context network):

1. accuracy of the single models (2020 pitchnet, RMVPE, pitchnet_2026_11_13 with the tuning correction) on the
   mix and on the vocals, and on the notes that both the 2020 model and RMVPE miss (the "17 %" of the report:
   does a third model find them?);
2. every note the packaged default pipeline (mix + vocals, four GRU average) gets wrong, sorted by the six single
   votes (three models x mix / vocals): some vote hits the label (information is there), all six agree on
   another class (label error or an error all models share), a majority of at least four on another class, or no
   agreement;
3. the same by note length, the intervals between label and the unanimous vote, and a list of the strongest
   label error candidates to listen to (dev/results/remaining_errors_listen.csv: song, time in the audio,
   syllable, label, what all models hear).

Reference per note: the label rotated by the song transposition (field `target` of the prepared arrays of
context_gru_2026.py prepare), so all models are compared on the same pitch classes.

    CUDA_VISIBLE_DEVICES= .venv/bin/python dev/remaining_errors_val.py > dev/results/remaining_errors_val.txt
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import pipeline_weights_val as pw  # noqa: E402
from ultrastar_pitch import ProjectParser  # noqa: E402
from ultrastar_pitch import context_model as cm  # noqa: E402

RESULTS = ROOT / "dev" / "results"
FCPE_CACHE = RESULTS / "fcpe_cache"
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LEN_EDGES = [0.06, 0.1, 0.2, 0.4, 0.8]
LEN_LABELS = ["<60", "60-100", "100-200", "200-400", "400-800", ">800"]


def split(z):
    order = np.argsort(z["song"], kind="stable")
    bounds = np.searchsorted(z["song"][order], np.arange(len(z["names"]) + 1))
    return [order[bounds[s]:bounds[s + 1]] for s in range(len(z["names"]))]


def frame_scores(z, source):
    """ pitchnet_2026_11_13 scores as the pipeline uses them: tuning-corrected when the estimate is reliable """
    cents, reliable = z["tuning_" + source]
    return z["ch64_tuned_" + source] if reliable else z["ch64_" + source]


def pct(a, b):
    return 100.0 * a / max(b, 1)


def main() -> int:
    net = cm.ContextModel()
    arrays = {v: dict(np.load(RESULTS / "gru2026_val_ch64t{0}.npz".format("_vocals" if v else ""))) for v in (False, True)}
    names = list(arrays[False]["names"])
    assert names == list(arrays[True]["names"])
    rows = {r["id"]: r for r in pw.val_songs()}
    parser = ProjectParser()

    per_note = []      # one dict per note
    for s, idx in enumerate(split(arrays[True])):
        sid = names[s]
        z = dict(np.load(pw.CACHE / (sid + ".npz")))
        votes = {"2020 " + src: z["old_" + src].argmax(1) for src in ("mix", "vocals")}
        votes.update({"RMVPE " + src: z["rmvpe_" + src].argmax(1) for src in ("mix", "vocals")})
        votes.update({"2026 " + src: frame_scores(z, src).argmax(1) for src in ("mix", "vocals")})
        fz = FCPE_CACHE / (sid + ".npz")
        if fz.exists():         # FCPE (dev/fcpe_val.py): an independent model, reported apart from the six votes
            f = dict(np.load(fz))
            votes.update({"FCPE " + src: f["fcpe_" + src].argmax(1) for src in ("mix", "vocals") if "fcpe_" + src in f})
        final = {}
        for vocals in (False, True):
            a = arrays[vocals]
            i = split(a)[s]
            feed = {k: a[k][i][None].astype(np.float32) for k in ("circ", "other", "base")}
            out = net.sess.run(None, feed)[0][0]
            final[vocals] = (out.argmax(1), out)
        target = arrays[True]["target"][idx]
        assert (target == arrays[False]["target"][split(arrays[False])[s]]).all()
        dur = arrays[True]["dur"][idx]
        # note times in the audio (timing correction applied) and the syllables, for the listening list
        parser.load_note_file(rows[sid]["path"])
        segments = parser.note_segments()
        assert len(segments) == len(target), sid
        applied = float(z["applied_s"])
        texts = z["ctx_texts"]
        for n in range(len(target)):
            logp = final[True][1][n]
            top2 = np.sort(logp)[-2:]
            per_note.append({"song": sid, "path": rows[sid]["path"], "note": n, "start": segments[n][0] + applied,
                             "dur": float(dur[n]), "text": str(texts[n]).strip(), "ref": int(z["ref"][n]),
                             "target": int(target[n]), "mix": int(final[False][0][n]), "voc": int(final[True][0][n]),
                             "margin": float(top2[1] - top2[0]),
                             **{k: int(v[n]) for k, v in votes.items()}})

    keys = [k for k in per_note[0] if k.startswith(("2020", "RMVPE", "2026"))]
    fkeys = [k for k in ("FCPE mix", "FCPE vocals") if all(k in p for p in per_note)]
    tgt = np.array([p["target"] for p in per_note])
    vote = {k: np.array([p[k] for p in per_note]) for k in keys + fkeys}
    dur = np.array([p["dur"] for p in per_note])
    mix, voc = np.array([p["mix"] for p in per_note]), np.array([p["voc"] for p in per_note])
    n = len(tgt)
    print("249 validation songs, {0} notes; reference = label rotated by the song transposition".format(n))
    print()
    print("1. single models and the pipeline, exact %")
    for k in keys + fkeys:
        print("   {0:14s} {1:5.1f}".format(k, pct((vote[k] == tgt).sum(), n)))
    print("   {0:14s} {1:5.1f}".format("pipeline mix", pct((mix == tgt).sum(), n)))
    print("   {0:14s} {1:5.1f}".format("pipeline m+v", pct((voc == tgt).sum(), n)))
    print()
    both = (vote["2020 mix"] != tgt) & (vote["RMVPE mix"] != tgt)
    print("2. notes that both the 2020 model and RMVPE miss on the mix: {0} = {1:.1f} %".format(both.sum(), pct(both.sum(), n)))
    for k in ["2026 mix", "2026 vocals", "2020 vocals", "RMVPE vocals"] + fkeys:
        print("   {0:14s} right on {1:5.1f} % of them".format(k, pct(((vote[k] == tgt) & both).sum(), both.sum())))
    print("   {0:14s} right on {1:5.1f} % of them".format("pipeline m+v", pct(((voc == tgt) & both).sum(), both.sum())))
    print()

    wrong = voc != tgt
    any_hit = np.zeros(n, bool)
    for k in keys:
        any_hit |= vote[k] == tgt
    stack = np.stack([vote[k] for k in keys], axis=1)
    counts = np.stack([(stack == c).sum(1) for c in range(12)], axis=1)
    top = counts.argmax(1)
    top_n = counts.max(1)
    unanimous = wrong & ~any_hit & (top_n == len(keys))
    majority = wrong & ~any_hit & (top_n >= 4) & ~unanimous
    unclear = wrong & ~any_hit & ~unanimous & ~majority
    hit = wrong & any_hit
    print("3. notes the default pipeline (mix + vocals) gets wrong: {0} = {1:.1f} % of all notes".format(wrong.sum(), pct(wrong.sum(), n)))
    cats = [("some single vote hits the label", hit), ("all six votes agree on another class", unanimous),
            ("at least four of six agree on another class", majority), ("no agreement, no vote right", unclear)]
    for label, m in cats:
        print("   {0:46s} {1:5d}  {2:5.1f} % of the errors, {3:4.1f} % of all notes".format(
            label, m.sum(), pct(m.sum(), wrong.sum()), pct(m.sum(), n)))
    print("   of the 'hit' notes, the pipeline's class agrees with the vote majority: {0:.1f} %".format(
        pct(((voc == top) & hit).sum(), hit.sum())))
    # control: the label mirrored at the pipeline's class (same distance, other side); if the votes hit the real
    # label clearly more often than this pseudo label, the 'hit' notes carry information beyond chance
    mirror = (2 * voc - tgt) % 12
    ok = wrong & (mirror != tgt)               # a tritone error mirrors onto itself
    hit_real = np.zeros(n, bool)
    hit_mirror = np.zeros(n, bool)
    for k in keys:
        hit_real |= vote[k] == tgt
        hit_mirror |= vote[k] == mirror
    print("   control on {0} errors: some vote hits the label {1:.1f} %, the mirrored pseudo label {2:.1f} %".format(
        ok.sum(), pct((hit_real & ok).sum(), ok.sum()), pct((hit_mirror & ok).sum(), ok.sum())))
    print("   which single vote hits the label on the pipeline's errors (share of the errors):")
    for k in keys:
        print("     {0:14s} {1:5.1f} %   (control: mirrored {2:4.1f} %)".format(
            k, pct(((vote[k] == tgt) & ok).sum(), ok.sum()), pct(((vote[k] == mirror) & ok).sum(), ok.sum())))
    print()
    bins = np.digitize(dur, LEN_EDGES)
    print("4. by note length: share of the notes wrong, and of those the categories above (%)")
    print("   {0:9s} {1:>7s} {2:>7s} | {3:>6s} {4:>9s} {5:>8s} {6:>8s}".format("length", "notes", "wrong", "hit", "unanimous", "majority", "unclear"))
    for b, lab in enumerate(LEN_LABELS):
        m = bins == b
        w = wrong & m
        print("   {0:9s} {1:7d} {2:6.1f}% | {3:6.1f} {4:9.1f} {5:8.1f} {6:8.1f}".format(
            lab, m.sum(), pct(w.sum(), m.sum()), *[pct((c & m).sum(), w.sum()) for _, c in cats]))
    print()
    if fkeys:
        print("4b. the independent FCPE votes on the pipeline's errors, per category (%):")
        print("    hits label = FCPE hears the label; follows top = FCPE hears the class most of the six votes hear;")
        print("    control = FCPE hears the label mirrored at the pipeline's class (chance level for 'hits label')")
        mirror = (2 * voc - tgt) % 12
        for k in fkeys:
            print("    {0}".format(k))
            for label, m in cats + [("all errors", wrong)]:
                print("      {0:46s} hits label {1:5.1f} | follows top {2:5.1f} | control {3:4.1f}".format(
                    label, pct(((vote[k] == tgt) & m).sum(), m.sum()), pct(((vote[k] == top) & m).sum(), m.sum()),
                    pct(((vote[k] == mirror) & m & (mirror != tgt)).sum(), (m & (mirror != tgt)).sum())))
        if "FCPE vocals" in fkeys:
            fv = vote["FCPE vocals"]
            long_u = unanimous & (dur >= 0.3)
            print("    unanimous notes of at least 300 ms ({0}): FCPE vocals hears the label {1:.1f} %, the six votes' class {2:.1f} %, other {3:.1f} %".format(
                long_u.sum(), pct(((fv == tgt) & long_u).sum(), long_u.sum()), pct(((fv == top) & long_u).sum(), long_u.sum()),
                pct(((fv != tgt) & (fv != top) & long_u).sum(), long_u.sum())))
        print()
    diff = (top - tgt) % 12
    signed = np.where(diff > 6, diff - 12, diff)
    print("5. interval label -> unanimous vote (semitones, + = models hear higher), share of the unanimous notes")
    for d in range(-5, 7):
        c = (signed[unanimous] == d).sum()
        if c:
            print("   {0:+3d}: {1:5.1f} %  ({2})".format(d, pct(c, unanimous.sum()), c))
    print()
    # strongest label error candidates: unanimous, long, confident
    cand = [i for i in np.flatnonzero(unanimous) if dur[i] >= 0.3]
    cand.sort(key=lambda i: -per_note[i]["margin"])
    out = RESULTS / "remaining_errors_listen.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "song", "note", "audio_time_s", "duration_s", "syllable", "label_class",
                    "models_hear", "fcpe_hears", "interval", "pipeline_margin", "path"])
        for r, i in enumerate(cand, 1):
            p = per_note[i]
            shift = (p["target"] - p["ref"] % 12) % 12      # label class -> audio class
            heard = (top[i] - shift) % 12                    # unanimous vote in the label's key
            mm, ss = divmod(p["start"], 60)
            w.writerow([r, Path(p["path"]).parent.name, p["note"], "{0:.0f}:{1:05.2f}".format(mm, ss),
                        "{0:.2f}".format(p["dur"]), p["text"], NAMES[p["ref"] % 12], NAMES[heard],
                        NAMES[(p["FCPE vocals"] - shift) % 12] if "FCPE vocals" in p else "",
                        "{0:+d}".format(int(signed[i])), "{0:.2f}".format(p["margin"]), p["path"]])
    print("6. {0} unanimous notes of at least 300 ms -> {1} (strongest first)".format(len(cand), out))
    # short list: the strongest note of each song, the 30 strongest songs
    with open(out, newline="", encoding="utf-8") as f:
        listed = list(csv.DictReader(f))
    per_song = {}
    for r in listed:
        per_song.setdefault(r["song"], []).append(r)
    short = sorted(({**rs[0], "notes_in_song": len(rs)} for rs in per_song.values()),
                   key=lambda r: -float(r["pipeline_margin"]))[:30]
    out30 = RESULTS / "remaining_errors_listen_top30.csv"
    with open(out30, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(short[0]))
        w.writeheader()
        for i, r in enumerate(short, 1):
            w.writerow({**r, "rank": i})
    print("   {0} songs; strongest note per song, 30 songs -> {1}".format(len(per_song), out30))
    return 0


if __name__ == "__main__":
    sys.exit(main())
