#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          context_gru_2026.py
@brief         the note context GRU on top of the pipeline with the frame model pitchnet_2026
@author        paradigm

A first study (not in this repository) measured a small bidirectional GRU over the notes of a song on top of the
old pitchnet (+0.8 points on 197 songs, mix, timing of the file). This script repeats it for the current default
pipeline:
the scores of the frame model and RMVPE with the timing correction and the weights of
DetectionPipeline.FRAME_MODEL_WEIGHTS, repetition pooling, soft key prior. The GRU adds a correction to exactly
what the pipeline decides (the "base"), so it starts at the pipeline result.

Data: caches of dev/pipeline_weights_val.py extract (same format): training songs of pitchnet_2026
(dev/results/context_cache/train), the validation songs (dev/results/pipeline_weights_cache, for choosing the
epoch and the variants) and the 197 evaluation songs (dev/results/context_cache/test, only at the end).

The packaged network (ultrastar_pitch/binaries/context_gru_ensemble_2026_11_13.onnx, run by
ultrastar_pitch/context_model.py, which also computes the note features for prepare) is the average of four GRUs
(combine, see dev/README.md); the first of them (context_gru_2026_11_13) was made like this, the others the same
way with --seed 1 and on the merged mix and mix + vocals arrays (prepare --vocals, merge):

    # 1. per note arrays (project environment): scores of pitchnet_2026_11_13 (cached as "ch64") with the tuning correction
    .venv/bin/python dev/context_gru_2026.py prepare dev/results/context_cache/train --model ch64 --tuning 0 \\
        --out dev/results/gru2026_train_ch64t.npz
    .venv/bin/python dev/context_gru_2026.py prepare dev/results/pipeline_weights_cache --model ch64 --tuning 0 \\
        --out dev/results/gru2026_val_ch64t.npz
    # 2. training on the GPU (numpy + torch), the epoch chosen on the validation songs; --export writes the onnx
    uv run --no-project --isolated --with numpy --with torch --with onnx python dev/context_gru_2026.py train \\
        dev/results/gru2026_train_ch64t.npz dev/results/gru2026_val_ch64t.npz --hidden 96 --epochs 20 --batch 16 \\
        --device cuda --save dev/results/gru2026_ch64t_gru96.pt
    uv run ... train none dev/results/gru2026_val_ch64t.npz --load dev/results/gru2026_ch64t_gru96.pt \\
        --export dev/results/gru2026_ch64t_gru96.onnx [--test dev/results/gru2026_test_ch64t.npz]
    # 3. one onnx with the mean of the log softmax of several exported networks
    .venv/bin/python dev/context_gru_2026.py combine OUT.onnx A.onnx B.onnx ...

Input per note: circular block (rotated together by the augmentation, 36): E - max(E), the key log prior,
Erep - max(Erep) (E = combined log scores, Erep = after the repetition pooling); other (7): pause bin before the
note (one hot, 5), log note length, phrase start. Output: 12 logits added to base = Erep + k * key prior.
Target: file pitch class rotated by the song transposition (estimated from argmax E), so target and audio agree.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEN_EDGES = [0, 0.06, 0.1, 0.2, 0.4, 0.8, 99]
ARRAYS = ("circ", "other", "base", "target", "ref", "song", "dur", "text")
TEXT_BUCKETS = 1024                          # syllables hashed into this many buckets (0 = no text)


def text_ids(texts) -> np.ndarray:
    """ syllable per note -> bucket 1 .. TEXT_BUCKETS (lower case, without spaces and '~'), 0 for no text """
    import zlib
    out = []
    for t in texts:
        t = str(t).strip().lower().replace("~", "")
        out.append(zlib.crc32(t.encode("utf-8")) % TEXT_BUCKETS + 1 if t else 0)
    return np.array(out, dtype=np.int64)


def _package():
    for _p in (HERE, ROOT):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from ultrastar_pitch import repetition_pooling, StochasticPostprocessor
    return repetition_pooling, StochasticPostprocessor


def frame_scores(z, model, source, tuning=None):
    """ the frame model scores of one source ("mix" / "vocals"), with the tuning correction like song_scores """
    if tuning is not None:
        cents, reliable = z["tuning_" + source]
        if reliable and abs(cents) >= tuning:
            return z[model + "_tuned_" + source]
    return z[model + "_" + source]


def song_scores(z, model, w, v, k, vocals=False, tuning=None):
    """ (E, Erep, key log prior, base) of one cached song like DetectionPipeline.transform, None if scores are
        missing; base.argmax = what the pipeline writes. tuning: the frame model scores with the tuning
        correction when the RMVPE estimate is reliable and at least this many cents off (None: never) """
    repetition_pooling, SP = _package()
    sources = ("mix", "vocals") if vocals else ("mix",)
    if any(model + "_" + s not in z or "rmvpe_" + s not in z for s in sources) or (
            tuning is not None and any(model + "_tuned_" + s not in z for s in sources)):
        return None

    def model_scores(source):
        return frame_scores(z, model, source, tuning)

    E = w * z["rmvpe_mix"] + (1 - w) * model_scores("mix")
    if vocals:
        E = (1 - v) * E + v * (w * z["rmvpe_vocals"] + (1 - w) * model_scores("vocals"))
    context = {key[4:]: z[key].tolist() for key in z if key.startswith("ctx_")}
    Erep = repetition_pooling.pool_repetitions(E, weight=1.0, **context)
    key = SP.detect_key(Erep.argmax(axis=1).tolist())
    keylp = np.log(SP.key_prior(key))[None].repeat(len(E), axis=0)
    base = Erep + k * keylp
    # the pipeline decides exactly this (correct_scores)
    assert (base.argmax(axis=1) == SP.correct_scores(Erep, weight=k)).all()
    return E, Erep, keylp, base


def prepare(args) -> int:
    _package()
    from ultrastar_pitch import tuning_correction as tc
    from ultrastar_pitch import context_model as cm      # the features as the tool computes them
    files = sorted(Path(args.cache).glob("*.npz"))
    sources = ("mix", "vocals") if args.vocals else ("mix",)
    rows = {name: [] for name in ARRAYS}
    names, n_skip = [], 0
    for f in files:
        z = dict(np.load(f))
        res = song_scores(z, args.model, args.w, args.v, args.k, args.vocals, args.tuning)
        if res is None:
            n_skip += 1
            continue
        E, Erep, keylp, base = res
        shift = tc.estimate_shift_semitones(E.argmax(axis=1), z["ref"]).shift_semitones
        circ, other = cm.note_features(E, Erep, keylp, z["durations"], z["ctx_beats"], z["ctx_lengths"], z["ctx_phrases"])
        if args.separate:   # experiment: the single sources as further circular blocks (each - its max per note)
            parts = [frame_scores(z, args.model, src, args.tuning) for src in sources] + [z["rmvpe_" + src] for src in sources]
            circ = np.concatenate([circ] + [(x - x.max(axis=1, keepdims=True)).astype(np.float32) for x in parts], axis=1)
        for name, arr in (("circ", circ), ("other", other), ("base", base), ("target", (z["ref"] - shift) % 12),
                          ("ref", z["ref"]), ("song", np.full(len(E), len(names))), ("dur", z["durations"]),
                          ("text", text_ids(z["ctx_texts"]))):
            rows[name].append(np.asarray(arr))
        names.append(f.stem)
    out = {name: np.concatenate(rows[name]) for name in ARRAYS}
    out["circ"], out["other"], out["base"] = (out[x].astype(np.float32) for x in ("circ", "other", "base"))
    np.savez_compressed(args.out, names=np.array(names), **out)
    print("{0} songs ({1} without the scores skipped), {2} notes -> {3}".format(len(names), n_skip, len(out["ref"]), args.out))
    return 0


def merge(args) -> int:
    """ several prepared files -> one (the songs one after the other, names prefixed by the file number), e.g. the
        mix and the mix + vocals versions of the same songs as one training set """
    parts = [dict(np.load(f)) for f in args.inputs]
    out, names, offset = {k: [] for k in ARRAYS}, [], 0
    for i, z in enumerate(parts):
        for k in ARRAYS:
            if k in z:
                out[k].append(z[k] + offset if k == "song" else z[k])
        names += ["{0}:{1}".format(i, n) for n in z["names"]]
        offset += len(z["names"])
    res = {k: np.concatenate(v) for k, v in out.items() if len(v) == len(parts)}
    np.savez_compressed(args.out, names=np.array(names), **res)
    print("{0} songs, {1} notes -> {2}".format(len(names), len(res["ref"]), args.out))
    return 0


def policy(args) -> int:
    """ the pipeline (no GRU) with the tuning correction of the frame model for several thresholds, paired per
        song against no correction; needs a cache of extract --tuned """
    zs = [dict(np.load(f)) for f in sorted(Path(args.cache).glob("*.npz"))]
    thresholds = [None] + [float(x) for x in args.thresholds.split(",")]
    for vocals in (False, True):
        acc = {}
        for th in thresholds:
            rows = []
            for z in zs:
                res = song_scores(z, args.model, args.w, args.v, args.k, vocals, th)
                if res is None:
                    break
                d = dist(res[3].argmax(axis=1), z["ref"])
                rows.append([np.sum(d == 0), np.sum(d <= 1), len(d)])
            else:
                acc[th] = np.array(rows, dtype=float)
        if None not in acc:
            continue
        a = acc[None]
        cents = np.array([z["tuning_mix"][0] for z in zs])
        print("{0}: {1} songs, {2:.0f} notes; |tuning| >= 10 cents: {3}, >= 15: {4}, >= 25: {5} songs".format(
            "mix + vocals" if vocals else "mix", len(a), a[:, 2].sum(), *(int(np.sum(np.abs(cents) >= c)) for c in (10, 15, 25))))
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(a), (4000, len(a)))
        for th, b in acc.items():
            d = b[idx, 0].sum(1) / b[idx, 2].sum(1) - a[idx, 0].sum(1) / a[idx, 2].sum(1)
            print("  correction {0:<16} exact {1:6.2f} %  within 1 {2:6.2f} %   vs none {3:+.2f} [{4:+.2f} .. {5:+.2f}], "
                  "songs better {6}, worse {7}".format(
                      "none" if th is None else "|off| >= {0:g} c".format(th), 100 * b[:, 0].sum() / b[:, 2].sum(),
                      100 * b[:, 1].sum() / b[:, 2].sum(), 100 * (b[:, 0].sum() / b[:, 2].sum() - a[:, 0].sum() / a[:, 2].sum()),
                      100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5),
                      int(np.sum(b[:, 0] > a[:, 0])), int(np.sum(b[:, 0] < a[:, 0]))))
    return 0


def compare(args) -> int:
    """ pipeline configurations (no GRU) on a cache, paired per song against the first one: frame model NAME,
        NAME@T with the tuning correction when the estimate is reliable and |offset| >= T cents """
    zs = [dict(np.load(f)) for f in sorted(Path(args.cache).glob("*.npz"))]
    configs = []
    for item in args.configs.split(","):
        name, _, th = item.partition("@")
        configs.append((item, name, float(th) if th else None))
    for vocals in (False, True):
        rows = {item: [] for item, _, _ in configs}
        n_skip = 0
        for z in zs:
            res = [song_scores(z, name, args.w, args.v, args.k, vocals, th) for _, name, th in configs]
            if any(r is None for r in res):
                n_skip += 1
                continue
            for (item, _, _), r in zip(configs, res):
                d = dist(r[3].argmax(axis=1), z["ref"])
                rows[item].append([np.sum(d == 0), np.sum(d <= 1), len(d)])
        acc = {item: np.array(r, dtype=float) for item, r in rows.items()}
        a = acc[configs[0][0]]
        if not len(a):
            continue
        print("{0} (w {1}, v {2}, k {3}): {4} songs ({5} without all scores skipped), {6:.0f} notes".format(
            "mix + vocals" if vocals else "mix", args.w, args.v, args.k, len(a), n_skip, a[:, 2].sum()))
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(a), (4000, len(a)))
        for item, b in acc.items():
            d = b[idx, 0].sum(1) / b[idx, 2].sum(1) - a[idx, 0].sum(1) / a[idx, 2].sum(1)
            print("  {0:<14} exact {1:6.2f} %  within 1 {2:6.2f} %   vs {3} {4:+.2f} [{5:+.2f} .. {6:+.2f}], "
                  "songs better {7}, worse {8}".format(
                      item, 100 * b[:, 0].sum() / b[:, 2].sum(), 100 * b[:, 1].sum() / b[:, 2].sum(), configs[0][0],
                      100 * (b[:, 0].sum() / b[:, 2].sum() - a[:, 0].sum() / a[:, 2].sum()),
                      100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5),
                      int(np.sum(b[:, 0] > a[:, 0])), int(np.sum(b[:, 0] < a[:, 0]))))
    return 0


def dist(pred, ref):
    """ distance in semitones after the global transposition per song (most frequent difference) """
    hist = np.bincount((ref - pred) % 12, minlength=12)
    b = int(hist.argmax())
    sh = b - 12 if b > 6 else b
    d = ((pred + sh) % 12 - ref) % 12
    return np.minimum(d, 12 - d)


def train(args) -> int:
    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)

    def load(path):
        z = dict(np.load(path))
        order = np.argsort(z["song"], kind="stable")       # the notes of every song in their original order
        bounds = np.searchsorted(z["song"][order], np.arange(len(z["names"]) + 1))
        songs = []
        for s in range(len(z["names"])):
            idx = order[bounds[s]:bounds[s + 1]]
            songs.append({k: z[k][idx] for k in ARRAYS if k != "song" and k in z})
        return songs

    ck = None
    if args.load:     # a saved network: no training, its configuration replaces --arch / --hidden / --layers / --text
        ck = torch.load(args.load, map_location="cpu")
        args.arch, args.hidden, args.layers, args.text = ck["arch"], ck["hidden"], ck["layers"], ck.get("text", False)
    tr, va = ([] if args.load else load(args.train)), load(args.val)
    te = load(args.test) if args.test else None
    if args.limit_train:
        tr = tr[:args.limit_train]
    n_circ, n_other = va[0]["circ"].shape[1], va[0]["other"].shape[1]

    class TNet(torch.nn.Module):
        """ the transformer alternative: attention over all notes of the song, sinusoidal note index """

        def __init__(self, d, layers, heads=4):
            super().__init__()
            self.d = d
            self.inp = torch.nn.Linear(n_circ + n_other, d)
            self.emb = torch.nn.Embedding(TEXT_BUCKETS + 1, d, padding_idx=0) if args.text else None
            layer = torch.nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.1, batch_first=True)
            self.enc = torch.nn.TransformerEncoder(layer, layers)
            self.out = torch.nn.Linear(d, 12)

        def forward(self, circ, other, base, text=None, lengths=None):
            n = circ.shape[1]
            pos = torch.arange(n, dtype=torch.float32, device=circ.device)[:, None]
            freq = torch.exp(-np.log(1000.0) * torch.arange(0, self.d, 2, dtype=torch.float32, device=circ.device) / self.d)
            pe = torch.zeros(n, self.d, device=circ.device)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
            mask = None if lengths is None else torch.arange(n, device=circ.device)[None] >= lengths.to(circ.device)[:, None]
            x = self.inp(torch.cat([circ, other], dim=-1)) + pe[None]
            if self.emb is not None:
                x = x + self.emb(text)
            h = self.enc(x, src_key_padding_mask=mask)
            return base + self.out(h)

    class Net(torch.nn.Module):
        def __init__(self, hidden, layers):
            super().__init__()
            self.inp = torch.nn.Linear(n_circ + n_other, 64)
            self.emb = torch.nn.Embedding(TEXT_BUCKETS + 1, 64, padding_idx=0) if args.text else None
            self.gru = torch.nn.GRU(64, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                    dropout=0.1 if layers > 1 else 0.0)
            self.out = torch.nn.Linear(2 * hidden, 12)

        def forward(self, circ, other, base, text=None, lengths=None):
            h = self.inp(torch.cat([circ, other], dim=-1))
            h = torch.relu(h if self.emb is None else h + self.emb(text))
            if lengths is None:
                h, _ = self.gru(h)
            else:   # padded batch: the backward direction must start at the last note of each song
                packed = torch.nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
                h, _ = torch.nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=h.shape[1])
            return base + self.out(h)

    def rotate(s, k):
        c = np.concatenate([np.roll(s["circ"][:, 12 * b:12 * b + 12], k, axis=1) for b in range(n_circ // 12)], axis=1)
        return c, np.roll(s["base"], k, axis=1), (s["target"] + k) % 12

    def predict(net, s):
        with torch.no_grad():
            t = [torch.from_numpy(np.ascontiguousarray(s[x]))[None].to(dev) for x in ("circ", "other", "base")]
            text = torch.from_numpy(s["text"])[None].to(dev) if args.text else None
            return net(*t, text)[0].cpu().numpy().argmax(axis=1)

    def batch_tensors(items):
        """ songs (circ, other, base, target, text) padded to the longest one; target -100 = padding """
        n = max(len(it[3]) for it in items)
        pad = lambda a, value: np.pad(a, [(0, n - len(a))] + [(0, 0)] * (a.ndim - 1), constant_values=value)  # noqa: E731
        c, o, b, t, x = (np.stack([pad(it[i], -100 if i == 3 else 0) for it in items]) for i in range(5))
        lengths = torch.tensor([len(it[3]) for it in items])
        return (torch.from_numpy(c).to(dev), torch.from_numpy(o).to(dev), torch.from_numpy(b).to(dev),
                torch.from_numpy(t.astype(np.int64)).to(dev), torch.from_numpy(x.astype(np.int64)).to(dev), lengths)

    def song_acc(preds, songs):
        return np.array([[np.sum(dist(p, s["ref"]) == 0), np.sum(dist(p, s["ref"]) <= 1), len(p)]
                         for p, s in zip(preds, songs)], dtype=float)

    def boot(a, b, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(a), (4000, len(a)))
        d = b[idx, 0].sum(1) / b[idx, 2].sum(1) - a[idx, 0].sum(1) / a[idx, 2].sum(1)
        return 100 * (b[:, 0].sum() / b[:, 2].sum() - a[:, 0].sum() / a[:, 2].sum()), \
            100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)

    def by_length(preds, songs):
        d = np.concatenate([dist(p, s["ref"]) for p, s in zip(preds, songs)])
        dur = np.concatenate([s["dur"] for s in songs])
        b = np.digitize(dur, LEN_EDGES[1:-1])
        return " ".join("{0:5.1f}".format(100 * np.mean(d[b == i] == 0)) for i in range(6))

    base_va = song_acc([s["base"].argmax(axis=1) for s in va], va)
    print("train {0} songs ({1} notes), val {2} songs ({3} notes); hidden {4}, layers {5}, lr {6}".format(
        len(tr), sum(len(s["ref"]) for s in tr), len(va), int(base_va[:, 2].sum()), args.hidden, args.layers, args.lr))
    print("val hand-made base (pipeline): exact {0:.2f} %".format(100 * base_va[:, 0].sum() / base_va[:, 2].sum()), flush=True)
    net = TNet(args.hidden, args.layers) if args.arch == "transformer" else Net(args.hidden, args.layers)
    net = net.to(dev)
    print("{0}{1}: {2:,} parameters".format(args.arch, " + text" if args.text else "", sum(p.numel() for p in net.parameters())))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best = (-1.0, None, 0) if ck is None else (None, ck["state_dict"], ck["epoch"])
    for epoch in range(1, 0 if ck is not None else args.epochs + 1):
        t0 = time.time()
        net.train()
        losses = []
        order = rng.permutation(len(tr))
        for start in range(0, len(order), args.batch):
            items = []
            for i in order[start:start + args.batch]:
                c, b, t = rotate(tr[i], int(rng.integers(12)))
                items.append((c, tr[i]["other"], b, t, tr[i].get("text", np.zeros(len(t), dtype=np.int64))))
            c, o, b, t, x, lengths = batch_tensors(items)
            logits = net(c, o, b, x if args.text else None, lengths if args.batch > 1 else None)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 12), t.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        net.eval()
        acc = song_acc([predict(net, s) for s in va], va)
        exact = acc[:, 0].sum() / acc[:, 2].sum()
        print("epoch {0:2d}: loss {1:.4f} | val exact {2:.2f} %, gru - base {3:+.2f} [{4:+.2f} .. {5:+.2f}] | {6:.0f} s".format(
            epoch, np.mean(losses), 100 * exact, *boot(base_va, acc), time.time() - t0), flush=True)
        if exact > best[0]:
            best = (exact, {k: v.clone() for k, v in net.state_dict().items()}, epoch)
    net.load_state_dict(best[1])
    net.eval()
    print("best epoch {0}".format(best[2]))
    pred_va = [predict(net, s) for s in va]
    print("val (epoch chosen on it):        base {0} | gru {1}   (exact by length <60 .. >800 ms)".format(
        by_length([s["base"].argmax(axis=1) for s in va], va), by_length(pred_va, va)))
    if args.export:
        # inputs per song: circ [1, n, 36], other [1, n, 7], base [1, n, 12] -> corrected scores [1, n, 12]
        assert not args.text and args.arch == "gru", "the export covers the GRU without text"
        cpu_net = (TNet(args.hidden, args.layers) if args.arch == "transformer" else Net(args.hidden, args.layers)).eval()
        cpu_net.load_state_dict({k: v.cpu() for k, v in best[1].items()})
        s0 = va[0]
        dummy = tuple(torch.from_numpy(np.ascontiguousarray(s0[x]))[None] for x in ("circ", "other", "base"))
        torch.onnx.export(cpu_net, dummy, args.export, input_names=["circ", "other", "base"], output_names=["scores"],
                          dynamic_axes={k: {1: "notes"} for k in ("circ", "other", "base", "scores")}, opset_version=17,
                          dynamo=False)
        print("exported", args.export)
    if args.save and ck is None:
        torch.save({"state_dict": best[1], "arch": args.arch, "hidden": args.hidden, "layers": args.layers, "text": args.text,
                    "epoch": best[2]}, args.save)
    for label, songs in [(Path(p).name, load(p)) for p in args.eval] + ([("test", te)] if te is not None else []):
        base_te = song_acc([s["base"].argmax(axis=1) for s in songs], songs)
        pred_te = [predict(net, s) for s in songs]
        acc_te = song_acc(pred_te, songs)
        print("{0} {1} songs: base exact {2:.2f} % / within 1 {3:.2f} %, gru exact {4:.2f} % / within 1 {5:.2f} %, "
              "gru - base {6:+.2f} [{7:+.2f} .. {8:+.2f}], songs better {9}, worse {10}".format(
                  label, len(songs), 100 * base_te[:, 0].sum() / base_te[:, 2].sum(),
                  100 * base_te[:, 1].sum() / base_te[:, 2].sum(), 100 * acc_te[:, 0].sum() / acc_te[:, 2].sum(),
                  100 * acc_te[:, 1].sum() / acc_te[:, 2].sum(), *boot(base_te, acc_te),
                  int(np.sum(acc_te[:, 0] > base_te[:, 0])), int(np.sum(acc_te[:, 0] < base_te[:, 0]))))
        print("{0} by length <60 .. >800 ms:    base {1} | gru {2}".format(
            label, by_length([s["base"].argmax(axis=1) for s in songs], songs), by_length(pred_te, songs)))
    return 0


def ensemble(args) -> int:
    """ exported networks (onnx) and averages of several (mean of the log softmax) on prepared arrays, paired
        per song against the first network (onnxruntime, CPU, no torch needed) """
    import onnxruntime as rt
    models = dict(m.split("=", 1) for m in args.models)
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = {k: rt.InferenceSession(v, opts, providers=["CPUExecutionProvider"]) for k, v in models.items()}
    combos = [c.split("+") for c in args.combos.split(",")] if args.combos else [[k] for k in models]
    ref_name = combos[0]

    def log_softmax(x):
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

    for path in args.arrays:
        z = dict(np.load(path))
        order = np.argsort(z["song"], kind="stable")
        bounds = np.searchsorted(z["song"][order], np.arange(len(z["names"]) + 1))
        acc = {i: [] for i in range(len(combos))}
        for s in range(len(z["names"])):
            idx = order[bounds[s]:bounds[s + 1]]
            feed = {k: z[k][idx][None].astype(np.float32) for k in ("circ", "other", "base")}
            out = {k: log_softmax(v.run(None, feed)[0][0].astype(np.float64)) for k, v in sess.items()}
            for i, c in enumerate(combos):
                d = dist(np.mean([out[k] for k in c], axis=0).argmax(axis=1), z["ref"][idx])
                acc[i].append([np.sum(d == 0), len(d)])
        acc = {i: np.array(v, dtype=float) for i, v in acc.items()}
        boot = np.random.default_rng(0).integers(0, len(z["names"]), (4000, len(z["names"])))
        ref = acc[0]
        print("{0} ({1} songs), paired against {2}".format(path, len(z["names"]), "+".join(ref_name)))
        for i, c in enumerate(combos):
            a = acc[i]
            diff = a[boot, 0].sum(1) / a[boot, 1].sum(1) - ref[boot, 0].sum(1) / ref[boot, 1].sum(1)
            per_song = a[:, 0] - ref[:, 0]
            print("  {0:34s} exact {1:6.2f} %  {2:+.2f} [{3:+.2f} .. {4:+.2f}], songs better {5}, worse {6}".format(
                "+".join(c), 100 * a[:, 0].sum() / a[:, 1].sum(),
                100 * (a[:, 0].sum() / a[:, 1].sum() - ref[:, 0].sum() / ref[:, 1].sum()),
                100 * np.percentile(diff, 2.5), 100 * np.percentile(diff, 97.5),
                int((per_song > 0).sum()), int((per_song < 0).sum())), flush=True)
    return 0


def combine(args) -> int:
    """ several exported networks -> one onnx with the same inputs and output: the mean of their log softmax
        (exactly what `ensemble` evaluates; the tool only takes the argmax) """
    import onnx
    from onnx import compose, helper
    models = [onnx.load(path) for path in args.models]
    assert len({(m.opset_import[0].version, m.ir_version) for m in models}) == 1, "different opsets"
    nodes, inits = [], []
    for i, m in enumerate(models):
        g = compose.add_prefix(m, "m{0}_".format(i)).graph       # everything renamed, inputs too
        nodes += [helper.make_node("Identity", [x.name], ["m{0}_{1}".format(i, x.name)]) for x in models[0].graph.input]
        nodes += list(g.node)
        inits += list(g.initializer)
        nodes.append(helper.make_node("LogSoftmax", ["m{0}_scores".format(i)], ["m{0}_logp".format(i)], axis=-1))
    nodes.append(helper.make_node("Mean", ["m{0}_logp".format(i) for i in range(len(models))], ["scores"]))
    graph = helper.make_graph(nodes, "context_gru_ensemble", list(models[0].graph.input), list(models[0].graph.output),
                              inits, doc_string="mean log softmax of " + ", ".join(Path(x).name for x in args.models))
    out = helper.make_model(graph, opset_imports=list(models[0].opset_import), producer_name="context_gru_2026.py combine")
    out.ir_version = models[0].ir_version
    onnx.checker.check_model(out)
    onnx.save(out, args.out)
    print("{0} networks, {1:,} parameters -> {2}".format(
        len(models), sum(int(np.prod(t.dims)) for t in inits), args.out))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("cache")
    a.add_argument("--out", required=True)
    a.add_argument("--model", default="base", help="name of the frame model scores in the cache")
    a.add_argument("--vocals", action="store_true", help="mix and vocals combined (default: mix only)")
    a.add_argument("--w", type=float, default=0.25, help="RMVPE weight")
    a.add_argument("--v", type=float, default=0.5, help="vocals weight")
    a.add_argument("--k", type=float, default=0.15, help="key prior weight")
    a.add_argument("--tuning", type=float, default=None, help="frame model with the tuning correction when the "
                   "RMVPE estimate is reliable and at least this many cents off (needs extract --tuned)")
    a.add_argument("--separate", action="store_true", help="experiment: also the scores of every single source "
                   "(frame model and RMVPE of mix [and vocals]) as circular inputs (not supported by --export)")
    m = sub.add_parser("merge", help="concatenate prepared files (e.g. mix and mix + vocals versions)")
    m.add_argument("out")
    m.add_argument("inputs", nargs="+")
    c = sub.add_parser("policy", help="tuning correction thresholds of the pipeline on a cache (no GRU)")
    c.add_argument("cache")
    c.add_argument("--model", default="base")
    c.add_argument("--thresholds", default="0,5,10,15,20")
    c.add_argument("--w", type=float, default=0.25)
    c.add_argument("--v", type=float, default=0.5)
    c.add_argument("--k", type=float, default=0.15)
    d = sub.add_parser("compare", help="pipeline configurations on a cache (no GRU), paired against the first")
    d.add_argument("cache")
    d.add_argument("--configs", default="base,base@10", help="NAME or NAME@T (tuning correction from T cents)")
    d.add_argument("--w", type=float, default=0.25)
    d.add_argument("--v", type=float, default=0.5)
    d.add_argument("--k", type=float, default=0.15)
    b = sub.add_parser("train")
    b.add_argument("train")
    b.add_argument("val")
    b.add_argument("--test")
    b.add_argument("--epochs", type=int, default=15)
    b.add_argument("--arch", default="gru", choices=["gru", "transformer"])
    b.add_argument("--hidden", type=int, default=48, help="GRU hidden size per direction / transformer width")
    b.add_argument("--layers", type=int, default=1)
    b.add_argument("--lr", type=float, default=2e-3)
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--limit-train", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--batch", type=int, default=1, help="songs per step (padded)")
    b.add_argument("--text", action="store_true", help="the hashed syllable of every note as an embedding")
    b.add_argument("--device", default="cpu")
    b.add_argument("--load", help="evaluate a network saved with --save instead of training (train file unused)")
    b.add_argument("--export", help="write the chosen network as onnx (GRU without text)")
    b.add_argument("--eval", action="append", default=[], help="further prepared arrays to report (repeatable)")
    b.add_argument("--save")
    e = sub.add_parser("ensemble", help="exported networks and their averages on prepared arrays (onnxruntime)")
    e.add_argument("arrays", nargs="+", help="prepared arrays (e.g. the validation mix and mix + vocals)")
    e.add_argument("--models", nargs="+", required=True, help="NAME=PATH.onnx, the first one is the reference")
    e.add_argument("--combos", help="comma separated, networks joined by '+', e.g. pkg,pkg+s1 (default: every network)")
    f = sub.add_parser("combine", help="exported networks -> one onnx with the mean of their log softmax")
    f.add_argument("out")
    f.add_argument("models", nargs="+")
    args = ap.parse_args(argv)
    return {"prepare": prepare, "merge": merge, "policy": policy, "compare": compare, "train": train,
            "ensemble": ensemble, "combine": combine}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
