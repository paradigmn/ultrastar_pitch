#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          pitchnet_train.py
@brief         train pitchnet_2026 on the cache of pitchnet_data.py (one run = one folder of results)
@author        paradigm

    python dev/pitchnet_train.py --name base --loss smooth --aug remix,gain,polarity
    python dev/pitchnet_train.py --name near --loss neighbour --aug remix,gain,polarity,tuning,noise

Writes dev/results/pitchnet_runs/<name>/: log.txt, history.json (one entry per epoch), best.pt (the epoch with
the best exact note accuracy on the validation songs with the mix as input), model.onnx and model_wave.onnx of it.

Training examples are random crops of CROP_S seconds of the cached songs (dev/pitchnet_2026.ipynb explains the
cache). Augmentations (--aug):
  remix      the accompaniment (mix - vocals) is added to the vocals with a random gain: half of the crops the
             original mix (gain 1), a quarter the vocals alone (gain 0), the rest a gain in 0 .. 1.5. Without
             it every crop is the mix
  gain       random level -6 .. +6 dB
  polarity   random sign
  tuning     random resampling by up to +-0.35 semitones (the labels keep their class, the frame grid is
             stretched along), so a detuned singer or instrument tuning is seen in training
  noise      white noise at a random SNR of 10 .. 40 dB with probability 0.3
Losses (--loss) on the labeled frames:
  smooth     cross entropy with label smoothing 0.05
  neighbour  cross entropy against a soft target that gives the two neighbouring pitch classes 0.05 each
             (and 0.05 spread over all): a prediction one semitone off is a smaller error than a random one,
             which is what the game's tolerance rewards too
plus 0.2 x binary cross entropy of the voicing on the frames with a voicing target.
Metrics per epoch: exact and within-one-semitone pitch class accuracy per note (probabilities summed over the
frames of the note, as the tool does), with the mix and with the vocals of the validation songs.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitchnet_model import HOP, N_FFT, SR, PitchNet2026, export_onnx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "dev" / "results" / "pitchnet_cache"
LISTS = ROOT / "dev" / "lists" / "pitchnet"
RUNS = ROOT / "dev" / "results" / "pitchnet_runs"
CROP_S = 4.0
CROP = int(CROP_S * SR)
N_FRAMES = CROP // HOP + 1
TUNING_MAX = 0.35             # semitones
NOISE_P, NOISE_SNR = 0.3, (10.0, 40.0)
VOICING_WEIGHT = 0.2
AUGMENTATIONS = ("remix", "gain", "polarity", "tuning", "noise")


def read_selection(split: str, limit: int = 0) -> list:
    with open(LISTS / "selection.csv", newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["split"] == split and (CACHE / (r["id"] + ".labels.npz")).exists()]
    return rows[:limit] if limit else rows


class SongCrops(Dataset):
    """ random crops of the cached songs; one epoch = crops_per_song crops of every song """

    def __init__(self, rows, crops_per_song=8, aug=(), seed=0):
        self.rows, self.crops_per_song, self.aug = rows, crops_per_song, set(aug)
        self.seed = seed
        self.rng = None
        # plain arrays (an open NpzFile can't be sent to the worker processes), about 50 kB per song
        self.labels = []
        for r in rows:
            with np.load(CACHE / (r["id"] + ".labels.npz")) as z:
                self.labels.append({"labels": z["labels"], "voicing": z["voicing"]})

    def __len__(self):
        return len(self.rows) * self.crops_per_song

    def __getitem__(self, index):
        if self.rng is None:   # one generator per worker process
            self.rng = np.random.default_rng([self.seed, os.getpid(), index])
        rng = self.rng
        r = self.rows[index % len(self.rows)]
        z = self.labels[index % len(self.rows)]
        labels, voicing = z["labels"], z["voicing"]
        mix = np.load(CACHE / (r["id"] + ".mix.npy"), mmap_mode="r")
        vocals = np.load(CACHE / (r["id"] + ".vocals.npy"), mmap_mode="r")
        n = min(len(mix), len(vocals))
        # resampling factor of the tuning augmentation: the source crop is longer by that factor
        factor = 2.0 ** (rng.uniform(-TUNING_MAX, TUNING_MAX) / 12.0) if "tuning" in self.aug else 1.0
        src_len = int(np.ceil(CROP * factor)) + 1
        # a crop that contains labeled frames: start at a random labeled frame minus a random part of the crop
        labeled = np.flatnonzero(labels >= 0)
        centre = int(labeled[rng.integers(len(labeled))]) * HOP if len(labeled) else n // 2
        start = int(np.clip(centre - rng.integers(CROP), 0, max(0, n - src_len)))
        v = vocals[start:start + src_len].astype(np.float32) / 32768.0
        m = mix[start:start + src_len].astype(np.float32) / 32768.0
        if "remix" in self.aug:
            u = rng.random()
            gain_acc = 1.0 if u < 0.5 else 0.0 if u < 0.75 else float(rng.uniform(0.0, 1.5))
            x = v + gain_acc * (m - v)
        else:
            x = m
        if len(x) < src_len:
            x = np.pad(x, (0, src_len - len(x)))
        f0 = start // HOP
        if factor != 1.0:
            # y[i] = x[i * factor]: pitch * factor, time / factor; frame t of y is frame t * factor of x
            x = np.interp(np.arange(CROP) * factor, np.arange(len(x)), x).astype(np.float32)
            src_frames = np.rint(np.arange(N_FRAMES) * factor).astype(int) + f0
        else:
            x = x[:CROP]
            src_frames = np.arange(N_FRAMES) + f0
        if "gain" in self.aug:
            x = x * 10 ** (rng.uniform(-6, 6) / 20)
        if "polarity" in self.aug and rng.random() < 0.5:
            x = -x
        if "noise" in self.aug and rng.random() < NOISE_P:
            rms = max(float(np.sqrt(np.mean(x ** 2))), 1e-4)
            x = x + rng.normal(0.0, rms * 10 ** (-rng.uniform(*NOISE_SNR) / 20), size=len(x)).astype(np.float32)
        valid = src_frames < len(labels)
        y = np.full(N_FRAMES, -1, dtype=np.int64)
        y[valid] = labels[src_frames[valid]]
        w = np.full(N_FRAMES, -1, dtype=np.float32)
        w[valid] = voicing[src_frames[valid]]
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(y), torch.from_numpy(w)


def pitch_loss(logprob, y, kind: str) -> torch.Tensor:
    """ logprob [N, 12], y [N] with -1 = ignore """
    valid = y >= 0
    if not valid.any():
        return logprob.sum() * 0.0
    logprob, y = logprob[valid], y[valid]
    if kind == "smooth":
        return F.cross_entropy(logprob, y, label_smoothing=0.05)
    if kind == "neighbour":
        eps, delta = 0.05, 0.05
        target = torch.full_like(logprob, eps / 12)
        idx = torch.arange(len(y), device=y.device)
        target[idx, y] += 1.0 - eps - 2 * delta
        target[idx, (y + 1) % 12] += delta
        target[idx, (y - 1) % 12] += delta
        return -(target * logprob).sum(dim=1).mean()
    raise ValueError("unknown loss " + kind)


def note_accuracy(net, rows, device, source="mix", chunk_s=30) -> dict:
    """ exact and within-one-semitone pitch class accuracy per note (probabilities summed inside the note) """
    net.eval()
    hit = near = total = 0
    with torch.no_grad():
        for r in rows:
            z = np.load(CACHE / (r["id"] + ".labels.npz"))
            audio = np.load(CACHE / (r["id"] + ".{0}.npy".format(source)), mmap_mode="r")
            x = torch.from_numpy(audio.astype(np.float32) / 32768.0)[None].to(device)
            # chunks without margin are fine for a validation estimate; a last chunk shorter than the STFT
            # padding is merged into the previous one. A chunk of L samples gives L // HOP + 1 frames, the last
            # one centred on the first sample of the next chunk, so it is dropped: T = N // HOP + 1 as a whole
            n = x.shape[1]
            bounds = list(range(0, n, chunk_s * SR)) + [n]
            if len(bounds) > 2 and bounds[-1] - bounds[-2] < N_FFT:
                bounds.pop(-2)
            probs = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                lp, _ = net.forward_wave(x[:, s:e])
                probs.append(lp[0, :-1 if e < n else None].exp().cpu().numpy())
            prob = np.concatenate(probs)
            cs = np.vstack([np.zeros((1, 12)), np.cumsum(prob, axis=0)])
            lo = np.minimum((z["segments"][:, 0] / (HOP / SR)).round().astype(int), len(prob))
            hi = np.minimum((z["segments"][:, 1] / (HOP / SR)).round().astype(int), len(prob))
            hi = np.maximum(hi, lo + 1)
            pred = (cs[hi] - cs[lo]).argmax(axis=1)
            dist = np.abs((pred - z["classes"] + 6) % 12 - 6)
            hit += int((dist == 0).sum()); near += int((dist <= 1).sum()); total += len(pred)
    return {"exact": hit / max(total, 1), "within1": near / max(total, 1), "notes": total}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--loss", default="smooth", choices=["smooth", "neighbour"])
    ap.add_argument("--aug", default="remix,gain,polarity", help="comma separated, of: " + ", ".join(AUGMENTATIONS) + " or none")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--crops-per-song", type=int, default=8)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--stem-ch", type=int, default=16)
    ap.add_argument("--dilations", default="1,12,1", help="frequency dilation (semitones) per block, comma separated")
    ap.add_argument("--time-dilations", default="", help="time dilation (frames) per block (default: 1 for every block)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="any cached songs as train and val (a quick test of the code)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    aug = tuple(a for a in args.aug.split(",") if a and a != "none")
    unknown = set(aug) - set(AUGMENTATIONS)
    if unknown:
        ap.error("unknown augmentation " + ", ".join(sorted(unknown)))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a", encoding="utf-8")

    def say(text):
        print(text, flush=True)
        log.write(text + "\n"); log.flush()

    if args.smoke:
        rows = read_selection("val") + read_selection("train")
        train_rows, val_rows = rows[:max(2, args.limit_train)], rows[:max(2, args.limit_val)]
    else:
        train_rows, val_rows = read_selection("train", args.limit_train), read_selection("val", args.limit_val)
    say("{0}: {1} | device {2} | train {3} songs, val {4} songs | loss {5} | aug {6}".format(
        args.name, time.strftime("%Y-%m-%d %H:%M"), device, len(train_rows), len(val_rows), args.loss, ",".join(aug) or "none"))
    if not train_rows or not val_rows:
        say("no cached songs, run dev/pitchnet_data.py extract first")
        return 1

    dil_f = tuple(int(x) for x in args.dilations.split(","))
    dil_t = tuple(int(x) for x in args.time_dilations.split(",")) if args.time_dilations else None
    net = PitchNet2026(stem_ch=args.stem_ch, ch=args.ch, dilations=dil_f, time_dilations=dil_t).to(device)
    say("parameters: {0:,} | config {1}".format(sum(p.numel() for p in net.parameters()), net.config))
    loader = DataLoader(SongCrops(train_rows, args.crops_per_song, aug, args.seed), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.1)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        losses = []
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            logprob, voicing = net.forward_wave(x)
            loss = pitch_loss(logprob.reshape(-1, 12), y.reshape(-1), args.loss)
            mask = w >= 0
            if mask.any():
                loss = loss + VOICING_WEIGHT * F.binary_cross_entropy_with_logits(voicing[mask], w[mask])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        t_train = time.time() - t0
        mix, voc = note_accuracy(net, val_rows, device, "mix"), note_accuracy(net, val_rows, device, "vocals")
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "mix_exact": mix["exact"], "mix_within1": mix["within1"],
                 "vocals_exact": voc["exact"], "vocals_within1": voc["within1"], "train_s": t_train, "epoch_s": time.time() - t0}
        history.append(entry)
        say("epoch {epoch:2d}: loss {loss:.4f} | val notes exact / within 1: mix {mix_exact:.2%} / {mix_within1:.2%}, "
            "vocals {vocals_exact:.2%} / {vocals_within1:.2%} | {train_s:.0f} s + val {val:.0f} s".format(
                val=entry["epoch_s"] - t_train, **entry))
        with open(out / "history.json", "w", encoding="utf-8") as file:
            json.dump({"args": vars(args), "history": history}, file, indent=1)
        if best is None or mix["exact"] > best[0]:
            best = (mix["exact"], epoch)
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "config": net.config,
                        "epoch": epoch, "val": entry, "args": vars(args), "history": history}, out / "best.pt")
    say("best epoch {0} (val mix exact {1:.2%})".format(best[1], best[0]))
    ckpt = torch.load(out / "best.pt", map_location="cpu")
    net = PitchNet2026(**ckpt["config"])
    net.load_state_dict(ckpt["state_dict"])
    export_onnx(net, out / "model.onnx")
    export_onnx(net, out / "model_wave.onnx", wave=True)
    say("exported {0} and model_wave.onnx".format(out / "model.onnx"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
