#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          fcpe_val.py
@brief         FCPE as an independent third pitch model on the validation songs
@author        paradigm

FCPE (Fast Context-based Pitch Estimation, torchfcpe by CNChTu, MIT licence, https://github.com/CNChTu/FCPE) is
a singing pitch estimator trained on other data than our karaoke labels. Its per-frame pitch distribution (360
bins, 10 ms) is mapped onto the RMVPE bin grid and folded to 12 pitch classes per note exactly like RMVPE
(rmvpe.chroma_per_note, tuning offset = the cached RMVPE estimate of the same source).

    # 1. per note scores of the 249 validation songs (GPU; separate cache, the main cache stays untouched)
    .venv/bin/python dev/fcpe_val.py extract
    # 2. FCPE as an extra vote in the cached pipeline (before / with the note context network), val only
    CUDA_VISIBLE_DEVICES= .venv/bin/python dev/fcpe_val.py ensemble > dev/results/fcpe_ensemble_val.txt

dev/remaining_errors_val.py reads the cache too and reports FCPE's vote on the remaining errors.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import pipeline_weights_val as pw  # noqa: E402

CACHE = ROOT / "dev" / "results" / "fcpe_cache"
CHUNK_S = 120.0         # FCPE runs on chunks of this length ...
CONTEXT_S = 5.0         # ... with this much audio on each side


class Fcpe:
    def __init__(self, device="cuda"):
        import torch
        from torchfcpe import spawn_bundled_infer_model
        self.torch = torch
        self.device = device
        self.model = spawn_bundled_infer_model(device=device)
        from ultrastar_pitch import rmvpe as rm
        self.rm = rm
        cents = self.model.model.cent_table.detach().cpu().numpy().astype(np.float64)
        target = rm.CENTS_BASE + rm.CENTS_STEP * np.arange(rm.N_CLASS)
        # linear interpolation FCPE bins -> RMVPE bins (0 outside FCPE's range)
        pos = np.interp(target, cents, np.arange(len(cents)), left=-1, right=-1)
        self.valid = pos >= 0
        self.lo = np.clip(np.floor(pos).astype(int), 0, len(cents) - 2)
        self.frac = np.clip(pos - self.lo, 0, 1)

    def latent(self, samples: np.ndarray) -> np.ndarray:
        """ FCPE pitch distribution [frames, 360] of 16 kHz audio, frame i at i * 10 ms """
        torch, sr, hop = self.torch, 16000, 160
        samples = np.asarray(samples, dtype=np.float32)
        if samples.dtype.kind in "iu" or np.abs(samples).max(initial=0.0) > 1.5:
            samples = samples / 32768.0          # the parser decodes int16 (parity with the old pitchnet)
        n_frames = len(samples) // hop + 1
        out = np.zeros((n_frames, 360), dtype=np.float32)
        chunk, ctx = int(CHUNK_S * sr), int(CONTEXT_S * sr)
        for start in range(0, len(samples), chunk):
            a, b = max(0, start - ctx), min(len(samples), start + chunk + ctx)
            x = torch.from_numpy(np.ascontiguousarray(samples[a:b], dtype=np.float32))[None, :, None].to(self.device)
            with torch.no_grad():
                lat = self.model.model.forward(self.model.wav2mel(x, sr))[0].float().cpu().numpy()
            f0 = start // hop                      # first frame of this chunk in the song
            skip = (start - a) // hop              # context frames at the start of the window
            n = min(chunk // hop, n_frames - f0, len(lat) - skip)
            out[f0:f0 + n] = lat[skip:skip + n]
        return out

    def salience(self, samples: np.ndarray) -> np.ndarray:
        """ FCPE distribution on the RMVPE bin grid [frames, 360] """
        lat = self.latent(samples)
        sal = lat[:, self.lo] * (1 - self.frac) + lat[:, self.lo + 1] * self.frac
        return sal * self.valid

    def note_scores(self, samples, segments, tuning_cents: float) -> np.ndarray:
        """ log chroma per note [n, 12], folded like the RMVPE scores of the pipeline """
        sal = self.salience(samples)
        return np.log(self.rm.chroma_per_note(sal, segments, tuning_cents) + self.rm.EPS), sal


def extract(args) -> int:
    from ultrastar_pitch import ProjectParser
    CACHE.mkdir(parents=True, exist_ok=True)
    fcpe = Fcpe(args.device)
    parser = ProjectParser()
    rows = pw.val_songs(args.limit)
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        out = CACHE / (row["id"] + ".npz")
        if out.exists() and not args.force:
            continue
        z = dict(np.load(pw.CACHE / (row["id"] + ".npz")))
        parser.load_note_file(row["path"])
        applied = float(z["applied_s"])
        segments = [(s + applied, e + applied) for s, e in parser.note_segments()]
        assert len(segments) == len(z["ref"]), row["id"]
        data = {}
        for source in ("mix", "vocals"):
            if "rmvpe_" + source not in z:
                continue
            if source == "mix":
                samples = parser.load_main_audio(16000)
            else:
                samples = parser.load_audio(16000, separator=None, use_vocals_file=True)
                if parser.audio_source not in ("#VOCALS", "vocals.wav", "separated"):
                    continue
            scores, sal = fcpe.note_scores(samples, segments, float(z["tuning_" + source][0]))
            data["fcpe_" + source] = scores
            # strength of the voice per note: mean of the per-frame maximum (FCPE's own confidence)
            times = np.arange(len(sal)) * 0.01
            peak = sal.max(axis=1)
            data["fcpe_strength_" + source] = np.array(
                [peak[np.searchsorted(times, s):max(np.searchsorted(times, e), np.searchsorted(times, s) + 1)].mean()
                 for s, e in segments])
        pw.save_npz(out, data)
        print("[{0}/{1}] {2} {3}: {4} ({5:.0f} s)".format(i, len(rows), row["id"], Path(row["path"]).parent.name[:40],
                                                         ", ".join(sorted(data)), time.time() - t0), flush=True)
    return 0


def ensemble(args) -> int:
    """ FCPE as a third log-linear vote in each source: S = (1 - u) * (w RMVPE + (1 - w) frame) + u * FCPE, then
        pooling, key prior and (optionally) the packaged note context network on top, as trained without FCPE """
    import context_gru_2026 as cg
    from ultrastar_pitch import context_model as cm
    repetition_pooling, SP = cg._package()
    net = cm.ContextModel()
    w, v, k = 0.25, 0.5, 0.15
    rows = pw.val_songs()
    zs = [(dict(np.load(pw.CACHE / (r["id"] + ".npz"))), dict(np.load(CACHE / (r["id"] + ".npz")))) for r in rows]
    targets = {}
    for vocals in (False, True):
        a = dict(np.load(ROOT / "dev" / "results" / "gru2026_val_ch64t{0}.npz".format("_vocals" if vocals else "")))
        order = np.argsort(a["song"], kind="stable")
        bounds = np.searchsorted(a["song"][order], np.arange(len(a["names"]) + 1))
        assert list(a["names"]) == [r["id"] for r in rows]
        targets[vocals] = [a["target"][order[bounds[s]:bounds[s + 1]]] for s in range(len(rows))]

    def song(z, f, vocals, u, fcpe_sources):
        def source(src):
            s = w * z["rmvpe_" + src] + (1 - w) * cg.frame_scores(z, "ch64", src, 0.0)
            return (1 - u) * s + u * f["fcpe_" + src] if src in fcpe_sources else s
        E = source("mix")
        if vocals:
            E = (1 - v) * E + v * source("vocals")
        context = {key[4:]: z[key].tolist() for key in z if key.startswith("ctx_")}
        Erep = repetition_pooling.pool_repetitions(E, weight=1.0, **context)
        keylp = np.log(SP.key_prior(SP.detect_key(Erep.argmax(axis=1).tolist())))[None].repeat(len(E), axis=0)
        base = Erep + k * keylp
        out = net.correct(E, Erep, keylp, base, z["durations"], z["ctx_beats"], z["ctx_lengths"], z["ctx_phrases"])
        return base.argmax(1), out.argmax(1)

    def boot(a, b, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(a), (4000, len(a)))
        d = b[idx, 0].sum(1) / b[idx, 1].sum(1) - a[idx, 0].sum(1) / a[idx, 1].sum(1)
        return (100 * (b[:, 0].sum() / b[:, 1].sum() - a[:, 0].sum() / a[:, 1].sum()),
                100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5))

    settings = [("mix", False, ("mix",)), ("mix + vocals, FCPE on both", True, ("mix", "vocals")),
                ("mix + vocals, FCPE on the vocals only", True, ("vocals",))]
    print("249 validation songs; S = (1-u)(w RMVPE + (1-w) frame) + u FCPE per source, w {0}, v {1}, k {2};".format(w, v, k))
    print("'before GRU' = pooling + key prior, 'with GRU' = + packaged note context network (trained without FCPE);")
    print("difference of pooled accuracies vs u = 0, songs resampled (4000), 95 % interval")
    for label, vocals, fsrc in settings:
        print()
        print(label)
        ref = None
        for u in args.weights:
            acc = []
            for (z, f), tgt in zip(zs, targets[vocals]):
                b, g = song(z, f, vocals, u, fsrc)
                acc.append([(b == tgt).sum(), (g == tgt).sum(), len(tgt)])
            acc = np.array(acc, dtype=float)
            before, with_gru = acc[:, [0, 2]], acc[:, [1, 2]]
            if ref is None:
                ref = (before, with_gru)
            print("  u {0:4.2f}: before GRU {1:6.2f} % {2:+.2f} [{3:+.2f} .. {4:+.2f}] | with GRU {5:6.2f} % {6:+.2f} [{7:+.2f} .. {8:+.2f}]".format(
                u, 100 * before[:, 0].sum() / before[:, 1].sum(), *boot(ref[0], before),
                100 * with_gru[:, 0].sum() / with_gru[:, 1].sum(), *boot(ref[1], with_gru)), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--device", default="cuda")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--force", action="store_true")
    s = sub.add_parser("ensemble")
    s.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    args = ap.parse_args(argv)
    return {"extract": extract, "ensemble": ensemble}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
