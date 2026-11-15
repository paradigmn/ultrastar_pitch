#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          pitchnet_model.py
@brief         the pitchnet_2026 model: front end, harmonic stacking, transposition equivariant body, ONNX export
@author        paradigm

The reasoning behind the design is in dev/pitchnet_2026.ipynb. In short: the model takes the power spectrum of
the 536 FFT bins below 4.2 kHz per 10 ms frame (power_spectrogram, scipy; or the waveform with the STFT inside
via forward_wave / WaveModel) and returns per frame the log probabilities of the 12 pitch classes and a voicing
logit. Everything after the STFT is in the model: the log-frequency filterbank (3 bins per semitone, C2 .. B7),
dB and per-frame standardisation, the harmonic stacking, convolutions over (time, frequency) that are exactly
equivariant to transpositions, a salience per semitone folded to the classes with logsumexp over the octaves.
"""

import math
from typing import Tuple

import numpy as np
import scipy.fft
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
N_FFT = 2048                  # 128 ms Hann window, 7.8 Hz per FFT bin
HOP = 160                     # 10 ms
BINS_PER_SEMITONE = 3
MIDI_LO = 36                  # C2 = 65.4 Hz; a multiple of 12, so semitone index % 12 = pitch class
N_SEMITONES = 72              # up to B7 = 3951 Hz
N_BINS = N_SEMITONES * BINS_PER_SEMITONE
HARMONICS = (0.5, 1, 2, 3, 4, 5)
STEM_CH, CH, DILATIONS = 16, 32, (1, 12, 1)
TIME_DILATIONS = (1, 1, 1)     # frames; (1, 1, 1) = +-4 frames of context in total with the stem
assert MIDI_LO % 12 == 0 and N_SEMITONES % 12 == 0


def log_filterbank(sr=SR, n_fft=N_FFT, bins_per_semitone=BINS_PER_SEMITONE, midi_lo=MIDI_LO, n_semitones=N_SEMITONES):
    """ (matrix [n_bins, n_fft // 2 + 1] of triangular filters that sum to one, centre frequencies) """
    n_bins = n_semitones * bins_per_semitone
    midi = midi_lo + np.arange(n_bins) / bins_per_semitone
    centres = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    fft_freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    df = sr / n_fft
    step = 2.0 ** (1.0 / (12 * bins_per_semitone))
    fb = np.zeros((n_bins, n_fft // 2 + 1))
    for k, c in enumerate(centres):
        lo = min(c / step, c - df)          # at least one FFT bin wide on each side
        hi = max(c * step, c + df)
        w = np.maximum(0.0, np.minimum((fft_freqs - lo) / (c - lo), (hi - fft_freqs) / (hi - c)))
        fb[k] = w / max(w.sum(), 1e-9)
    return fb.astype(np.float32), centres


FB, CENTRES = log_filterbank()
N_FFT_USED = int(np.max(np.nonzero(FB.sum(axis=0))[0])) + 1   # 536: FFT bins above this are not needed


def power_spectrogram(audio: np.ndarray, chunk: int = 4096, workers: int = -1) -> np.ndarray:
    """ float32 waveform [N] (-1 .. 1) -> power of the first N_FFT_USED FFT bins per 10 ms frame [T, N_FFT_USED],
        T = N // HOP + 1, frame t centred at t * HOP (reflect padded), periodic Hann window: the same numbers as
        torch.stft(center=True, pad_mode="reflect") in forward_wave. scipy.fft keeps float32 and uses all cores """
    pad = N_FFT // 2
    if len(audio) <= pad:   # the reflect padding needs pad + 1 samples (64 ms)
        audio = np.pad(audio, (0, pad + 1 - len(audio)))
    x = np.concatenate([audio[pad:0:-1], audio, audio[-2:-pad - 2:-1]]).astype(np.float32)
    n = 1 + (len(x) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, N_FFT), (x.strides[0] * HOP, x.strides[0]))
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
    out = np.empty((n, N_FFT_USED), dtype=np.float32)
    for s in range(0, n, chunk):   # chunks keep the buffers small
        spec = scipy.fft.rfft(frames[s:s + chunk] * window, axis=1, workers=workers)[:, :N_FFT_USED]
        out[s:s + chunk] = spec.real ** 2 + spec.imag ** 2
    return out


class FrontEnd(nn.Module):
    """ power spectrum [B, T, N_FFT_USED] -> standardised log-frequency spectrogram [B, T, N_BINS];
        from_wave: waveform [B, N] (-1 .. 1) -> the same, with the STFT inside (T = N // HOP + 1) """

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.register_buffer("fb", torch.from_numpy(FB[:, :N_FFT_USED].T.copy()))   # [N_FFT_USED, N_BINS]

    def forward(self, power):
        db = 10.0 * torch.log10(torch.matmul(power, self.fb) + 1e-10)
        mean = db.mean(dim=-1, keepdim=True)
        std = torch.sqrt(((db - mean) ** 2).mean(dim=-1, keepdim=True))
        return (db - mean) / (std + 1.0)                             # silent frames stay near 0

    def from_wave(self, audio):
        # return_complex=False (real/imaginary as the last axis) is what the ONNX exporter understands
        spec = torch.stft(audio, N_FFT, HOP, N_FFT, self.window, center=True, pad_mode="reflect", return_complex=False)
        return self(spec[:, :N_FFT_USED].pow(2).sum(-1).transpose(1, 2))


class HarmonicStack(nn.Module):
    """ [B, T, N_BINS] -> [B, H, T, N_BINS]: channel h holds the spectrogram shifted by the interval of harmonic h,
        so position k of every channel is the energy at h * f(k) """

    def __init__(self, harmonics=HARMONICS, bins_per_semitone=BINS_PER_SEMITONE):
        super().__init__()
        self.shifts = [int(round(12 * bins_per_semitone * math.log2(h))) for h in harmonics]
        self.pad = max(abs(s) for s in self.shifts)

    def forward(self, x):
        n = x.shape[-1]
        xp = F.pad(x, (self.pad, self.pad))
        return torch.stack([xp[..., self.pad + s:self.pad + s + n] for s in self.shifts], dim=1)


class DSBlock(nn.Module):
    """ depthwise-separable residual block over (time, frequency); dil_f: frequency dilation in semitones,
        dil_t: time dilation in frames """

    def __init__(self, ch, dil_f=1, dil_t=1):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, (3, 3), padding=(dil_t, dil_f), dilation=(dil_t, dil_f), groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


class PitchNet2026(nn.Module):
    """ power spectrum [B, T, N_FFT_USED] (or waveform [B, N] via forward_wave)
        -> (log probabilities of the 12 pitch classes [B, T, 12], voicing logit [B, T]) """

    def __init__(self, stem_ch=STEM_CH, ch=CH, dilations=DILATIONS, time_dilations=None):
        super().__init__()
        time_dilations = tuple(time_dilations) if time_dilations else (1,) * len(dilations)
        assert len(time_dilations) == len(dilations)
        self.config = {"stem_ch": stem_ch, "ch": ch, "dilations": tuple(dilations), "time_dilations": time_dilations}
        self.frontend = FrontEnd()
        self.stack = HarmonicStack()
        self.stem = nn.Sequential(
            nn.Conv2d(len(HARMONICS), stem_ch, (3, 5), stride=(1, BINS_PER_SEMITONE), padding=(1, 2), bias=False),
            nn.BatchNorm2d(stem_ch), nn.LeakyReLU(0.1),
            nn.Conv2d(stem_ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.LeakyReLU(0.1))
        self.blocks = nn.Sequential(*[DSBlock(ch, d, t) for d, t in zip(dilations, time_dilations)])
        self.head = nn.Conv2d(ch, 1, 1)
        self.voicing = nn.Linear(ch + 1, 1)

    def salience(self, spec):
        """ standardised log-frequency spectrogram [B, T, N_BINS] -> salience per semitone [B, T, 72] """
        return self.head(self.blocks(self.stem(self.stack(spec)))).squeeze(1)

    def body(self, spec):
        """ standardised log-frequency spectrogram [B, T, N_BINS] -> (log probabilities, voicing) """
        x = self.blocks(self.stem(self.stack(spec)))        # [B, C, T, 72]
        sal = self.head(x).squeeze(1)                        # salience per semitone [B, T, 72]
        b, t, s = sal.shape
        logits = torch.logsumexp(sal.reshape(b, t, s // 12, 12), dim=2)          # fold the octaves -> classes
        feat = torch.cat([x.mean(dim=3).transpose(1, 2), sal.max(dim=2, keepdim=True).values], dim=-1)
        return torch.log_softmax(logits, dim=-1), self.voicing(feat).squeeze(-1)

    def forward(self, power):
        return self.body(self.frontend(power))

    def forward_wave(self, audio):
        return self.body(self.frontend.from_wave(audio))


class WaveModel(nn.Module):
    """ the same network with the STFT inside, for the export """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, audio):
        return self.net.forward_wave(audio)


DOC = ("pitch class log probabilities and voicing per 10 ms frame; frame t at t*10 ms; 16 kHz; "
       "fold: logsumexp over octaves; dev/pitchnet_2026.ipynb")


def export_onnx(net: PitchNet2026, path, wave: bool = False) -> list:
    """ export to ONNX (opset 17), returns the operators used. wave: input "audio" [batch, samples] with the STFT
        inside, else input "power" [batch, frames, N_FFT_USED] """
    import onnx

    if wave:
        module, example, names, axes = WaveModel(net), torch.randn(1, 2 * SR) * 0.1, ["audio"], {"audio": {0: "batch", 1: "samples"}}
    else:
        module, example, names, axes = net, torch.rand(1, 201, N_FFT_USED), ["power"], {"power": {0: "batch", 1: "frames"}}
    axes.update({"pitch_logprob": {0: "batch", 1: "frames"}, "voicing": {0: "batch", 1: "frames"}})
    # the exporter restores the training flag of the module it gets afterwards (recursively): a wrapper
    # created in training mode would leave the network in training mode, so the wrapper is set to eval
    module.eval()
    torch.onnx.export(module, (example,), str(path), opset_version=17, dynamo=False, input_names=names,
                      output_names=["pitch_logprob", "voicing"], dynamic_axes=axes, do_constant_folding=True)
    net.eval()
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    m.doc_string = DOC + (" (input: waveform -1..1)" if wave else " (input: power of the first 536 FFT bins, see power_spectrogram)")
    onnx.save(m, str(path))
    return sorted({n.op_type for n in m.graph.node})


def load_checkpoint(path, device="cpu") -> Tuple[PitchNet2026, dict]:
    """ (model in eval mode, checkpoint dict) of a .pt written by pitchnet_train.py """
    ckpt = torch.load(path, map_location=device)
    net = PitchNet2026(**ckpt.get("config", {})).to(device)
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt


def harmonic_tone(f0: float, seconds: float = 2.0, n_harm: int = 8, sr: int = SR) -> torch.Tensor:
    """ a synthetic harmonic tone [1, N] for the checks """
    t = np.arange(int(seconds * sr)) / sr
    y = sum((0.6 ** h) * np.sin(2 * np.pi * f0 * (h + 1) * t) for h in range(n_harm))
    return torch.tensor(0.3 * y / np.abs(y).max(), dtype=torch.float32)[None]
