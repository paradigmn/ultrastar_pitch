#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file          export_demucs_vocals.py
@brief         export the vocal stem of HTDemucs (Demucs v4) to ONNX for ultrastar_pitch
@author        paradigm

Usage:
    python export_demucs_vocals.py [--model htdemucs_ft] --vocals-only models/htdemucs_vocals.onnx

    (the tool expects the --vocals-only export: same output as the export with all four stems,
    but about 16 % faster)

The exported model is used by ultrastar_pitch/vocal_separator.py (numpy + onnxruntime only).
This script is the only part which needs torch and the ONNX-capable demucs fork:

    pip install torch torchaudio einops julius lameenc dora-search openunmix pyyaml tqdm omegaconf onnx onnxruntime
    pip install --no-deps git+https://github.com/mixxxdj/demucs.git

Why the fork?
    Upstream demucs (adefossez/demucs, v4.1.0) can't be exported, because torch.stft / istft
    produce complex tensors. The Mixxx fork adds an `onnx_exportable` flag, which replaces them
    by real-valued convolutions (numerically equivalent, no retraining needed).

Notes:
    * the graph has a FIXED input length (model.segment, 7.8 s @ 44.1 kHz -> (1, 2, 343980))
    * htdemucs_ft is a bag of four specialists, only the vocals specialist is exported
    * the license of the pretrained weights is unclear (see the demucs README)
    * by default the graph returns all four stems, shape (1, 4, 2, 343980), and the caller picks
      the vocals index. With --vocals-only the other three stems are cut out of the last decoder
      layers before the export: the output is (1, 1, 2, 343980) and bit-identical to the vocals
      channel of the full model, but the inverse STFT (a dense ConvTranspose) and the last decoder
      layers only compute one stem instead of four (roughly 15 % faster in a CPU benchmark of
      a random-weight model with the same architecture, not measured with the real weights).
      The caller must then read output[0, -1] instead of output[0, 3]. The ONNX metadata keys
      "sources" and "vocals_index" describe the exported output.
"""
import argparse
import inspect
from pathlib import Path

import numpy as np


def pick_core_model(model, stem="vocals"):
    """ returns the single HTDemucs network responsible for the stem and the stem's output index """
    from demucs.htdemucs import HTDemucs

    if isinstance(model, HTDemucs):
        return model, model.sources.index(stem)
    idx = model.sources.index(stem)
    if len(model.models) == 1:  # htdemucs
        return model.models[0], idx
    for core, weights in zip(model.models, model.weights):  # htdemucs_ft: one specialist per stem
        if weights[idx] == 1.0 and sum(weights) == 1.0:
            return core, idx
    raise ValueError("no specialist model for stem '{0}' found in bag".format(stem))


def keep_only_stem(core, stem_index):
    """ cut all stems except one out of the network (in place)

    The last decoder layers (frequency and time branch) are transposed convolutions whose output
    channels are grouped by stem: x.view(B, S, -1, Fq, T). Keeping only the channels of one stem
    gives exactly the same values for this stem, because the output channels of a convolution do
    not depend on each other (the last layers have no normalization).
    """
    import torch.nn as nn

    n_stems = len(core.sources)
    for layers in (core.decoder, core.tdecoder):
        last = layers[-1]  # the layer built for index 0 comes last in the list
        if not last.last or not isinstance(last.norm2, nn.Identity):
            raise RuntimeError("unexpected decoder structure, can't cut stems")
        conv = last.conv_tr
        per_stem = conv.out_channels // n_stems
        keep = slice(stem_index * per_stem, (stem_index + 1) * per_stem)
        conv.weight = nn.Parameter(conv.weight.data[:, keep].clone())  # (in, out, ...)
        conv.bias = nn.Parameter(conv.bias.data[keep].clone())
        conv.out_channels = per_stem
    core.sources = [core.sources[stem_index]]
    return core


def export_onnx(core, path, opset=17, check=True, stem_index=None):
    """ export a HTDemucs network to ONNX and compare the result with PyTorch

    stem_index: None exports all stems, otherwise only this stem (see keep_only_stem)
    """
    import torch
    from demucs.htdemucs import HTDemucs

    if "onnx_exportable" not in inspect.signature(HTDemucs.__init__).parameters:
        raise RuntimeError(
            "this demucs version has no ONNX support, install the fork: "
            "pip install --no-deps git+https://github.com/mixxxdj/demucs.git"
        )
    core.eval()
    n_samples = int(core.segment * core.samplerate)
    torch.manual_seed(0)
    dummy = torch.randn(1, core.audio_channels, n_samples)
    with torch.no_grad():
        core.onnx_exportable = False
        reference = core(dummy).numpy()
    sources = list(core.sources)
    if stem_index is not None:
        reference = reference[:, stem_index:stem_index + 1]
        keep_only_stem(core, stem_index)
    core.onnx_exportable = True
    args = dict(export_params=True, opset_version=opset, do_constant_folding=True,
                input_names=["input"], output_names=["output"])
    # NOTE: don't export inside torch.no_grad(): in eval mode PyTorch would then use its fused
    # attention fast path (aten::_native_multi_head_attention), which has no ONNX equivalent.
    try:
        # torch >= 2.9 uses the torch.export based exporter by default, which fails on
        # this model. The TorchScript based exporter works.
        torch.onnx.export(core, (dummy,), str(path), dynamo=False, **args)
    except TypeError:  # torch < 2.5 has no 'dynamo' argument
        torch.onnx.export(core, (dummy,), str(path), **args)
    try:
        import onnx
        model = onnx.load(str(path))
        for key, value in (("sources", ",".join(sources if stem_index is None else [sources[stem_index]])),
                           ("vocals_index", str(sources.index("vocals") if stem_index is None else 0))):
            entry = model.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.save(model, str(path))
    except ImportError:
        print("onnx not installed, metadata not written")
    print("exported to {0} ({1:.0f} MB)".format(path, Path(path).stat().st_size / 1e6))
    if check:
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        result = session.run(None, {"input": dummy.numpy()})[0]
        if result.shape != reference.shape:
            raise RuntimeError("output shape {0} differs from PyTorch {1}".format(result.shape, reference.shape))
        err = float(np.abs(result - reference).max())
        print("max abs error onnx vs torch: {0:.2e} (output peak {1:.2e})".format(
            err, float(np.abs(reference).max())))
        if err > 1e-3:
            raise RuntimeError("ONNX output deviates from PyTorch, don't use this export")


def main():
    parser = argparse.ArgumentParser(description="export the vocals model of HTDemucs to ONNX")
    parser.add_argument("output", help="output file, e.g. htdemucs_vocals.onnx")
    parser.add_argument("--model", default="htdemucs_ft", help="htdemucs or htdemucs_ft")
    parser.add_argument("--vocals-only", action="store_true",
                        help="export only the vocals stem, output (1, 1, 2, N) instead of (1, 4, 2, N); "
                             "the caller must read output[0, -1]")
    args = parser.parse_args()
    from demucs.pretrained import get_model
    core, idx = pick_core_model(get_model(args.model))
    export_onnx(core, args.output, stem_index=idx if args.vocals_only else None)


if __name__ == "__main__":
    main()
