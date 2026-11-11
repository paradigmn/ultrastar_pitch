#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          vocal_separator.py
@brief         isolate the vocals of a song with an exported HTDemucs (Demucs v4) onnx model
@author        paradigm

The model has to be exported with dev/export_demucs_vocals.py (with or without --vocals-only,
the vocals channel is found automatically, see VocalSeparator). It has a fixed input length
(7.8 s, 44.1 kHz, stereo), longer songs are processed in overlapping chunks, which are
blended with a triangular window (same scheme as demucs.apply.apply_model with shifts=0).
Only numpy and onnxruntime are needed, torch isn't.
"""

import logging
import os
import sys
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

from .model_files import VOCAL_MODEL_ENV_VAR as MODEL_ENV_VAR, VOCAL_MODEL_FILE_NAME as MODEL_FILE_NAME, find_model


class VocalSeparatorError(Exception):
    """ base class of all errors raised by the vocal separator """


class SeparatorModelError(VocalSeparatorError):
    """ the model file is missing, corrupt or doesn't have the expected signature """


class SeparationFailedError(VocalSeparatorError):
    """ the separation of an audio signal failed """


def default_model_path() -> Optional[str]:
    """ path of the separation model ($ULTRASTAR_PITCH_VOCAL_MODEL, executable folder, user data
        folder, models/ next to the sources) or None """
    return find_model(MODEL_FILE_NAME, MODEL_ENV_VAR)


class VocalSeparator:
    """ isolate the vocal stem of a stereo signal (not thread safe, use one instance per thread) """

    SAMPLE_RATE = 44100

    def __init__(
        self,
        model_path: str,
        vocals_index: Optional[int] = None,
        overlap: float = 0.25,
        num_threads: int = 0,
        providers: Optional[Sequence[str]] = None,
    ):
        """ load the onnx model, vocals_index is the vocals channel of the model output.
            None (default) detects it: the onnx metadata key "vocals_index" if present (written by
            export_demucs_vocals.py), else 0 for a model with a single output stem, else 3 (drums,
            bass, other, vocals). num_threads=0 lets onnxruntime decide """
        if not 0 <= overlap < 1:
            raise ValueError("overlap has to be in the range [0, 1)")
        model_path = os.fspath(model_path)
        if not os.path.isfile(model_path):
            raise SeparatorModelError("vocal separation model not found: '{0}'".format(model_path))
        import onnxruntime as ort  # imported here, so the module stays cheap to import

        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        if num_threads > 0:
            options.intra_op_num_threads = num_threads
        try:
            active_providers = list(
                providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._session = ort.InferenceSession(
                model_path, options, providers=active_providers
            )
        except Exception as err:  # onnxruntime raises a variety of exception types
            raise SeparatorModelError(
                "can't load vocal separation model '{0}': {1}".format(model_path, err)
            ) from err
        in_shape = self._session.get_inputs()[0].shape
        out_shape = self._session.get_outputs()[0].shape
        if vocals_index is None:
            vocals_index = self._detect_vocals_index(out_shape)
        if (
            len(in_shape) != 3
            or in_shape[1] != 2
            or not isinstance(in_shape[2], int)
            or len(out_shape) != 4
            or not isinstance(out_shape[1], int)
            or not 0 <= vocals_index < out_shape[1]
        ):
            raise SeparatorModelError(
                "'{0}' is not a fixed length HTDemucs model (input {1}, output {2}, "
                "vocals index {3})".format(model_path, in_shape, out_shape, vocals_index)
            )
        self.model_name = os.path.basename(model_path)
        session_providers = self._session.get_providers()
        if "CUDAExecutionProvider" not in session_providers:
            logger.warning(
                "ONNX Runtime is using CPU only for %s; active providers: %s",
                self.model_name,
                session_providers,
            )
        else:
            logger.info("ONNX Runtime providers for %s: %s", self.model_name, session_providers)
        self._input_name = self._session.get_inputs()[0].name
        self._segment = in_shape[2]
        self._vocals_index = vocals_index
        self._overlap = overlap

    def _detect_vocals_index(self, out_shape) -> int:
        """ vocals channel of the model output: onnx metadata, else single stem -> 0, else 3 """
        try:
            meta = self._session.get_modelmeta().custom_metadata_map or {}
            if "vocals_index" in meta:
                return int(meta["vocals_index"])
        except Exception:  # metadata is optional, don't fail because of it
            logger.debug("can't read the onnx metadata", exc_info=True)
        if len(out_shape) == 4 and out_shape[1] == 1:
            return 0
        return 3

    def separate(
        self, wav: np.ndarray, progress: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """ wav: float array of shape (2, samples) at 44.1 kHz, returns the vocal stem (same shape).
            progress(done, total) is called after every chunk, it may raise an exception to abort """
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 2 or wav.shape[0] != 2:
            raise ValueError("expected audio of shape (2, samples) at 44.1 kHz")
        n_samples = wav.shape[1]
        if n_samples == 0:
            raise SeparationFailedError("the audio signal is empty")
        if not np.isfinite(wav).all():
            raise SeparationFailedError("the audio signal contains NaN or infinite values")
        segment = self._segment
        # the model expects normalised audio, the original level is restored afterwards
        mono = wav.mean(0)
        mean = mono.mean()
        std = mono.std(ddof=1) + 1e-8 if n_samples > 1 else 1e-8
        wav_norm = (wav - mean) / std
        stride = int((1 - self._overlap) * segment)
        # triangular window with maximum in the middle of the chunk
        weight = np.concatenate(
            [np.arange(1, segment // 2 + 1), np.arange(segment - segment // 2, 0, -1)]
        ).astype(np.float32)
        weight /= weight.max()
        out = np.zeros_like(wav_norm)
        total_weight = np.zeros(n_samples, dtype=np.float32)
        offsets = range(0, n_samples, stride)
        for idx, offset in enumerate(offsets):
            length = min(segment, n_samples - offset)
            # a short last chunk is centered in real audio context instead of zero padding
            # (like demucs.apply.TensorChunk.padded) and the output is center trimmed again
            delta = segment - length
            start = offset - delta // 2
            end = start + segment
            low, high = max(0, start), min(n_samples, end)
            chunk = np.pad(wav_norm[:, low:high], ((0, 0), (low - start, end - high)))
            try:
                stems = self._session.run(None, {self._input_name: chunk[None]})[0]
            except Exception as err:  # e.g. out of memory
                raise SeparationFailedError("inference failed: {0}".format(err)) from err
            if stems.shape[-1] != segment or stems.shape[-2] != 2:
                raise SeparationFailedError("unexpected model output shape {0}".format(stems.shape))
            vocals = stems[0, self._vocals_index, :, delta // 2 : segment - (delta - delta // 2)]
            out[:, offset : offset + length] += weight[:length] * vocals
            total_weight[offset : offset + length] += weight[:length]
            if progress is not None:
                progress(idx + 1, len(offsets))
        return (out / total_weight) * std + mean
