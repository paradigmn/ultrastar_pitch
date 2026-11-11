#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          model_download.py
@brief         download the optional onnx models from the GitHub release "models-v1"
@author        paradigm

The files are written to the user data folder (see model_files.user_model_dir), where the
pipeline finds them without any configuration. Every file is checked against its SHA-256
checksum before it replaces an existing file; a file that is already present and correct is
not downloaded again.
"""

import hashlib
import logging
import os
import tempfile
import urllib.request
from typing import Callable, Dict, List, NamedTuple, Optional

from .model_files import RMVPE_MODEL_FILE_NAME, VOCAL_MODEL_FILE_NAME, user_model_dir
from .version import __version__

logger = logging.getLogger(__name__)

RELEASE_URL = "https://github.com/paradigmn/ultrastar_pitch/releases/download/models-v1/"


class ModelFile(NamedTuple):
    file_name: str
    sha256: str
    size: int
    description: str


MODELS: Dict[str, ModelFile] = {
    "rmvpe": ModelFile(
        RMVPE_MODEL_FILE_NAME,
        "648e1a7bf3795f39498841d35c2a100ab4997c045e7d5ed555b02aef0f3dfa32",
        185845903,
        "RMVPE pitch model (better pitch detection)",
    ),
    "vocals": ModelFile(
        VOCAL_MODEL_FILE_NAME,
        "d6f9ed0bb9e0e68b65ab3c1abe3a9cad94da43eb0a61255909742880bd44ee70",
        304385595,
        "Demucs vocal separation model (option -s)",
    ),
}


class ModelDownloadError(Exception):
    """ the download failed or the downloaded file has the wrong checksum """


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_installed(name: str, target_dir: Optional[str] = None) -> bool:
    """ True, if the model is in the target folder with the expected size and checksum """
    model = MODELS[name]
    path = os.path.join(target_dir or user_model_dir(), model.file_name)
    return os.path.isfile(path) and os.path.getsize(path) == model.size and file_sha256(path) == model.sha256


def download_model(
    name: str,
    target_dir: Optional[str] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
    base_url: str = RELEASE_URL,
    timeout: float = 60.0,
) -> str:
    """ download one model ("rmvpe" or "vocals") into target_dir (default: user data folder) and
        return its path. progress(file_name, bytes_done, bytes_total) is called while loading """
    if name not in MODELS:
        raise ValueError("unknown model {0!r}, expected one of {1}".format(name, ", ".join(MODELS)))
    model = MODELS[name]
    target_dir = target_dir or user_model_dir()
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, model.file_name)
    if is_installed(name, target_dir):
        logger.info("%s is already installed: %s", model.file_name, target)
        if progress:
            progress(model.file_name, model.size, model.size)
        return target
    url = base_url + model.file_name
    request = urllib.request.Request(url, headers={"User-Agent": "ultrastar_pitch/" + __version__})
    # load into a temporary file in the target folder, so the final rename is atomic
    fd, tmp_path = tempfile.mkstemp(prefix=model.file_name + ".", suffix=".part", dir=target_dir)
    digest = hashlib.sha256()
    done = 0
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or model.size)
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(model.file_name, done, total)
        if digest.hexdigest() != model.sha256:
            raise ModelDownloadError(
                "checksum mismatch for {0} (got {1}), the download is corrupt".format(model.file_name, digest.hexdigest())
            )
        os.replace(tmp_path, target)
    except ModelDownloadError:
        raise
    except Exception as err:
        raise ModelDownloadError("download of {0} failed: {1}".format(url, err)) from err
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    logger.info("downloaded %s (%d bytes) to %s", model.file_name, done, target)
    return target


def download_models(
    names: List[str],
    target_dir: Optional[str] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> List[str]:
    """ download several models, see download_model """
    return [download_model(name, target_dir, progress) for name in names]
