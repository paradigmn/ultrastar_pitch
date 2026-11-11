#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          model_files.py
@brief         locate the optional (large) onnx models, which are not shipped with the package
@author        paradigm
"""

import logging
import os
import sys
from typing import List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# environment variables and file names of the optional models
VOCAL_MODEL_ENV_VAR = "ULTRASTAR_PITCH_VOCAL_MODEL"
# export of the Demucs vocals model with the vocals stem only (dev/export_demucs_vocals.py --vocals-only)
VOCAL_MODEL_FILE_NAME = "htdemucs_vocals.onnx"
RMVPE_MODEL_ENV_VAR = "ULTRASTAR_PITCH_RMVPE_MODEL"
RMVPE_MODEL_FILE_NAME = "rmvpe.onnx"


def user_model_dir() -> str:
    """ folder for downloaded models: %LOCALAPPDATA%/ultrastar_pitch or ~/.local/share/ultrastar_pitch """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "ultrastar_pitch")


def model_candidates(file_name: str) -> List[str]:
    """ places a model file is looked for, in this order: folder of the executable (pyinstaller),
        user data folder, models/ folder next to the source tree (development) """
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(os.path.dirname(sys.executable), file_name))
    candidates.append(os.path.join(user_model_dir(), file_name))
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", file_name))
    return candidates


def find_model(file_names: Union[str, Sequence[str]], env_var: str) -> Optional[str]:
    """ path of the model or None: the environment variable wins, then the candidate folders
        (all file names in the first folder, then the next folder) """
    from_env = os.environ.get(env_var)
    if from_env:
        if os.path.isfile(from_env):
            return from_env
        logger.warning("%s points to a missing file: %s", env_var, from_env)
    names = [file_names] if isinstance(file_names, str) else list(file_names)
    per_name = [model_candidates(name) for name in names]
    for folder_idx in range(len(per_name[0])):
        for candidates in per_name:
            if os.path.isfile(candidates[folder_idx]):
                return candidates[folder_idx]
    return None
