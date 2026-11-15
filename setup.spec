#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          setup.spec
@brief         specification file for pyinstaller
@author        paradigm
"""

import os

MODULE = "ultrastar_pitch_runner.py"
FFMPEG_BIN = "ffmpeg.exe"
FFMPEG_DIR = os.path.join("ffmpeg", "bin")
MODEL_ONNX = "pitchnet_2026_11_13.onnx"
CONTEXT_ONNX = "context_gru_ensemble_2026_11_13.onnx"
ICON_BIN = "icon.ico"
BIN_DIR = os.path.join("ultrastar_pitch", "binaries")

a = Analysis(
    [MODULE],
    pathex=[os.getcwd()],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries
    + [(FFMPEG_BIN, os.path.join(FFMPEG_DIR, FFMPEG_BIN), "BINARY")]
    + [(MODEL_ONNX, os.path.join(BIN_DIR, MODEL_ONNX), "BINARY")]
    + [(CONTEXT_ONNX, os.path.join(BIN_DIR, CONTEXT_ONNX), "BINARY")]
    + [(ICON_BIN, os.path.join(BIN_DIR, ICON_BIN), "BINARY")],
    a.zipfiles,
    a.datas,
    [],
    name="ultrastar_pitch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    icon=os.path.join(BIN_DIR, ICON_BIN),
)
