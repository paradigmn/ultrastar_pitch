# -*- mode: python -*-

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

MODULE = 'ultrastar-pitch-runner.py'
FFMPEG_BIN = 'ffmpeg.exe'
FFMPEG_DIR = 'ffmpeg\\bin\\'
MODEL_NAME = 'tf2_256_96_12_stft_pca_median.model'
MODEL_PATH = 'ultrastar_pitch\\binaries\\tf2_256_96_12_stft_pca_median.model\\'
MARKOV_TRANS = 'makov_trans_cube.npy'
PCA_COMP = 'pca_components.npy'
PCA_MEAN = 'pca_mean.npy'
BIN_DIR = 'ultrastar_pitch\\binaries\\'
block_cipher = None

a = Analysis([MODULE],
             pathex=[os.getcwd()],
             binaries=[],
             datas = collect_data_files('tensorflow_core', subdir=None, include_py_files=True) + collect_data_files('astor', subdir=None, include_py_files=True),
             hiddenimports = collect_submodules('tensorflow_core'),
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries + [(FFMPEG_BIN, os.path.join(FFMPEG_DIR, FFMPEG_BIN), 'BINARY')] +
                       [(MARKOV_TRANS, os.path.join(BIN_DIR, MARKOV_TRANS), 'BINARY')] +
                       [(PCA_COMP, os.path.join(BIN_DIR, PCA_COMP), 'BINARY')] +
                       [(PCA_MEAN, os.path.join(BIN_DIR, PCA_MEAN), 'BINARY')],
          a.zipfiles,
          a.datas + Tree(MODEL_PATH, prefix=MODEL_NAME),
          [],
          name='ultrastar_pitch',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True)
