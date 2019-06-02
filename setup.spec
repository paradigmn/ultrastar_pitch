# -*- mode: python -*-

import os

MODULE = 'examples\\get_pitch.py'
FFMPEG_BIN = 'ffmpeg.exe'
FFMPEG_DIR = 'C:\\ffmpeg\\bin\\'
KERAS_MODEL = 'keras_tf_1025_240_120_12_fft_0.model'
KERAS_DIR = 'keras\\'
block_cipher = None


a = Analysis([MODULE],
             pathex=[os.getcwd()],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

# custom code to move _pywrap_tensorflow_internal.pyd to tensorflow/python/_pywrap_tensorflow_internal.pyd			 
for i in range(len(a.binaries)):
	dest, origin, kind = a.binaries[i]
	if '_pywrap_tensorflow_internal' in dest:
		a.binaries[i] = ('tensorflow.python.' + dest, origin, kind)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries + [(FFMPEG_BIN, os.path.join(FFMPEG_DIR, FFMPEG_BIN), 'BINARY')] + [(KERAS_MODEL, os.path.join(KERAS_DIR, KERAS_MODEL), 'BINARY')],
          a.zipfiles,
          a.datas + [(KERAS_MODEL, 'KERAS')],
          [],
          name='ultrastar_pitch',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True)
