 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          setup.py
@brief         project build script
@author        paradigm
"""

import os
from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = {}
with open(os.path.join(os.path.dirname(__file__), "ultrastar_pitch", "version.py")) as fp:
    exec(fp.read(), version)

setup(
    name="ultrastar_pitch",
    version=version["__version__"],
    description="Automate the pitch detection process for Ultrastar Deluxe karaoke projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: End Users/Desktop",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: End Users/Desktop",
    ],
    url="https://github.com/paradigmn/ultrastar_pitch",
    author="paradigm",
    author_email="ultrastarpitch@gmail.com",
    packages=["ultrastar_pitch"],
    entry_points={
        "console_scripts": [
            "ultrastar-pitch = ultrastar_pitch.ultrastar_pitch:main"
        ]
    },
    install_requires=["numpy", "scipy", "onnxruntime"],
    include_package_data=True,
    zip_safe=False,
)
