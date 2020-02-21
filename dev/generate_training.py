#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          generate_training.py
@brief         read in audio segments and generates training data from them
@author        paradigm
"""

import os
import numpy as np

from ultrastar_pitch.preprocessing import AverageFourier

# list with database root folders
INPUT_DIRS = [#"/path/to/folder/1/",
              #"/path/to/folder/2/",
              #"/path/to/folder/3/",
             ]
# directory for the training data
OUTPUT_DIR = "/path/for/output/data/"
# sample rate for audio
SR = 16000

def clear_training_data():
    """ remove previously created training data """
    for pitch in range(12):
        pitch_dir = os.path.join(OUTPUT_DIR, str(pitch).zfill(2))
        if os.path.exists(pitch_dir):
            filelist = [x for x in os.listdir(pitch_dir) if x.endswith(".npy")]
            for file in filelist:
                os.remove(os.path.join(pitch_dir, file))
            print("cleared " + pitch_dir)
    print()

def main():
    # list with audio segment files
    segment_paths = []
    # counter to name training data
    file_counter = 0

    # init data preprocessor
    trafo = AverageFourier(sample_rate=SR, fft_len=2048, fg_l=80, adv_len=1024, noise_th=0.0)

    # create subfolders if neccessary
    for pitch in range(12):
        pitch_dir = os.path.join(OUTPUT_DIR, str(pitch).zfill(2))
        os.makedirs(pitch_dir, exist_ok=True)

    # remove old data if desired
    # clear_training_data()

    # parse folder for usable files
    for input_dir in INPUT_DIRS:
        for root, dirs, files in os.walk(input_dir, topdown=False):
            for file in files:
                if '.npy' in file:
                    segment_paths.append(os.path.join(root, file))

    # iterate usdx files
    for segment_path in segment_paths:
        print(segment_path)
        # extract pitch from folder name
        pitch = int(os.path.basename(os.path.dirname(segment_path)))
        segment = np.load(segment_path)
        # emphasize audio segment
        segment_emph = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])
        # transform segment into features for prediction
        features = trafo.transform_audio_segment(segment_emph)
        # set label location for features
        feature_path = os.path.join(OUTPUT_DIR, str(pitch).zfill(2), str(file_counter))
        # do not overwrite already existing data
        if not os.path.exists(feature_path + ".npy"):
            np.save(feature_path, features)
        file_counter += 1
    print("finished!\n")


if __name__ == '__main__':
    main()
