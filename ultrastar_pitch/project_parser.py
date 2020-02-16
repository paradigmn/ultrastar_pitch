#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          project_parser.py
@brief         parse an USDX note file and the corresponding audio file
               for further processing
@author        paradigm
"""
import os
import sys
import subprocess
from scipy.io import wavfile

class ProjectParser:
    """ parse an USDX note file and the corresponding audio file  """
    def __init__(self):
        # dictionary with metadata
        self.__meta = {}
        # list of dictionarys with singable notes
        self.__singable = []
        # buffer for note_file content
        self.__file_buffer = []
        # derived project path from note file
        self.__proj_dir = 0
        # change the ffmpeg path depending on using script or executable
        if getattr(sys, 'frozen', False):
            self.__FFMPEG = os.path.join(sys._MEIPASS, 'ffmpeg.exe')
        else:
            self.__FFMPEG = 'ffmpeg'
            
    @property
    def meta(self):
        """ getter for metadata """
        return self.__meta
    
    @meta.setter
    def meta(self, meta):
        """ setter for metadata """
        self.__meta = meta
        
    def load_note_file(self, note_file):
        """ load metadata and notelist into iterable objects for manipulation\n
        @param  note_file USDX project file
        """
        self.__meta.clear()
        self.__singable.clear()
        self.__file_buffer.clear()
        self.__proj_dir = os.path.dirname(note_file)
        note_file = open(note_file, 'r', encoding="utf-8")
        # buffer file for later reuse
        self.__file_buffer = note_file.read().splitlines(True)
        for line in self.__file_buffer:
            # parse header
            if line.startswith('#') and not self.__singable:
                # remove trailing whitespaces
                line = line.rstrip('\r').rstrip('\n')
                key, value = line.split(':', 1)
                if key in ("#BPM", "#GAP"):
                    self.__meta[key] = float(value.replace(',', '.'))
                else:
                    self.__meta[key] = value
            # parse singable notes
            elif line.startswith((':', '*')):
                line = line.split(' ')
                pitch = int(line[3]) % 12
                # start = gap + start_beat * (15000 / bpm)
                # the 15000 indicates an 1/16 note, while a beat is usually a 1/4 note
                t_start = self.__meta["#GAP"] + float(line[1]) * (15000 / self.__meta["#BPM"])
                # end = gap + (start_beat + end_beat) * (15000 / bpm)
                t_end = self.__meta["#GAP"] + (float(line[1]) + \
                                               float(line[2])) * (15000 / self.__meta["#BPM"])
                # append line data to singable list
                self.__singable.append({"t_start" : t_start, "t_end" : t_end, "pitch" : pitch})

    def dump_pitches(self):
        """ returns the singable pitches\n
        @return  array of singable pitches
        """
        return [singable["pitch"] for singable in self.__singable]
        
    def update_pitches(self, new_pitches):
        """ replace old singable pitches by newly calculated ones\n
        @param  new_pitches list of newly calculated pitches
        """
        assert (len(new_pitches) == len(self.__singable)), \
                "pitches can't be updated, array size doesn't match!"
        for singable, new_pitch in zip(self.__singable, new_pitches):
            singable["pitch"] = new_pitch

    def save_note_file(self, note_file):
        """ save updated note file under a new name\n
        @param  note_file location+name for updated USDX project file
        """
        note_file = open(note_file, 'w+', encoding="utf-8")
        singable = iter(self.__singable)
        # go through old file and update it when needed
        for line in self.__file_buffer:
            if line.startswith((':', '*')):
                line = line.split(" ")
                line[3] = str(int(next(singable)["pitch"]))
                line = ' '.join(line)
            note_file.write(line)

    def process_audio(self, sample_rate=16000):
        """ convert and resample audio file before dividing it into audio segments\n
        @param sample_rate  sampling frequency for the conversion
        @return  list of audio segments
        """
        err_str1 = "{0} has to be an positive integer value"
        assert (isinstance(sample_rate, int) and sample_rate > 0), err_str1.format("sample_rate")
        # convert mp3 to temporary mono wav file
        audio_path = os.path.join(self.__proj_dir, self.__meta["#MP3"])
        wav_path = os.path.join(self.__proj_dir, "tmp.wav")
        subprocess.run([self.__FFMPEG, '-i', audio_path, '-y',
                        '-ac', '1', '-ar', str(sample_rate), wav_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # load wav into numpy array for processing and discard file
        _, samples_mono = wavfile.read(wav_path)
        os.remove(wav_path)
        audio_segments = []
        for segment in self.__singable:
            start_sample = int(round((segment["t_start"] * sample_rate) / 1000))
            end_sample = int(round((segment["t_end"] * sample_rate) / 1000))
            audio_segments.append(samples_mono[start_sample:end_sample])
        return audio_segments
