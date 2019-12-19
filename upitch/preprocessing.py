#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          preprocessing.py
@brief         parse an USDX note file and the corresponding audio file
               for further processing
@author        paradigm
"""

class NoteParser:
    """ read an USDX note file and makes it content available """
    def __init__(self):
        # dictionary with metadata
        self.__meta = {}
        # list of dictionarys with singable notes
        self.__singable = []
        # buffer for note_file content
        self.__file_buffer = []

    @property
    def meta(self):
        """ getter method for meta variable """
        return self.__meta

    @property
    def singable(self):
        """ getter method for singable variable """
        return self.__singable

    def load_note_file(self, note_file):
        """ load metadata and notelist into iterable objects for manipulation\n
        @param  note_file USDX project file
        """
        self.meta.clear()
        self.singable.clear()
        self.__file_buffer.clear()
        note_file = open(note_file, 'r', encoding="utf-8")
        # buffer file for later reuse
        self.__file_buffer = note_file.read().splitlines(True)
        for line in self.__file_buffer:
            # parse header
            if line.startswith('#'):
                # remove trailing whitespaces
                line = line.rstrip('\r').rstrip('\n')
                key, value = line.split(':', 1)
                if key in ("#BPM", "#GAP"):
                    self.meta[key] = float(value.replace(',', '.'))
                else:
                    self.meta[key] = value
            # parse singable notes
            elif line.startswith((':', '*')):
                line = line.split(' ')
                pitch = int(line[3]) % 12
                # start = gap + start_beat * (1500 / bpm)
                # I have no idea where the 15000 comes from, I found it manually by trial and error
                t_start = self.meta["#GAP"] + float(line[1]) * (15000 / self.meta["#BPM"])
                # end = gap + (start_beat + end_beat) * (1500 / bpm)
                t_end = self.meta["#GAP"] + (float(line[1]) + float(line[2])) * (15000 / self.meta["#BPM"])
                # append line data to singable list
                self.singable.append({"t_start" : t_start, "t_end" : t_end, "pitch" : pitch})

    def update_pitches(self, new_pitches):
        """ replace old singable pitches by newly calculated ones\n
        @param  new_pitches list of newly calculated pitches
        """
        assert (len(new_pitches) == len(self.singable)), "pitches can't be updated, array size doesn't match!"
        for singable, new_pitch in zip(self.singable, new_pitches):
            singable["pitch"] = new_pitch

    def save_note_file(self, note_file):
        """ save updated note file under a new name\n
        @param  note_file location+name for updated USDX project file
        """
        note_file = open(note_file, 'w+', encoding="utf-8")
        singable = iter(self.singable)
        # go through old file and update it when needed
        for line in self.__file_buffer:
            if line.startswith((':', '*')):
                line = line.split(" ")
                line[3] = str(int(next(singable)["pitch"]))
                line = ' '.join(line)
            note_file.write(line)

class AudioProcessor:
    """ convert and resample audio file before dividing it into audio segments """
    def __init__(self):
        # tbd!
        pass

    def load_audio(self, audio_file, sample_rate=16000):
        """ convert audio file into a wav array with a given sample rate\n
        @param audio_file   mp3 file which will be converted\n
        @param sample_rate  sampling frequency for the conversion
        """
        # tbd!
        pass

    def segment_audio(self, audio_data, segment_data):
        """ convert audio file into a wav array with a given sample rate\n
        @param audio_data    wav array which will be segmented\n
        @param segment_data  time stamps for audio segmentation
        """
        # tbd!
        pass
        
