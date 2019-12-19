#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          note_parser.py
@brief         read an USDX note file and yield metadata and singable notes
@author        paradigm
@date          11-Dec-2019 21:00:00
@version       11.12.2019, rp
               @li creation
"""

class NoteParser:
    """ read an USDX note file and makes it content available """
    def __init__(self):
        """ define instance variables """
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
        """ load metadata and notelist into iterable objects for manipulation """
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
        """ replace old singable pitches by newly calculated ones """
        assert (len(new_pitches) == len(self.singable)), "pitches can't be updated, array size doesn't match!"
        for singable, new_pitch in zip(self.singable, new_pitches):
            singable["pitch"] = new_pitch

    def save_note_file(self, note_file):
        """ save updated note file under a new name """
        note_file = open(note_file, 'w+', encoding="utf-8")
        singable = iter(self.singable)
        # go through old file and update it when needed
        for line in self.__file_buffer:
            if line.startswith((':', '*')):
                line = line.split(" ")
                line[3] = str(int(next(singable)["pitch"]))
                line = ' '.join(line)
            note_file.write(line)
            