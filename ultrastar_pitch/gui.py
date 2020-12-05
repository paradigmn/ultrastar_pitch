#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          application_window.py
@brief         gui implementation for ultrastar-pitch
@author        paradigm
"""

import os
import tkinter as tk
from tkinter import filedialog
import logging

from .detection_pipeline import DetectionPipeline


class Gui(tk.Frame):
    """ basic graphical interface for user interaction """
    def __init__(self, detection_pipeline: DetectionPipeline, root: tk.Tk=tk.Tk(), postproc: bool=True):
        """ configure gui widgets for the main application window """
        tk.Frame.__init__(self, root)
        root.title("ultrastar_pitch")
        root.geometry('600x300')
        root.iconphoto(True, tk.PhotoImage(file=os.path.join(os.path.dirname(__file__),
                                                             "binaries", "icon.png")))
        # text labels
        path_lbl = tk.Label(root, text="notes.txt:")
        path_lbl.place(x=160, y=80)
        self.done_lbl = tk.Label(root, text="")
        self.done_lbl.place(x=440, y=100)
        # text field
        self.path_txt = tk.Entry(root)
        self.path_txt.place(width=275, x=160, y=100)
        self.path_txt.focus()
        # load and run buttons
        load_btn = tk.Button(root, text="Load", command=self.load_notes)
        load_btn.place(width=100, x=185, y=130)
        convert_btn = tk.Button(root, text="Run", command=self.predict_pitches)
        convert_btn.place(width=100, x=310, y=130)
        # postprocessing checkbutton
        self.postproc_chk_state = tk.BooleanVar()
        self.postproc_chk_state.set(True)
        self.postproc_chk = tk.Checkbutton(root, text="Enable Postprocessing",
                                           var=self.postproc_chk_state)
        self.postproc_chk.place(x=10, y=250)
        self.note_file = None

        root.mainloop()

    def load_notes(self):
        """ callback for load button, load a project notes.txt with a filedialog """
        self.done_lbl.configure(text="")
        self.note_file = filedialog.askopenfile(filetypes = (("Text files","*.txt"),
                                                             ("all files","*.*")))
        self.path_txt.delete(0, tk.END)
        self.path_txt.insert(0, self.note_file.name)
        logging.info("test")

    def predict_pitches(self):
        """ callback for run button, predicts new pitches and generate new notes.txt """
        self.done_lbl.configure(text="done!")
