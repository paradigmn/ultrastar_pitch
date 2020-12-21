#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          application_window.py
@brief         gui implementation for ultrastar-pitch
@author        paradigm
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from subprocess import CalledProcessError
import logging

from .detection_pipeline import DetectionPipeline


class Gui(tk.Frame):
    """ basic graphical interface for user interaction """

    def __init__(self, detection_pipeline: DetectionPipeline, root: tk.Tk = tk.Tk()):
        """ configure gui widgets for the main application window """
        tk.Frame.__init__(self, root)
        self.detection_pipeline = detection_pipeline
        # build window
        root.title("ultrastar_pitch")
        root.geometry("600x300")
        # set icon depending on os and execution method
        if getattr(sys, "frozen", False):
            icon_path = sys._MEIPASS
        else:
            icon_path = os.path.join(os.path.dirname(__file__), "binaries")
        if sys.platform == "win32":
            root.iconbitmap(os.path.join(icon_path, "icon.ico"))
        else:
            root.iconphoto(True, tk.PhotoImage(file=os.path.join(icon_path, "icon.png")))
        # text labels
        path_lbl = tk.Label(root, text="notes.txt:")
        path_lbl.place(x=160, y=80)
        self.done_lbl = tk.Label(root, text="")
        self.done_lbl.place(x=440, y=100)
        # text field
        self.path_txt = tk.Entry(root)
        self.path_txt.insert(0, "notes.txt")
        self.path_txt.bind("<Return>", (lambda event: self.predict_pitches()))
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
        postproc_chk = tk.Checkbutton(
            root, text="Enable Postprocessing", var=self.postproc_chk_state
        )
        postproc_chk.place(x=10, y=250)
        root.mainloop()

    def load_notes(self):
        """ callback for load button, load a project notes.txt with a filedialog """
        self.done_lbl.configure(text="")
        note_file = filedialog.askopenfile(
            filetypes=(("Text files", "*.txt"), ("all files", "*.*"))
        ).name
        self.path_txt.delete(0, tk.END)
        self.path_txt.insert(0, note_file)

    def predict_pitches(self):
        """ callback for run button, predicts new pitches and generate new notes.txt """
        status = "failed!"
        try:
            notes_org = self.path_txt.get()
            notes_new = os.path.join(os.path.dirname(notes_org), "notes_new.txt")
            logging.debug("predicting file: %s", notes_org)
            logging.debug("postprocessing: %s", self.postproc_chk_state.get())
            self.detection_pipeline.transform(notes_org, notes_new, self.postproc_chk_state.get())
            self.done_lbl.configure(text="done!")
            status = "done!"
            logging.info("predicted pitches successfully!")
        except KeyError as e:
            logging.exception("input file error!\n")
            messagebox.showerror("error", "invalid input file!\n" + str(e))
        except CalledProcessError as e:
            logging.exception("audio file error!\n")
            messagebox.showerror("error", "invalid audio file!\n" + str(e))
        except Exception as e:
            logging.exception("undefined error!\n")
            messagebox.showerror("error", e)
        self.done_lbl.configure(text=status)
