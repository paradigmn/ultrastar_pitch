#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          gui.py
@brief         gui implementation for ultrastar-pitch
@author        paradigm
"""

import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from subprocess import CalledProcessError
import logging

from .detection_pipeline import DetectionPipeline
from .vocal_separator import default_model_path
from .rmvpe import default_model_path as default_rmvpe_path


class Gui(tk.Frame):
    """ basic graphical interface for user interaction """

    def __init__(self, detection_pipeline: DetectionPipeline, root: tk.Tk = None):
        """ configure gui widgets for the main application window """
        root = root or tk.Tk()
        tk.Frame.__init__(self, root)
        self.root = root
        self.detection_pipeline = detection_pipeline
        # messages from the worker thread (progress text or the final result)
        self._events = queue.Queue()
        self._worker = None
        # build window
        root.title("ultrastar_pitch")
        root.geometry("600x330")
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
        path_lbl.place(x=160, y=60)
        self.done_lbl = tk.Label(root, text="")
        self.done_lbl.place(x=160, y=150)
        # text field
        self.path_txt = tk.Entry(root)
        self.path_txt.insert(0, "notes.txt")
        self.path_txt.bind("<Return>", (lambda event: self.predict_pitches()))
        self.path_txt.place(width=275, x=160, y=80)
        self.path_txt.focus()
        # load and run buttons
        load_btn = tk.Button(root, text="Load", command=self.load_notes)
        load_btn.place(width=100, x=185, y=110)
        self.run_btn = tk.Button(root, text="Run", command=self.predict_pitches)
        self.run_btn.place(width=100, x=310, y=110)
        # postprocessing checkbutton
        self.postproc_chk_state = tk.BooleanVar()
        self.postproc_chk_state.set(True)
        postproc_chk = tk.Checkbutton(
            root, text="Enable Postprocessing", var=self.postproc_chk_state
        )
        postproc_chk.place(x=10, y=200)
        # vocal separation: checkbutton and model path
        self.separate_chk_state = tk.BooleanVar()
        self.separate_chk_state.set(False)
        separate_chk = tk.Checkbutton(
            root, text="Separate vocals first (slow, result is saved as vocals.wav)",
            var=self.separate_chk_state,
        )
        separate_chk.place(x=10, y=225)
        model_lbl = tk.Label(root, text="Separation model:")
        model_lbl.place(x=10, y=255)
        self.model_txt = tk.Entry(root)
        self.model_txt.insert(0, detection_pipeline.vocal_model or default_model_path() or "")
        self.model_txt.place(width=350, x=130, y=255)
        model_btn = tk.Button(root, text="...", command=self.load_model)
        model_btn.place(width=30, x=485, y=253)
        # RMVPE pitch model (optional, combined with the built-in classifier)
        rmvpe_lbl = tk.Label(root, text="RMVPE model:")
        rmvpe_lbl.place(x=10, y=285)
        self.rmvpe_txt = tk.Entry(root)
        self.rmvpe_txt.insert(0, detection_pipeline.rmvpe_model or default_rmvpe_path() or "")
        self.rmvpe_txt.place(width=350, x=130, y=285)
        rmvpe_btn = tk.Button(root, text="...", command=self.load_rmvpe)
        rmvpe_btn.place(width=30, x=485, y=283)
        root.mainloop()

    def load_notes(self):
        """ callback for load button, load a project notes.txt with a filedialog """
        self.done_lbl.configure(text="")
        note_file = filedialog.askopenfile(
            filetypes=(("Text files", "*.txt"), ("all files", "*.*"))
        )
        if note_file is None:
            return
        self.path_txt.delete(0, tk.END)
        self.path_txt.insert(0, note_file.name)

    def load_model(self):
        """ callback for the model button, choose the vocal separation model """
        model_file = filedialog.askopenfile(
            filetypes=(("ONNX models", "*.onnx"), ("all files", "*.*"))
        )
        if model_file is None:
            return
        self.model_txt.delete(0, tk.END)
        self.model_txt.insert(0, model_file.name)

    def load_rmvpe(self):
        """ callback for the RMVPE button, choose the pitch model """
        model_file = filedialog.askopenfile(
            filetypes=(("ONNX models", "*.onnx"), ("all files", "*.*"))
        )
        if model_file is None:
            return
        self.rmvpe_txt.delete(0, tk.END)
        self.rmvpe_txt.insert(0, model_file.name)

    def predict_pitches(self):
        """ callback for run button, predicts new pitches in a worker thread (the separation
            takes minutes) and generates a new notes.txt """
        if self._worker is not None and self._worker.is_alive():
            return
        notes_org = self.path_txt.get()
        notes_new = os.path.join(os.path.dirname(notes_org), "notes_new.txt")
        postproc = self.postproc_chk_state.get()
        separate = self.separate_chk_state.get()
        self.detection_pipeline.vocal_model = self.model_txt.get().strip() or None
        self.detection_pipeline.rmvpe_model = self.rmvpe_txt.get().strip() or None
        logging.debug("predicting file: %s", notes_org)
        logging.debug("postprocessing: %s, separate vocals: %s", postproc, separate)
        self.done_lbl.configure(text="working ...")
        self.run_btn.configure(state=tk.DISABLED)

        def progress(done: int, total: int):
            self._events.put(("progress", "separating vocals: chunk {0}/{1}".format(done, total)))

        def work():
            try:
                self.detection_pipeline.transform(
                    notes_org, notes_new, postproc, separate_vocals=separate, progress=progress
                )
                self._events.put(("done", None))
            except Exception as err:  # reported in the gui thread
                self._events.put(("error", err))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self.root.after(100, self._poll_events)

    def _poll_events(self):
        """ show the messages of the worker thread (tk is not thread safe) """
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "progress":
                    self.done_lbl.configure(text=payload)
                elif kind == "done":
                    self.done_lbl.configure(text="done! ({0}, {1})".format(
                        self.detection_pipeline.last_models, self.detection_pipeline.last_sources))
                    self.run_btn.configure(state=tk.NORMAL)
                    logging.info("predicted pitches successfully!")
                    return
                else:
                    self.done_lbl.configure(text="failed!")
                    self.run_btn.configure(state=tk.NORMAL)
                    self._show_error(payload)
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    @staticmethod
    def _show_error(err: Exception):
        if isinstance(err, KeyError):
            logging.error("input file error: %s", err)
            messagebox.showerror("error", "invalid input file!\n" + str(err))
        elif isinstance(err, CalledProcessError):
            logging.error("audio file error: %s", err)
            messagebox.showerror("error", "invalid audio file!\n" + str(err))
        else:
            logging.error("error: %s", err)
            messagebox.showerror("error", str(err))
