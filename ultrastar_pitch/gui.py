import tkinter as tk
from tkinter import filedialog
import logging

class ApplicationWindow(tk.Frame):
    """ basic graphical interface for user interaction """
    def __init__(self, top, *args, **kwargs):
        """ configure gui widgets for the main application window """
        tk.Frame.__init__(self, top, *args, **kwargs)
        self.top = top
        self.top.title("ultrastar_pitch")
        self.top.geometry('600x300')
        # text labels
        self.path_lbl = tk.Label(self.top, text="notes.txt:")
        self.path_lbl.place(x=160, y=80)
        self.done_lbl = tk.Label(self.top, text="")
        self.done_lbl.place(x=440, y=100)
        # text field
        self.path_txt = tk.Entry(self.top)
        self.path_txt.place(width=275, x=160, y=100)
        self.path_txt.focus()      
        # load and run buttons
        self.load_btn = tk.Button(self.top, text="Load", command=self.load_notes)
        self.load_btn.place(width=100, x=185, y=130)
        self.convert_btn = tk.Button(self.top, text="Run", command=self.predict_pitches)
        self.convert_btn.place(width=100, x=310, y=130)
        # postprocessing checkbutton
        self.postproc_chk_state = tk.BooleanVar()
        self.postproc_chk_state.set(True)
        self.postproc_chk = tk.Checkbutton(self.top, text="Enable Postprocessing", var=self.postproc_chk_state)
        self.postproc_chk.place(x=10, y=250)
        
    def load_notes(self):
        """ callback for load button, load a project notes.txt with a filedialog """
        self.done_lbl.configure(text="")
        self.note_file = filedialog.askopenfile(filetypes = (("Text files","*.txt"),("all files","*.*")))
        self.path_txt.delete(0, tk.END)
        self.path_txt.insert(0, self.note_file.name)
        logging.info("test")
        
    def predict_pitches(self):
        """ callback for run button, predicts new pitches and generate new notes.txt """
        self.done_lbl.configure(text="done!")
