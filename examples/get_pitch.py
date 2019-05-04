# example application for using ultastar_pitch

import os
from upitch import PitchDetection

def main():
    # read working directory
    work_dir = os.getcwd()
    
    print("start utility!")
    # set sampling rate fft window size and lower cut of frequency
    test = PitchDetection(sample_rate=16000, fft_len=2048, fg1=140)
    # start processing
    test.load_project(work_dir)
    print("data was computed")
    # save project to disk
    test.save_project()
    print("new file was created: 'notes.txt.new'")
if __name__ == "__main__": main()