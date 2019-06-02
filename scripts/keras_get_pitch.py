# example application for using ultastar_pitch

import os, sys
from upitch import PitchDetection

def main():
    # read working directory
    work_dir = os.getcwd()
    
    print("start utility!")
    # set sampling rate fft window size and lower cut of frequency
    test = PitchDetection(sample_rate=16000, fft_len=2048, fg1=140)
    # change the keras model path depending on using script or executable
    print("load model")    
    if getattr(sys,'frozen',False):
        test.load_keras_model(os.path.join(sys._MEIPASS, 'keras_tf_1025_240_120_12_fft_0.model'))
    else:
        test.load_keras_model(os.path.join('..', 'keras', 'keras_tf_1025_240_120_12_fft_0.model'))
    # start processing
    print("load project")
    test.load_project(work_dir)
    print("data was computed")
    # save project to disk
    test.save_project()
    print("new file was created: 'notes.txt.new'")
if __name__ == "__main__": main()
