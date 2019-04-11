# example application for using ultastar_pitch

from ultrastar_pitch import UltraStarPitch as upitch
import os

test_dir = "/home/ruben/shared/workspace/usdx_pitch/in/Velvet R. Wings - Cutie-H Equestria/"

def main():
    #work_dir = os.getcwd()
    work_dir = test_dir
    
    print("start utility!")
    test = upitch(sample_rate=16000, fft_len=2048, fg1=140)
    test.load_project(work_dir)
    print("data was computed")
    test.save_project()
    print("new file was created: 'notes.txt.new'")
if __name__ == "__main__": main()