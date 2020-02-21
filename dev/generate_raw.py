"""
@file          generate_raw.py
@brief         read in a karaoke database and generate raw audio segment data
@author        paradigm
"""

import os
import numpy as np

from ultrastar_pitch.project_parser import ProjectParser

# list with database root folders
INPUT_DIRS = [#"/path/to/folder/1/",
              #"/path/to/folder/2/",
              #"/path/to/folder/3/",
             ]
# directory for the training data
OUTPUT_DIR = "/path/for/output/data/"
# name of the project files
USDX_FILE = "notes.txt"
# sample rate for audio
SR = 16000

def clear_training_data():
    """ remove previously created training data """
    for pitch in range(12):
        pitch_dir = os.path.join(OUTPUT_DIR, str(pitch).zfill(2))
        if os.path.exists(pitch_dir):
            filelist = [x for x in os.listdir(pitch_dir) if x.endswith(".npy")]
            for file in filelist:
                os.remove(os.path.join(pitch_dir, file))
            print("cleared " + pitch_dir)
    print()
        
def main():
    # list with project files
    projects = []
    # counter to name the training data
    file_counter = 0
    
    # init project parser
    notes = ProjectParser()
    
    # create subfolders if neccessary
    for pitch in range(12):
        pitch_dir = os.path.join(OUTPUT_DIR, str(pitch).zfill(2))
        os.makedirs(pitch_dir, exist_ok=True)
        
    # remove old data
    clear_training_data()
        
    # parse folder for usable files
    for input_dir in INPUT_DIRS:
        for root, dirs, files in os.walk(input_dir, topdown=False):
            if USDX_FILE in files:
                projects.append(root)
                
    # iterate usdx files
    for project in projects:
        print(project)
        try:
            proj_file = os.path.join(project, USDX_FILE)
            # load and parse project file
            notes.load_note_file(proj_file)
            # divide audio into pitch segments
            audio_segments = notes.process_audio(sample_rate=SR)
            # analyse each segment
            for segment, pitch in zip(audio_segments, notes.dump_pitches()):
                # set label location for segment
                segment_path = os.path.join(OUTPUT_DIR, str(pitch).zfill(2), str(file_counter))
                # do not overwrite already existing data
                if not os.path.exists(segment_path + ".npy"):
                    np.save(segment_path, segment)     
                file_counter += 1
        except UnicodeDecodeError:
            print("wrong encoding!\n")
        except FileNotFoundError:
            print("file not found!\n")
        except ValueError:
            print("could not convert string to float!\n")
        except KeyError:
            print("incorrect note.txt file!\n")
        except:
            print("Unexpected error!")
    print("finished!\n")

if __name__ == '__main__':
    main()
