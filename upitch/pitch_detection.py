# this file defines a class used to automatically calculate the pitch for a given USDX project

import os, sys, subprocess
import numpy as np
import scipy.io.wavfile

class PitchDetection(object):
    # pitch map for conversion
    pitch_map = {0 : "C_", 1 : "C#", 2 : "D_", 3 : "D#", 4 : "E_", 5 : "F_", 6 : "F#", 7 : "G_", 8 : "G#", 9 : "A_", 10 : "A#", 11 : "B_"}
    # mp3 file name
    __usdx_song = "song.mp3"
    # pitch file name
    __usdx_file = "notes.txt"
    # keras model name
    __keras_model = "keras_tf_1025_240_120_12_fft_0.model"
    
    def __init__(self, sample_rate=16000, fft_len=2048, fg1=140, method="keras"):
        # sample rate for conversion
        self.__sample_rate = sample_rate
        # length of fft block
        self.__fft_len = fft_len
        # lowest identifiable frequency 
        self.__fg1 = fg1
        # highest identifiable frequency 
        self.__fg2 = 880
        # frequency array
        self.__f = np.linspace(0, sample_rate // 2, num = fft_len // 2 + 1, endpoint = True)
        # fft window function
        self.__fft_win = np.hanning(fft_len)
        # data template for dictionary array
        self.__usdx_dict = {"start_time" : 0, "end_time" : 0, "org_pitch" : 0, "calc_pitch" : 0, "avg_fft": []}
        # data array: list of usdx_dicts for every pitch
        self.__usdx_data = []
        # counter for naming training data
        self.__file_counter = 0     
        # change the ffmpeg path depending on using script or executable    
        if getattr(sys,'frozen',False):
            self.__FFMPEG = os.path.join(sys._MEIPASS, 'ffmpeg.exe')
        else:
            self.__FFMPEG = 'ffmpeg'
        # load default model if deep learning analysis is demanded
        self.__model = None
        if method == "keras":
            from keras.models import load_model
            if getattr(sys,'frozen',False):
                self.__model = load_model(os.path.join(sys._MEIPASS, self.__keras_model))
            else:
                keras_path = os.path.join(os.path.dirname(__file__), os.path.pardir, "keras")
                self.__model = load_model(os.path.join(keras_path, self.__keras_model))    
        
    @classmethod
    # return the pitch corresponding to a given frequency in different formats
    def get_pitch(cls, freq, form="short"):
        if freq < 16.35:
            raise ValueError
        if form == "octave":
            # calculate number of half steps from C0
            h = round(12*np.log2(freq/(440*2**(-4.75))))       
            # return numeric octave (0, 1, 2, ...)                                       
            return(int(h // 12))
        elif form == "ascii":
            h = round(12*np.log2(freq/(440*2**(-4.75)))) 
            # determine pitch 0 -> C, 11 -> B ...       
            pitch = h % 12 
            # return pitch notation (C, Cis, D, ...)
            return(cls.pitch_map[int(pitch)])
        elif form == "numeric":
            # return long numeric pitch
            return(int(round(12*np.log2(freq/(440*2**(-4.75))))))
        else:
            # return short numeric pitch (0-11)
            return(int(round(12*np.log2(freq/(440*2**(-4.75)))) % 12)) 
        
    @classmethod    
    # pads signal array with white spaces
    def zero_pad_array(cls, array, new_size):
        return np.pad(array, (0, (new_size - len(array))), 'constant') 
              
    # return number of averaging steps
    def __get_avg_steps(self, sample_len):
        if (sample_len <= self.__fft_len):
            return(int(1))
        else:
            return(int(sample_len // (self.__fft_len / 4) - 2))  
        
    # create an averaged fft of pitch
    def __avg_fft(self, pitch_samples):
        # average samples
        steps = self.__get_avg_steps(len(pitch_samples))
        avg_fft = 0
        for i in range(steps):
            # process unpadded samples
            if (i != steps - 1):
                time_frame = pitch_samples[(i * self.__fft_len) // 4:(i + 4) * self.__fft_len // 4] * self.__fft_win
            # process padded samples
            else:
                end_len = len(pitch_samples[(i * self.__fft_len) // 4:-1])
                time_frame = pitch_samples[(i * self.__fft_len) // 4:-1] * np.hanning(end_len)
                time_frame = self.zero_pad_array(time_frame, self.__fft_len)       
            # fourier transform, averaging and normalizing
            tmp = abs(np.fft.fft(time_frame) / self.__fft_len)
            fft_spectrum = tmp[0:self.__fft_len // 2 + 1]
            fft_spectrum[1:-1] *= 2
            avg_fft += fft_spectrum
        avg_fft = avg_fft / max(avg_fft)
        if len(avg_fft) != len(self.__f) or max(avg_fft) == 0:
            raise ValueError
        return(avg_fft)
    
    # calculates the pitch frequency from a fft
    def __calc_fft_to_pitch(self, fft):       
        idx, weight_max = 0, 0
        # iterate between lowest and highest achievable frequency
        for i in range((self.__fg1 * self.__fft_len) // self.__sample_rate, (self.__fg2 * self.__fft_len) // self.__sample_rate):
            # accumulate the base tone and the first two harmonics
            weight = fft[i];
            weight += + 1/3 * sum(fft[2 * i - 1:2 * i + 1])
            weight += + 1/5 * sum(fft[3 * i - 2:3 * i + 2])
            # catch the peak value
            if (weight > weight_max):
                idx, weight_max = i, weight
        return(self.get_pitch(self.__f[idx], form="short"))  
        
    # use keras model to turn fft to pitch
    def __keras_fft_to_pitch(self, fft): 
        # check dimensions for prediction
        if (fft.ndim == 1):
            fft = np.array([fft])
        # predict pitch and return node index with accuracy
        preds = self.__model.predict(fft)
        pitch = preds.argmax(axis=1)[0]
        return(pitch, preds[0][pitch])
    
    # load audio into mono numpy array
    def __load_samples(self):
        # convert mp3 to temporary mono wav file
        subprocess.run([self.__FFMPEG, '-i', os.path.join(self.__proj_dir, self.__usdx_song), '-y', '-ac', '1', '-ar', 
                        str(self.__sample_rate), os.path.join(self.__proj_dir, "tmp.wav")], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # load wav into numpy array for processing and discard file
        _, samples = scipy.io.wavfile.read(os.path.join(self.__proj_dir, "tmp.wav"))
        os.remove(os.path.join(self.__proj_dir, "tmp.wav"))
        self.__samples_mono = samples
    
    # analyse usdx file    
    def __parse_data(self):
        bpm, gap = 0, 0
        file = open(os.path.join(self.__proj_dir, self.__usdx_file), "r", encoding='utf-8')
        for line in file.readlines():
            # parse header
            if line.startswith("#"):
                if line.startswith("#BPM:"):
                    bpm = float(line.split(":")[1].replace(",", "."))
                elif line.startswith("#GAP:"):
                    gap = float(line.split(":")[1].replace(",", "."))
            # parse song data
            elif line.startswith(":") or line.startswith("*") and bpm and gap:
                tmp = line.split(" ")
                # read and calculate timing from file
                start_beat, end_beat, self.__usdx_dict["org_pitch"] = float(tmp[1]), float(tmp[2]), int(tmp[3]) % 12
                self.__usdx_dict["start_time"] = gap + start_beat * (15000 / bpm)
                self.__usdx_dict["end_time"] = gap + (start_beat + end_beat) * (15000 / bpm)
                start_sample = int(round((self.__usdx_dict["start_time"] * self.__sample_rate) / 1000))
                end_sample = int(round((self.__usdx_dict["end_time"] * self.__sample_rate) / 1000))
                # get samples for pitch analysis
                pitch_samples = self.__samples_mono[start_sample:end_sample]
                # analyze pitch and store data
                self.__usdx_dict["calc_pitch"], self.__usdx_dict["avg_fft"] = self.analyse_audio(pitch_samples)
                self.__usdx_data.append(self.__usdx_dict)
                self.__usdx_dict = self.__usdx_dict.fromkeys(self.__usdx_dict, 0)
          
    # load custom keras model for pitch detection      
    def load_keras_model(self, model_path):
        if self.__model == None:
            from keras.models import load_model
        self.__model = load_model(model_path)
    
    # turn a given fft into the corresponding pitch
    def fft_to_pitch(self, fft):
        if len(fft) != (self.__fft_len // 2 + 1):
            raise ValueError
        if (self.__model != None):
            pitch, _ = self.__keras_fft_to_pitch(fft)
        else:
            pitch = self.__calc_fft_to_pitch(fft) 
        return pitch
    
    # take an array of audio samples and return the corresponding pitch and fft
    def analyse_audio(self, audio_samples):
        fft = self.__avg_fft(audio_samples)
        pitch = self.fft_to_pitch(fft)
        return pitch, fft    
            
    
    # helper function to yield and display statistical data        
    def get_statistics(self):
        # create a list with for true and predicted pitches
        y_true = [u_dict["org_pitch"] for u_dict in self.__usdx_data]
        y_pred = [u_dict["calc_pitch"] for u_dict in self.__usdx_data]
        # calculate correctly predicted pitches
        matches = 0
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == y_p:
                matches += 1      
        # print statistical data
        print(str(len(y_true)) + " samples")
        if (len(y_true)):
            print(str(matches / len(y_true) * 100) + "% accuracy\n")
        else:
            print()          
        # return sample vectors
        return y_true, y_pred
               
    # project init    
    def load_project(self, proj_dir):
        # clear lists to avoid problems on reuse
        self.__usdx_data.clear()
        self.__usdx_dict = self.__usdx_dict.fromkeys(self.__usdx_dict, 0)
        # make directory public
        self.__proj_dir = proj_dir
        # extract sound samples and convert to mono
        self.__load_samples()
        # create pitch list from project file
        self.__parse_data()
        
    # create a new file to save the project
    def save_project(self):
        file_new = open(os.path.join(self.__proj_dir, self.__usdx_file + ".new"), "w+")
        file_org = open(os.path.join(self.__proj_dir, self.__usdx_file), "r")
        dict_it = iter(self.__usdx_data)
        # split line and replace the pitch value
        for line in file_org.readlines():
            if line.startswith(":") or line.startswith("*"):
                line_vals = line.split(" ")
                line_vals[3] = str(next(dict_it)["calc_pitch"])
                line = ' '.join(line_vals)  
            file_new.write(line)            

    # return list of pitch data (for debugging)
    def get_pitch_data(self):
        return(self.__usdx_data)
    
    # creates training data for deep learning
    def build_training_data(self, data_dir, mode="original"):
        # print informative statistics
        self.get_statistics()
        # create subfolder
        for pitch in self.pitch_map:
            pitch_dir = os.path.join(data_dir, str(pitch).zfill(2))
            os.makedirs(pitch_dir, exist_ok=True)
        # create a csv file for each analysed pitch   
        for u_dict in self.__usdx_data:
            # only write data, if calculated and original pitch match
            if u_dict["org_pitch"] != u_dict["calc_pitch"] and mode == "filtered":
                continue
            # create data from calculated pitch
            if mode == "calc":
                folder = str(u_dict["calc_pitch"])
            # create data form original pitch
            else:
                folder = str(u_dict["org_pitch"])
                
            data_path = os.path.join(data_dir, folder.zfill(2), str(self.__file_counter))
            # do not overwrite already existing data
            if not os.path.exists(data_path + ".npy"):
                np.save(data_path, u_dict["avg_fft"])     
            self.__file_counter += 1
 
    # remove previously created training data
    def clear_training_data(self, data_dir):
        for pitch in self.pitch_map:
            pitch_dir = os.path.join(data_dir, str(pitch).zfill(2))
            if os.path.exists(pitch_dir):
                filelist = [x for x in os.listdir(pitch_dir) if x.endswith(".npy")]
                for file in filelist:
                    os.remove(os.path.join(pitch_dir, file))
                print("cleared " + pitch_dir)
        print()
    
    # build_training accuracy with pretranscripted song
    def draw_confusion_matrix(self):
        # create a list with for true and predicted pitches
        y_true, y_pred = self.get_statistics()
        labels=list(self.pitch_map.values())      
        # calculate confusion matrix
        c_mat = np.zeros((len(labels),len(labels)))
        for y_t, y_p in zip(y_true, y_pred):
            c_mat[y_t][y_p] += 1         
        # print detailed confusion matrix
        print("pred", end="\t")
        for label in labels:
            print(label, end="\t")
        print("\ntrue") 
        for i, label in enumerate(labels):
            print(label, end="\t")
            for val in c_mat[i]:
                print(int(val), end="\t")
            print("")