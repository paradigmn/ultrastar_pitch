# this file defines a class used to automatically calculate the pitch for a given USDX project

import os, sys, subprocess
import numpy as np
import scipy.io.wavfile

class PitchDetection(object):
    # pitch map for conversion
    pitch_map = {0 : "C ", 1 : "C#", 2 : "D ", 3 : "D#", 4 : "E ", 5 : "F ", 6 : "F#", 7 : "G ", 8 : "G#", 9 : "A ", 10 : "A#", 11 : "B "}
    # mp3 file name
    __usdx_song = "song.mp3"
    # pitch file name
    __usdx_file = "notes.txt"
    
    def __init__(self, sample_rate=16000, fft_len=2048, fg1=140):
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
        # data array: [pitch start, pitch end, labeled pitch, calculated frequency]
        self.__usdx_data = []
        # array with averaged ffts for every pitch
        self.__fft_arr = []
        # counter for naming training data
        self.__file_counter = 0
        
        # change the ffmpeg path depending on using script or executable    
        if getattr(sys,'frozen',False):
            self.FFMPEG = os.path.join(sys._MEIPASS, 'ffmpeg.exe')
        else:
            self.FFMPEG = 'ffmpeg'
        
    @classmethod
    # return the pitch corresponding to a given frequency in different formats
    def get_pitch(cls, freq, form="short"):
        if form == "octave":
            # calculate number of half steps from C0
            h = round(12*np.log2(freq/(440*2**(-4.75))))       
            # determine pitch 0 -> C, 11 -> B ...  
            pitch = h % 12 
            # return numeric octave (0, 1, 2, ...)                                       
            return(int(h // 12))
        elif form == "ascii":
            h = round(12*np.log2(freq/(440*2**(-4.75))))       
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
        old_size = len(array)
        return np.pad(array, (0, (new_size - old_size)), 'constant') 
              
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
                tmp = len(pitch_samples[(i * self.__fft_len) // 4:-1])
                time_frame = pitch_samples[(i * self.__fft_len) // 4:-1] * np.hanning(tmp)
                time_frame = self.zero_pad_array(time_frame, self.__fft_len)       
            # fourier transform and averaging
            tmp = abs(np.fft.fft(time_frame) / self.__fft_len)
            fft_spectrum = tmp[0:self.__fft_len // 2 + 1]
            fft_spectrum[1:-1] *= 2
            avg_fft += fft_spectrum
        return(avg_fft)
    
    # calculates the pitch frequency from a fft
    def __fft_to_pitch_freq(self, fft):       
        idx, weight_max = 0, 0
        # accumulate the base tone and the first two harmonics
        for i in range((self.__fg1 * self.__fft_len) // self.__sample_rate, (self.__fg2 * self.__fft_len) // self.__sample_rate):
            weight = fft[i];
            weight = weight + 1/3 * sum(fft[2 * i - 1:2 * i + 1])
            weight = weight + 1/5 * sum(fft[3 * i - 2:3 * i + 2])
            # catch the peak value
            if (weight > weight_max):
                weight_max = weight
                idx = i
        return(self.__f[idx])  
        
    # load audio into mono numpy array
    def __load_samples(self):
        # convert mp3 to mono wav
        subprocess.run([self.FFMPEG, '-i', os.path.join(self.__proj_dir, self.__usdx_song), '-y', '-ac', '1', '-ar', 
                        str(self.__sample_rate), os.path.join(self.__proj_dir, "tmp.wav")], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # load wav into numpy array for processing
        samples = scipy.io.wavfile.read(os.path.join(self.__proj_dir, "tmp.wav"))
        self.__samples_mono = samples[1]
    
    # analyse usdx file    
    def __parse_data(self):
        bpm, gap = 0, 0
        file = open(os.path.join(self.__proj_dir, self.__usdx_file), "r")
        for line in file.readlines():
            # parse header
            if line.startswith("#"):
                if line.startswith("#BPM:"):
                    bpm = float(line.split(":")[1].replace(",", "."))
                elif line.startswith("#GAP:"):
                    gap = float(line.split(":")[1])
            # parse song data
            elif line.startswith(":") or line.startswith("*") and bpm and gap:
                tmp = line.split(" ")
                start_time = gap + float(tmp[1]) * (15000 / bpm)
                end_time = gap + (float(tmp[1]) + float(tmp[2])) * (15000 / bpm)
                self.__usdx_data.append([start_time, end_time, int(tmp[3]), float(0)])
    
    # iterate pitch list over song
    def __analyse_audio(self):
        for data in self.__usdx_data:
            start_sample = int((data[0] * self.__sample_rate) / 1000)
            end_sample = int((data[1] * self.__sample_rate) / 1000)
            pitch_samples = self.__samples_mono[start_sample:end_sample]
            self.__fft_arr.append(self.__avg_fft(pitch_samples))
            data[3] = self.__fft_to_pitch_freq(self.__fft_arr[-1])
               
    # project init    
    def load_project(self, proj_dir):
        # clear lists to avoid problems on reuse
        self.__usdx_data.clear()
        self.__fft_arr.clear()
        # make directory public
        self.__proj_dir = proj_dir
        # extract sound samples and convert to mono
        self.__load_samples()
        # create pitch list from project file
        self.__parse_data()
        # transform the samples and calculate the pitch
        self.__analyse_audio()
        
    # create a new file to save the project
    def save_project(self):
        file_new = open(os.path.join(self.__proj_dir, self.__usdx_file + str(".new")), "w+")
        file_org = open(os.path.join(self.__proj_dir, self.__usdx_file), "r")
        i = 0
        for line in file_org.readlines():
            if line.startswith(":") or line.startswith("*"):
                line = line.split(" ")
                line[3] = str(self.get_pitch(self.__usdx_data[i][3], "short"))
                line = ' '.join(line)
                i = i + 1    
            file_new.write(line)            
            
        
    # return numpy array of averaged fft samples
    def get_avg_fft(self):
        return(self.__fft_arr)

    # return list of pitch data
    def get_pitch_data(self):
        return(self.__usdx_data)
    
    # creates training data for deep learning
    def create_training_data(self, data_dir, label="original"):
        # create subfolder
        for pitch in self.pitch_map:
            os.makedirs(os.path.join(data_dir, self.pitch_map[pitch]), exist_ok=True)  
        # create a csv file for each analysed pitch   
        for i, fft_spectrum in enumerate(self.__fft_arr):
            fft_spectrum = fft_spectrum / max(fft_spectrum)
            if label == "original":
                folder = self.pitch_map[self.__usdx_data[i][2] % 12]
            else:
                folder = self.get_pitch(self.__usdx_data[i][3], "ascii")
               
            csv_path = os.path.join(data_dir, (str(folder) + "/" + str(self.__file_counter) + ".csv")) 
            np.savetxt(csv_path, fft_spectrum, delimiter="\r\n")         
            self.__file_counter += 1
 
    
    # test accuracy with pretranscripted song
    def draw_confusion_matrix(self):
        # create a list with for true and predicted pitches
        tmp = [row[2] for row in self.__usdx_data]
        y_true = [x % 12 for x in tmp]
        tmp = [row[3] for row in self.__usdx_data]
        y_pred = [self.get_pitch(x, form="short") for x in tmp]
        labels=list(self.pitch_map.values())
        
        # calculate correctly predicted pitches
        tmp = 0
        for i in enumerate(y_true):
            if y_true[i[0]] == y_pred[i[0]]:
                tmp = tmp + 1
        
        # print statistical data
        print(str(len(y_true)) + " samples")
        print(str(tmp / len(y_true) * 100) + "% accuracy\n")
        
        # calculate confusion matrix
        c_mat = np.zeros((len(labels),len(labels)))
        for i in range(len(y_true)):
            c_mat[y_true[i]][y_pred[i]] += 1
            
        # print detailed confusion matrix
        print("pred", end="\t")
        for label in labels:
            print(label, end="\t")
        print("\ntrue") 
        for i in range(len(labels)):
            print(labels[i], end="\t")
            for val in c_mat[i]:
                print(int(val), end="\t")
            print("")  

