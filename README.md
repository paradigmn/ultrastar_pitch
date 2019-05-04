# ultrastar_pitch
This Python script is an attempt to automate the pitch detection for USDX projects. 

## usage
The core file of this project is "pitch_detection.py" which provides the functionality 
for parsing and processing USDX projects. 

The examples folder provides the "get_pitch.py" script, which calls the mandatory functions to process a USDX project.
If all dependencies are met, just copy this file in your project folder and run it.  
To work properly it is necessary to name the usdx file "notes.txt" and the song file "song.mp3".
If everything went well, a new file "notes.txt.new" should appear.

For Windows 7 x64 there is a precompiled [executable](https://my.pcloud.com/publink/show?code=XZ1WuE7ZywiEsI3JbVF4j6ANeOVFY7Reft97) available.    
This binary still depends on ffmpeg, which can be installed according to this [tutorial](https://de.wikihow.com/FFmpeg-unter-Windows-installieren).

## dependencies
If you are using the binary, just follow the link above, otherwise continue reading.
### windows
Install the newest version of [Python 3](https://www.python.org/downloads/windows/)  
Install the newest build of [ffmpeg](https://ffmpeg.zeranoe.com/builds/)  
Open CMD/Powershell and type:  
`pip install ultrastar-pitch`
### linux (debian)
Open a terminal and type:  
`sudo apt-get install python3 python3-pip ffmpeg`  
`sudo pip3 install ultrastar-pitch`  

Pip should install all dependencies automatically, if not run  
`sudo pip3 install pydub numpy`

## development guide
### build instructions
!!!not implemented at this point!!!  
The software can be compiled into a single standalone binary. To achieve this, additional requirements need to be installed.  
`pip install pyinstaller pywin32 setuptools pypiwin32`  
to be continued ...

### implementation
The software takes a timed USDX file and the corresponding audio file. The song is converted into a mono wav file 
and gets split into the predefined syllables. These chunks will then be separated in blocks to be fourier transformed and averaged.
The fft gets iterated and the base tone is added to its first two harmonics. The greatest value is expected to be the desired pitch.

### accuracy
The precision of this method vary strongly with the analyzed audio. For example a ballad with slow background music and a strong female voice
can get an accuracy of over 90%, while a rock song with loud background music and a rough male voice can drop below 30%.

### functionality

1. initiation  
To access the functionality of the PitchDetection class, an object needs to be created.  
`test = PitchDetection()`  
The init method can take in several positional arguments:  
`sample_rate` the resampling frequency for analizing  
`fft_len` the lenght of the fft window  
`fg1` the lowest frequency that can be detected  
These three arguments can drastically influence the accuracy and should't be changed unless you know exactly what you are doing.

2. public methods
These are the general methods to create an pitch analizing application  
`load_project(proj_dir)`	read project data and begin analizing  
`save_project()` save successfully converted file to disk  
`create_training_data(data_dir, label="...")` save original or analized pitches as .csv files for machine learning applications  
`draw_confusion_matrix()` compare accuracy of analized pitches with a confusion matrix

3. class methods and variables
Some functionality, which doesn't require an object.  
`pitch_map` an dictionary which allows translation between numeric and ascii pitch notation  
`get_pitch(freq, form="...")` turns a given frequency into pitch notation  
`zero_pad_array(array, new_size)` zero pads a numpy array at the end

### todo
* get pyinstaller running reliable
* maybe replace pydub as converter interface
* add exception handling
* change from fft algorithm to wavlet transformation, to get a better overall frequency resolution
* implement GUI for easier access





