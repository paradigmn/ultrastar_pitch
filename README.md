# ultrastar_pitch
This python script is an attempt to automate the pitch detection for USDX projects. 

## usage
The core file of this project is "pitch_detection.py" which provides the functionality 
for parsing and processing USDX projects. 

The scripts folder provides the "get_pitch.py" script, which calls the mandatory functions to process a USDX project.
If all dependencies are met, just copy this file in your project folder and run it. 
To work properly, it is necessary to name the usdx file "notes.txt" and the song file "song.mp3".
If everything went well, a new file "notes.txt.new" should appear.

For Windows 7 x64 there is a precompiled [executable](https://my.pcloud.com/publink/show?code=kZt3wA7ZnxhL5olW9IkS2FX7DchyBp5k4J37) available. 
It can be used the same way as described above.  
  
Note: Some virus scanner identify the binary as Bitcoin miner and therefore prevent the execution. This is unfortunate but not in my power to
control. Either add an exception or use the script instead.

### deep learning
If you want to improve the accuracy, you can use the the "keras\_get\_pitch" script or executable. It uses the deep learning 
model located in the "keras" folder. 

## dependencies
If you are using the binary, everything should run out of the box. 
In case of error occurrence, try to install Microsoft Visual C++ Redistributable x64 (2010+2015).  
  
If you prefer using the python script or feel the need to develop you own solution, read on.
### windows
Install the newest version of [Python 3](https://www.python.org/downloads/windows/)  
Install the newest build of [ffmpeg](https://de.wikihow.com/FFmpeg-unter-Windows-installieren)  
Open CMD/Powershell and type:  
`pip install ultrastar-pitch`
### linux (debian)
Open a terminal and type:  
`sudo apt-get install python3 python3-pip ffmpeg`  
`pip install ultrastar-pitch`  

Pip should install all dependencies automatically, if not run  
`pip install scipy numpy keras tensorflow`

## developer information
### build instructions (windows only)
The software can be compiled into a single standalone binary. To achieve this, additional requirements need to be installed.  
`pip install pyinstaller pywin32 setuptools pypiwin32`  

The software used to generate the binary is called pyinstaller. The build recipe is called setup.spec. 
It links to the get_pitch.py script in the example folder. To build your own application the MODULE variable within the 
recipe needs to be changed to your own script. The same goes for FFMPEG_DIR, if your ffmpeg.exe isn't located in the 
standard folder.

To generate the executable, change to your working directory and run  
`pyinstaller setup.spec`

### implementation
The software takes a timed USDX file and the corresponding audio file. The song is converted into a mono wav file 
and gets split into the predefined syllables. These chunks will then be separated into blocks to be fourier transformed and 
averaged. The fft gets iterated and the base tone is added to its first two harmonics. 
The greatest value is expected to be the desired pitch.  
  
The deep learning analysis was trained by a large karaoke database. It uses an averaged 2048 samples fft like the 
original algorithm but performing much better. The training data was filtered beforehand by checking the label against 
the prediction of the algorithm.  
The model structure can be derived from its name. E.g: "keras\_tf\_1025\_240\_120\_12\_fft\_0.model" stands for a Keras model, 
which uses the Tensorflow backend. It takes 1025 input values, has two hidden layers with 240 and respectively 120 nodes 
and 12 outputs. Furthermore the input was fft transformed and the model revision is 0.  
  
You can build and load your own model for analysis. The necessary parameter and methods are listed below.

### accuracy
The precision of this method vary greatly with the analyzed audio. For example a ballad with slow background music and 
a strong female voice can get an accuracy of over 90%, while a rock song with loud background music and a rough male voice 
can drop below 30%.  
  
The average accuracy of the original approach is 54%, while deep learning is about 75% correct.

### functionality

1. initiation  
To access the functionality of the PitchDetection class, an object needs to be created.  
`test = PitchDetection()`  
The init method can take in several positional arguments:  
`sample_rate` the resampling frequency for analyzing  
`fft_len` the length of the fft window  
`fg1` the lowest frequency that can be detected  
`method` switch between classic and deep learning analysis  
These three arguments can drastically influence the accuracy and should't be changed unless you know exactly what you are doing.

2. public methods  
These are the general methods to create an pitch analyzing application  
`load_project(proj_dir)`	read project data and begin analyzing  
`save_project()` save successfully converted file to disk  
`analyse_audio(audio_samples)` takes a numpy audio array and returns the pitch and and averaged fft array  
`fft_to_pitch(fft)` takes a numpy fft array and returns the corresponding pitch  
`build_training_data(data_dir, mode="...")` save original or analyzed pitches as numpy binary for machine learning applications  
`clear_training_data(data_dir)` removes previously generated data to start fresh  
`draw_confusion_matrix()` compare accuracy of analyzed pitches with a confusion matrix  
`get_statistics()` print accuracy and other statistical data  
`load_keras_model(model_path)` load deep learning model to improove accuracy

3. class methods and variables  
Some functionality, which doesn't require an object.  
`pitch_map` an dictionary which allows translation between numeric and ascii pitch notation  
`get_pitch(freq, form="...")` turns a given frequency into pitch notation  
`zero_pad_array(array, new_size)` zero pads a numpy array at the end

### version history
v0.10 - first running implementation  
v0.20 - replaced pydub by subprocess and scipy wavfile read -> faster processing  
v0.21 - got pyinstaller running -> binary doesn't need separate ffmpeg anymore  
v0.30 - added deep learning support and improved source code readability  
v0.31 - substitute license field with classifier and updated installer script
v0.32 - added model to PyPi repo -> is now used by default
v0.33 - using absolute paths instead of relative ones

### todo
* improve exception handling
* change from fft algorithm to wavelet transformation, to get a better overall frequency resolution
* implement GUI for easier access





