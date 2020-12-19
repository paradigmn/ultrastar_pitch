# ultrastar_pitch
This python application automates the pitch detection for ultrastar deluxe projects.  
  
## usage
The software can be used with user interface or as command line application. The windows [binary](https://my.pcloud.com/publink/show?code=kZt3wA7ZnxhL5olW9IkS2FX7DchyBp5k4J37) automatically starts as a graphical user application.

For cli execution, just run the command in your project folder. Additional flags are listed below. If the usdx file is named "notes.txt", no arguments are needed. Otherwise it has to be explicitly stated:  
`ultrastar-pitch name.txt`  
If everything went well a new file "notes_new.txt" should appear. In case a different output name is desired, it can be changed with the "-o" flag:  
`ultrastar-pitch -o name_new.txt`  

### spleeter
The [spleeter project](https://github.com/deezer/spleeter) uses deep learning to separate the vocal- and instrumental part of a song. In some cases (mainly acoustic songs), ultrastar-pitch performs better with the isolated vocal data. In other cases, its accuracy drops due to the introduced artifacts / information loss.  
  
In order to use spleeter with ultrastar-pitch, a couple of steps need to be performed:  
  
* spleeter produces multiple outputs. The vocals.wav file needs to be placed in the same directory as the notes.txt file.  
* in the notes.txt file the "#MP3" tag needs to be changed, that it refers to the vocal file (e.g. "#MP3:vocals.wav").  
* after running ultrastar-pitch, the "#MP3" tag needs to be reverted back  
  
## installation
If you are using the [binary](https://my.pcloud.com/publink/show?code=kZt3wA7ZnxhL5olW9IkS2FX7DchyBp5k4J37), everything should run out of the box.  
In case of an error, try to install Microsoft Visual C++ Redistributable x64 (2010+2015).  
  
If you want to use the python application do the following:  
### windows
Install the newest version of [Python 3](https://www.python.org/downloads/windows/)  
Install the newest build of [ffmpeg](https://de.wikihow.com/FFmpeg-unter-Windows-installieren)  
Open CMD/Powershell and type:  
`pip install ultrastar-pitch`  
### linux (debian)
Open a terminal and type:  
`sudo apt-get install python3 python3-pip ffmpeg`  
`pip install ultrastar-pitch`  
  
## flags
Command line options for nono graphical execution:  
  
| flag | description                           |
|------|---------------------------------------|
| -h   | show this help message and exit       |
| -o   | specify output file name              |
| -g   | enable graphical user interface       |
| -m   | disable stochastic postprocessing     |
| -a   | show prediction accuracy (debug flag) |
| -l   | set logging level (debug flag)        |
  
## developer information
### build instructions (windows only)
The software can be compiled into a single standalone binary. To achieve this, an additional package has to be installed.  
`pip install pyinstaller`  
  
To include ffmpeg into the binary, it needs to be placed as specified by the setup.spec file. The default would be "ffmpeg\bin\ffmpeg.exe" within the project root directory.  
  
The building process is fairly easy. Just execute the following command within the cmd/powershell:  
`pyinstaller setup.spec`  
### implementation
The software takes a timed usdx project file and the corresponding audio file. The song is converted into a mono wav and gets split into the predefined audio segments. These chunks are divided into blocks to be transformed into features. The output is then fed into a neuronal network to determine the pitches. Statistical postprocessing is used to determine pseudo key of the song. The predicted pitches are reevaluated to match the pseudo key.
  
The deep learning model was trained on a large karaoke database. Details for building your own model can be found in the dev/ folder.
  
### accuracy
The precision of this method changes greatly with the analyzed audio. For example a ballad with slow background music and a strong female voice can get an accuracy of over 90%, while a rock song with loud background music and a rough male voice can drop below 30%.  

### api
The software consists of various modules:  
  
| module                  | description                                                |
|-------------------------|------------------------------------------------------------|
| ProjectParser           | parse note.txt project file for singable audio segments    |
| AudioPreprocessor       | transform audio segments into features for pitch detection |
| PitchClassifier         | predict pitches from features                              |
| StochasticPostprocessor | increase prediction accuracy by applying stochastics       |
| DetectionPipeline       | execute the models above in one pipeline                   |
  
Each modules can be used in your own project. Just import them like this:  
`from ultrastar_pitch import module`  
  
### changelog
| version | changelog                                                                             |
|---------|---------------------------------------------------------------------------------------|
| 1.0.0   | minimalistic gui, new preprocessing algo, new model, api restructuring                |
| 0.82    | use dynamic stride to accelerate processing, removed model initializer, minor fixes   |
| 0.81    | fixed exception for silent audio input                                                |
| 0.80    | switched to onnx framework for model load and inference                               |
| 0.73    | added support for ansi encoded note.txt files                                         |
| 0.72    | optimized performance with micro optimizations                                        |
| 0.71    | switched from median to highest likelihood pitch evaluation                           |
| 0.70    | use statistical distribution to improve the prediction                                |
| 0.64    | switched from scipy.io to wave library to load audio                                  |
| 0.6x    | improved model accuracy and prediction speed                                          |
| 0.60    | switching from average to median pitch evaluation, changed license                    |
| 0.50    | implemented PCA, bumped tensorflow to version 2 and improved model accuracy and speed |
| 0.41    | fixed behavior for some edge cases                                                    |
| 0.40    | complete restructuring and application works as command line application              |
| 0.34    | bug fixes                                                                             |
| 0.33    | using absolute paths instead of relative ones                                         |
| 0.32    | added model to PyPi repo -> is now used by default                                    |
| 0.31    | substitute license field with classifier and updated installer script                 |
| 0.30    | added deep learning support and improved source code readability                      |
| 0.21    | got pyinstaller running -> binary doesn't need separate ffmpeg anymore                |
| 0.20    | replaced pydub by subprocess and scipy wavfile read -> faster processing              |
| 0.10    | first running implementation                                                          |
  
### todo
* more sophisticated statistical postprocessing  
* evaluate approaches for partial automate timing detection  
* siamese neural network for multi octave detection  
* evaluate accuracy with higher samplerates  
* integrate preprocessing into model  
* evaluate fcnn networks with variable input length  
* integrate resampling into model  
* evaluate accuracy with sequence models (rnns, grus, lstms)  
* more logging  





