# ultrastar_pitch
This python application is an attempt to automate the pitch detection for USDX projects.  
  
## usage
Since version 0.40 the project is a standalone terminal application. After the installation it can simply be executed in the shell via:  
`ultrastar-pitch`  
  
If the usdx file is named "notes.txt", no arguments are necessary. If it is named any different it has to be explicitly stated:  
`ultrastar-pitch name.txt`  
If everything went well a new file "notes_new.txt" should appear. In case a different output name is desired it can be changed with the "-o" flag:  
`ultrastar-pitch -o name_new.txt`  
  
For Windows x64 there is a precompiled [executable](https://my.pcloud.com/publink/show?code=kZt3wA7ZnxhL5olW9IkS2FX7DchyBp5k4J37) available. Just place it in your project folder with a "note.txt" and "song.mp3" file and execute it.  
  
Note: Some virus scanner identify the binary as Bitcoin miner and therefore prevent the execution. This is unfortunate but not in my power to
control. Either add an exception or install the python application as described below.  
  
## installation
If you are using the binary, everything should run out of the box.  
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
  
Any instance of tensorflow < 2.0 should be removed before installing the software.
  
## developer information
### build instructions (windows only)
The software can be compiled into a single standalone binary. To achieve this an additional packet needs to be installed.  
`pip install pyinstaller`  
  
To include ffmpeg into the binary, it needs to be placed as specified by the setup.spec file. The default would be "ffmpeg\bin\ffmpeg.exe" within the project root.  
  
The building process is fairly easy. Just execute the following command within the cmd/powershell:  
`pyinstaller setup.spec`  
### implementation
The software takes a timed USDX file and the corresponding audio file. The song is converted into a mono wav file and gets split into the predefined audio segments. These chunks are divided into blocks to be fourier transformed and averaged. The output is fed into a neuronal network to determine the pitch.  
  
The deep learning model was trained by a large karaoke database.
The model structure can be derived from its name. E.g: "tf2\_256\_96\_12\_astft\_pca\_0.model" stands for a tensorflow 2 model, which takes 256 input values, has a hidden layers with 96 nodes and 12 outputs. Furthermore the input was averaged short time fourier transformed and decomposed with PCA. The model revision is 0.  
  
### accuracy
The precision of this method changes greatly with the analyzed audio. For example a ballad with slow background music and a strong female voice can get an accuracy of over 90%, while a rock song with loud background music and a rough male voice can drop below 30%.  
  
To get a  better impression, you can use the "-a" flag on a song which was already translated:  
`ultrastar-pitch -a`  
This will display the accuracy of the prediction and a confusion matrix to see how close the classifier was.  
  
### functionality
The software is based on three modules:  
  
* project_parser.py (parse the project for singable notes and yield audio segments)  
* preprocessing.py (transform segments into features of uniform size)  
* classification.py (predict pitch based on the provided pitches)  
  
Each modules contains one or more classes to provide the needed functionality. To make use of them in your own project just import them like this:  
`from ultrastar_pitch.module import class`  
  
### version history
v0.10 - first running implementation  
v0.20 - replaced pydub by subprocess and scipy wavfile read -> faster processing  
v0.21 - got pyinstaller running -> binary doesn't need separate ffmpeg anymore  
v0.30 - added deep learning support and improved source code readability  
v0.31 - substitute license field with classifier and updated installer script  
v0.32 - added model to PyPi repo -> is now used by default  
v0.33 - using absolute paths instead of relative ones  
v0.34 - bug fixes  
v0.40 - complete restructuring and application works as command line application  
v0.41 - fixed behavior for some edge cases  
v0.50 - implemented PCA, bumped tensorflow to version 2 and improved model accuracy and speed  
  
### todo
* change from averaged fft to block median classification  
* define labeling functions to sort out mislabeled data  
* use statistical distribution and markov chains to evaluate the prediction  
* improve exception handling  
* change from fft algorithm to wavelet transformation to get a better overall frequency resolution  
* implement GUI for easier access  





