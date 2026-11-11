# ultrastar_pitch
This python application automates the pitch detection for ultrastar deluxe projects.  
  
## usage
The software can be used with user interface or as command line application. The windows [binary](https://github.com/paradigmn/ultrastar_pitch/releases) automatically starts as a graphical user application.

For cli execution, just run the command in your project folder. Additional flags are listed below. If the usdx file is named "notes.txt", no arguments are needed. Otherwise it has to be explicitly stated:  
`ultrastar-pitch name.txt`  
If everything went well a new file "notes_new.txt" should appear. In case a different output name is desired, it can be changed with the "-o" flag:  
`ultrastar-pitch -o name_new.txt`  

### optional models (better accuracy)
Two large models are not part of the package, because of their size. They are published in the release
[models-v1](https://github.com/paradigmn/ultrastar_pitch/releases/tag/models-v1) and can be downloaded with
`ultrastar-pitch --download-models` (or the button in the gui), which checks the SHA-256 checksums and puts
them into the user data folder. The tool looks for them in
`$ULTRASTAR_PITCH_RMVPE_MODEL` / `$ULTRASTAR_PITCH_VOCAL_MODEL`, next to the binary, in the user data
folder (`%LOCALAPPDATA%\ultrastar_pitch` on windows, `~/.local/share/ultrastar_pitch` on linux) and in a
`models/` folder next to the sources; the path can also be given with `--rmvpe-model` / `--vocal-model`
or chosen in the gui.

* **rmvpe.onnx** (186 MB, MIT): the [RMVPE](https://huggingface.co/Scicom-intl/rmvpe-onnx) vocal pitch model.
  If present, its result is combined with the built-in classifier. Measured on 197 songs this raises the
  share of correct pitches from 68 % to 74 % (with the soft key postprocessing). Without it the built-in
  classifier is used alone.
* **htdemucs_vocals.onnx** (304 MB): the vocal separation model (export instructions in `dev/`). With `-s`
  the vocals are isolated first (slow: about 20 s per minute of audio on a cpu, seconds on a gpu) and saved
  as `vocals.wav` next to the note file, so it is only done once. Both models then run on the mix and on
  the vocals and the votes are combined: another +1.5 points. An existing `vocals.wav` or a `#VOCALS`
  header (e.g. from [spleeter](https://github.com/deezer/spleeter)) is used automatically.

## installation
If you are using the [binary](https://github.com/paradigmn/ultrastar_pitch/releases), everything should run out of the box.  
In case of an error, try to install Microsoft Visual C++ Redistributable x64 (2010+2015).  
  
If you want to use the Python application, install [uv](https://docs.astral.sh/uv/) first and then do the following:
### windows
Install the newest version of [Python 3](https://www.python.org/downloads/windows/)  
Install the newest build of [ffmpeg](https://de.wikihow.com/FFmpeg-unter-Windows-installieren)  
Open CMD/PowerShell and type:
`uv tool install ultrastar-pitch`

The Python package uses ONNX Runtime with GPU support by default. Install a compatible
NVIDIA driver and CUDA/cuDNN runtime for GPU inference. CPU-only installations can
replace `onnxruntime-gpu` with `onnxruntime` in a local environment if needed.

The command-line mode does not require Tkinter. For the optional graphical interface,
install Tkinter on Ubuntu with:
`sudo apt install python3-tk`
### linux (debian)
Open a terminal and type:
`sudo apt-get install ffmpeg`
`uv tool install ultrastar-pitch`
  
## flags
Command line options for nono graphical execution:  
  
| flag | description                           |
|------|---------------------------------------|
| -h   | show this help message and exit       |
| -o   | specify output file name              |
| -g   | enable graphical user interface       |
| -m   | disable stochastic postprocessing     |
| -s   | separate the vocals first (needs the separation model), result is reused next time |
| --vocal-model | path of the vocal separation model (htdemucs_vocals.onnx) |
| --rmvpe-model | path of the RMVPE pitch model (rmvpe.onnx)       |
| --no-rmvpe | use the built-in classifier only                    |
| --download-models | download the optional models (rmvpe, vocals or all) and exit |
| -a   | show prediction accuracy (debug flag) |
| -l   | set logging level (debug flag)        |
  
## developer information
### build instructions (windows only)
The software can be compiled into a single standalone binary. To achieve this, an additional package has to be installed.  
`uv sync --group dev`
  
To include ffmpeg into the binary, it needs to be placed as specified by the setup.spec file. The default would be "ffmpeg\bin\ffmpeg.exe" within the project root directory.  
  
The building process is fairly easy. Just execute the following command within CMD/PowerShell:
`uv run pyinstaller setup.spec`
### local development
Create the project environment and install all locked dependencies with:
`uv sync`

The command-line application can then be run with:
`uv run ultrastar-pitch`

Dependencies are declared without version pins in `pyproject.toml` and resolved in `uv.lock`.
Refresh the lockfile to the newest compatible versions with:
`uv lock --upgrade`
### implementation
The software takes a timed usdx project file and the corresponding audio file. The song is converted into a mono wav and gets split into the predefined audio segments. These chunks are divided into blocks to be transformed into features. The output is then fed into a neuronal network, which yields the probabilities of the 12 pitch classes per note. If the RMVPE model is installed, its salience map is summed per note, folded to the 12 pitch classes and combined with the probabilities above (log-linear, weight 0.6). With separated vocals the models run on the mix and on the vocals and both votes are combined (vocals weight 0.35). Statistical postprocessing determines the pseudo key of the song and adds its pitch class probabilities as a soft prior to the scores, so only uncertain notes are moved towards the key (the old hard correction snapped every out-of-key note and lost accuracy within one semitone).
  
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
| VocalSeparator          | isolate the vocals with the Demucs onnx model (optional)   |
| rmvpe                   | RMVPE pitch model and the note features derived from it    |
| tuning_correction       | tuning offset, pitch class per segment, global transposition |
| DetectionPipeline       | execute the models above in one pipeline                   |
  
Each modules can be used in your own project. Just import them like this:  
`from ultrastar_pitch import module`  
  
### changelog
| version | changelog                                                                             |
|---------|---------------------------------------------------------------------------------------|
| 2.0.0   | note file format v1/v2, optional RMVPE ensemble and vocal separation, soft key prior, evaluation scripts in dev/ |
| 1.0.1   | fixed encoding issues and enforce python version                                      |
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

### legacy downloads
[Old Binaries](https://drive.google.com/drive/folders/19c2Y6LVVfpqIB-leLPUehs-kihZVccyM?usp=sharing)  
[Old Python Wheels](https://drive.google.com/drive/folders/1oCyOqvieQTcq-MWRKlbaoa4dBo5O7V7y?usp=sharing)

### todo
* write unit tests  
* evaluate approaches for partial automate timing detection  
* siamese neural network for multi octave detection  
* evaluate accuracy with higher samplerates  
* integrate preprocessing into model  
* evaluate fcnn networks with variable input length  
* integrate resampling into model  
* evaluate accuracy with sequence models (rnns, grus, lstms)  
* more logging  
