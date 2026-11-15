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
  If present, its result is combined with the built-in classifier and the timing of the notes is checked
  against the singing. Measured on 197 songs the built-in classifier alone finds 75 % of the pitch
  classes (timing of the file, without postprocessing); with RMVPE, the timing check, the tuning
  correction and the postprocessing the tool finds 84 %. Without it the built-in classifier is used alone.
* **htdemucs_vocals.onnx** (304 MB): the vocal separation model (export instructions in `dev/`). With `-s`
  the vocals are isolated first (slow: about 20 s per minute of audio on a cpu, seconds on a gpu) and saved
  as `vocals.wav` next to the note file, so it is only done once. Both models then run on the mix and on
  the vocals and the votes are combined: another +0.8 points (83.8 % -> 84.6 %). An existing `vocals.wav` or a `#VOCALS`
  header (e.g. from [spleeter](https://github.com/deezer/spleeter)) is used automatically.

### note file hygiene
Besides the pitches the tool checks the structure of the note file and reports problems as warnings: overlapping notes, line breaks before the end of a note or after the start of the next one, double line breaks, notes without length or text, lines after `E`. On request it also fixes the output file (`--fix`, checkboxes in the gui):

* `structure`: overlapping notes are shortened, line breaks are moved into the pause, double line breaks and lines after `E` are removed
* `linebreaks`: every line break is put where the [ultrastar-scripts](https://github.com/barbeque-squared/ultrastar-scripts) of barbeque-squared put it (a fixed distance before the next line, depending on the pause)
* `whitespace`: a space at the end of a syllable moves to the start of the next one (ultrastar-scripts)

`ultrastar-pitch --fix all` applies all three, `--fix structure,linebreaks` a selection; without `--fix` only the pitches change. `--lint` (gui: "Style hints") also lists style hints: line break positions, spaces, the share of golden notes, missing `#YEAR`, `#LANGUAGE`, `#GENRE` or `#VERSION`, missing cover, background or video files.
A new pitch is written in the octave closest to the old note, so a note whose pitch class is right stays as it is and the melody keeps its contour (until 1.x every note was written as 0 .. 11).

The checks also run alone, without the pitch detection (no audio, no models, a fraction of a second per file): `ultrastar-pitch --check-only notes.txt` checks one file, `ultrastar-pitch --check-only songs/` all note files of a folder and its subfolders (13,000 files in about 30 s) and lists only the files with findings. With `--fix` a fixed copy `<name>_fixed.txt` is written next to every file a fix changes, the originals stay as they are; `--lint` adds the style hints. In the gui the "Check" button does the same for the chosen file.

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
| --no-timing-correction | analyse the notes exactly at the timing of the file |
| --no-context-model | postprocess without the note context network |
| --pitchnet-model | path of another pitch classifier (e.g. the 1.x model pitchnet_2020_12_14.onnx) |
| --check-only | only check the note file (or all note files of a folder), no pitch detection |
| --fix | fix the output file: structure, linebreaks, whitespace (comma separated) or all |
| --lint | also report style hints (ultrastar-scripts conventions, missing headers and files) |
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
The software takes a timed usdx project file and the corresponding audio file. The song is converted into a mono 16 kHz wav and its power spectrum is computed per 10 ms frame. The neuronal network (`pitchnet_2026_11_13`, 17,155 parameters, structure and training in `dev/pitchnet_2026.ipynb`) maps every frame onto a log-frequency axis, stacks the harmonics and yields the log probabilities of the 12 pitch classes per frame, which are summed over the frames of every note. If the RMVPE model is installed, it first estimates the tuning of the voice and the spectrum is shifted by it before the network runs, so detuned songs (e.g. 432 Hz) are analysed as if they were in tune; its salience map is summed per note, folded to the 12 pitch classes and combined with the probabilities above (log-linear, weight 0.25). With separated vocals the models run on the mix and on the vocals and both votes are combined (vocals weight 0.5). Measured end-to-end on 197 songs, the complete pipeline gets 83.8 % of the notes exact (84.6 % with vocals) against 78.8 % (80.3 %) with the model of the 1.x versions, and only 12.3 % of the notes are more than one semitone off (11.3 % with vocals); the network alone gets 75.4 % against 68.4 %. The old model `pitchnet_2020_12_14` stays in `ultrastar_pitch/binaries/` for comparisons (`--pitchnet-model`), and the numbers of the timing check, the repetition pooling and the key prior below were measured with it. With RMVPE the timing of the notes is checked first: the voicing of the mix (how clearly someone sings, per 10 ms) is cross-correlated with the on/off signal of the notes. A clear offset (wrong `#GAP`) is used for the analysis, the note file keeps its timing (end-to-end on 197 songs +3.2 points, 75.3 % -> 78.8 % on the mix and 76.8 % -> 80.3 % with vocals; with the thresholds chosen on one half of the songs and tested on the other +2.9 points; songs that are 60 ms or more off gain 9 to 27 points). Notes and audio that hardly fit together (wrong audio file) and a timing that drifts through the song (wrong `#BPM`) are reported as warnings. The postprocessing first shares the evidence of repeated phrases: notes at the same position of phrases with the same lyrics or the same rhythm (chorus, repeated lines) get the mean scores of their repetitions, each shifted by the transposition that fits best (key changes). This helps short notes most (+2.6 points below 60 ms). Then the pseudo key of the song is determined and its pitch class probabilities are added as a soft prior to the scores, so only uncertain notes are moved towards the key (the old hard correction snapped every out-of-key note and lost accuracy within one semitone). Measured end-to-end on 197 songs at the timing of the file: repetition pooling +1.6 points on the mix (73.6 % -> 75.3 %), +1.2 points with vocals (75.5 % -> 76.8 %). Finally small recurrent networks (four bidirectional GRUs of 98k parameters each, their log probabilities averaged, about 20 ms per song) read the whole sequence of notes, their scores before and after the pooling, the key, the pauses, the note lengths and the phrase starts, and correct uncertain notes from their neighbours: +2.1 points on the mix (81.7 % -> 83.8 %) and +1.7 with vocals (82.9 % -> 84.6 %), switched off with `--no-context-model`.
  
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
| repetition_pooling      | share the scores of repeated phrases (postprocessing)      |
| context_model           | correct uncertain notes from their neighbours (postprocessing) |
| VocalSeparator          | isolate the vocals with the Demucs onnx model (optional)   |
| rmvpe                   | RMVPE pitch model and the note features derived from it    |
| tuning_correction       | tuning offset, pitch class per segment, global transposition |
| timing_check            | offset, fit and tempo drift between notes and singing      |
| note_hygiene            | check and fix the structure of the note file (not the pitches) |
| DetectionPipeline       | execute the models above in one pipeline                   |
  
Each modules can be used in your own project. Just import them like this:  
`from ultrastar_pitch import module`  
  
### changelog
| version | changelog                                                                             |
|---------|---------------------------------------------------------------------------------------|
| 2.0.0   | new pitch model pitchnet_2026_11_13 (per frame, tuning corrected) and note context network (197 test songs 78.8 % -> 83.8 % of the notes exact), note file format v1/v2, optional RMVPE ensemble and vocal separation, timing check, repetition pooling and soft key prior, note file checks and fixes (rules of ultrastar-scripts) also without the pitch detection for whole folders, pitches keep their octave, evaluation scripts in dev/ |
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

### credits
The line break rules, the whitespace fix and the style checks of `note_hygiene` are adapted from [ultrastar-scripts](https://github.com/barbeque-squared/ultrastar-scripts) by barbeque-squared, used under the MIT license (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Thanks for sharing them!

### legacy downloads
[Old Binaries](https://drive.google.com/drive/folders/19c2Y6LVVfpqIB-leLPUehs-kihZVccyM?usp=sharing)  
[Old Python Wheels](https://drive.google.com/drive/folders/1oCyOqvieQTcq-MWRKlbaoa4dBo5O7V7y?usp=sharing)

### todo
* write unit tests  
* evaluate approaches for partial automate timing detection  
* siamese neural network for multi octave detection  
* evaluate accuracy with higher samplerates  
* integrate resampling into model  
* evaluate accuracy with sequence models (rnns, grus, lstms)  
* more logging  
