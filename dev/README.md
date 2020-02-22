# Overview
This folder contains anythink needed to build your own model for ultrastar-pitch.

## Training data
You can download the training data under the following link:  
https://drive.google.com/open?id=1FvfBQAhkE6mGlmFxjaoQxeZvQ1qC2EI5  

You will find the following zip archieves:  

### raw
The voice audio segments sampled at 16kHz mono.
### vocal_raw
The voice audio segments sampled at 16kHz mono, separated voice by Spleeter.
### fft
The averaged preemphasized 2048-fft of the raw dataset.
### fft_vocal
The averaged preemphasized 2048-fft of the vocal_raw dataset.  
<br/><br/>
Each archieve contains a beta and a stable folder. These are two seperated databases. The beta set has fewer misslabeld pitches but is a lot smaller than the stable one.  
The raw data will stay more or less constants during this process, while the transformation method to create features is going to change regulary until the most promissing one is found.  

## Scripts
### generate_raw.py
If you have your own karaoke database, you can try to generate your own raw data by running the generate_raw script.
### generate_spleeter_raw.py
Same as generate_raw, but apply Spleeter separation beforehand.
### generate_training.py
Use the previous generated audio segments (or the content of the raw/vocal_raw archieve) to create features to train a neuronal network.
### train_model.ipynb
A Jupyter notebook containing an minimal example how to load the training data and train a model.