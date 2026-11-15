# Overview
This folder contains everything needed to build and evaluate your own model for ultrastar-pitch.
The package itself lives in `ultrastar_pitch/`; everything here is for development only.

## Folder layout
| Folder | Content | In git |
|---|---|---|
| `dev/` | development scripts and notebooks | yes |
| `dev/lists/` | song lists (e.g. `exclude.txt`, the song selections of the evaluation runs) | no |
| `dev/results/` | measurement results (`scores.csv` of `label_quality.py`, batch CSVs and logs) | no |
| `dev/testdata/` | test audio and note files (`song.mp3`, `notes.txt`, ...) | no |
| `models/` (repo root) | large ONNX models: `rmvpe.onnx` (Hugging Face `Scicom-intl/rmvpe-onnx`), `htdemucs_vocals.onnx` (exported with `export_demucs_vocals.py --vocals-only`) | no |
| `songs/` (repo root) | the karaoke song database | no |

The vocal separation of the package looks for its model via the environment variable
`ULTRASTAR_PITCH_VOCAL_MODEL`, the RMVPE model via `ULTRASTAR_PITCH_RMVPE_MODEL`; both are also found in `models/` next to the sources (see `ultrastar_pitch/model_files.py`). CLI: `-s/--separate`, `--vocal-model`, `--rmvpe-model`, `--no-rmvpe`; GUI: checkbox and model paths.

## Training data
Note: The training data was removed to avoid copyright issues. If you are interested in the data, please contact me via e-mail.

## Scripts
The scripts of the 1.x model (`generate_raw.py`, `generate_spleeter_raw.py`, `train_model.ipynb`, training of
`pitchnet_2020_12_14`) are no longer part of this branch; they are in the history of `master`. The scripts below
run on a local song database (note files with their audio) that is not part of the repository.
### pitchnet_2026.ipynb / pitchnet_model.py / pitchnet_data.py / pitchnet_train.py
The new pitch class model: the first version `pitchnet_2026_11_11` (32 channels) and the packaged default `pitchnet_2026_11_13` (run `ch64`, 64 channels, notebook section 9). The notebook documents the design (old model and its weaknesses, front end, harmonic stacking, transposition equivariant body), runs the checks, exports the untrained ONNX with the measured runtime, compares the training runs and evaluates the chosen one on the test songs (section 8). The packaged default is `ultrastar_pitch/binaries/pitchnet_2026_11_13.onnx` (the first version stays in `models/`); the old model of the 1.x versions stays there as `pitchnet_2020_12_14.onnx` for comparisons (`--pitchnet-model`). `pitchnet_model.py` holds the model and the export, `pitchnet_data.py plan` selects and cleans the songs from `dev/results/scores.csv` (quality threshold, drift, evaluation songs and their duplicates excluded, group-aware train/val split -> `dev/lists/pitchnet/`), `extract` writes the 16 kHz audio and the frame labels into `dev/results/pitchnet_cache/`, `show` checks the label alignment of one song; `pitchnet_train.py` trains one configuration (loss and augmentation variants) into `dev/results/pitchnet_runs/<name>/`:

    python dev/pitchnet_data.py plan dev/results/scores.csv --min-quality 0.5
    python dev/pitchnet_data.py extract --workers 8
    python dev/pitchnet_train.py --name base --loss smooth --aug remix,gain,polarity   # needs torch with cuda
    jupyter nbconvert --to notebook --execute --inplace dev/pitchnet_2026.ipynb

### pitchnet_val_compare.py / tuning_survey.py / pipeline_weights_val.py
The choices after the training, all on the validation songs (the 197 evaluation songs stay the test set).
`pitchnet_val_compare.py` compares the exported `model.onnx` of several training runs through the package code
on the mix, the vocals and the mix detuned by +-0.29 semitones, paired per song (results in
`dev/results/pitchnet_runs/val_compare.txt`). `tuning_survey.py` measures how often songs of the database are
detuned (RMVPE tuning estimate on a random sample, `dev/results/tuning_survey.txt`). `pipeline_weights_val.py`
caches the per note scores of RMVPE, the old pitchnet and the frame models on the validation songs (`extract`)
and searches the ensemble weights of `DetectionPipeline` (RMVPE, vocals, key prior) on them (`evaluate`,
`dev/results/pipeline_weights_val*.txt`):

    python dev/pitchnet_val_compare.py base near near_aug > dev/results/pitchnet_runs/val_compare.txt
    python dev/tuning_survey.py --n 600 > dev/results/tuning_survey.txt
    python dev/pipeline_weights_val.py extract --model base=dev/results/pitchnet_runs/base/model.onnx
    python dev/pipeline_weights_val.py evaluate > dev/results/pipeline_weights_val.txt

`extract` also builds the caches of other song sets (`--songs train` = the training songs of pitchnet_2026 in a
fixed random order, `--songs list --list FILE --root .` = the 197 evaluation songs, ids t000, ...; `--cache DIR`,
`--shard i/n` for parallel processes, `--sources mix`, `--no-old`, `--cpu --rmvpe-threads N` while the GPU is
busy). `--tuned` adds the RMVPE tuning estimate per source (`tuning_mix` = offset in cents, reliable) and the
frame model scores with the tuning correction (`NAME_tuned_mix`); `--update` adds missing models or the tuning
data to cached songs without running RMVPE again (only for a missing tuning estimate):

    python dev/pipeline_weights_val.py extract --songs val --update --tuned --no-old --model ch64=dev/results/pitchnet_runs/ch64/model.onnx

### pitchnet_tuning_val.py
Tuning correction for the frame model on the validation songs: the RMVPE tuning estimate of the voice shifts the
power spectrum before the model (`pitch_classifier.shift_power`, `PitchClassifier.predict_frames(samples,
tuning_cents)`), instead of training on detuned audio. Measures the sensitivity of the model to the tuning and the
correction on the mix, on the mix resampled by +-29 cents and on the vocals (`dev/results/pitchnet_tuning_val.txt`):

    CUDA_VISIBLE_DEVICES= python dev/pitchnet_tuning_val.py --workers 2 --threads 4 > dev/results/pitchnet_tuning_val.txt

Results (2026-11-13): the correction is neutral on songs in tune and recovers the detuned ones (mix +29 cents
79.1 -> 84.3 %, -29 cents 75.2 -> 84.3 %). In the pipeline (`context_gru_2026.py policy` / `compare`, threshold
0 = always when the estimate is reliable) it adds little to `base` (val +0.12, n.s.) but more to the wider run
`ch64`: val ch64@0 vs base mix +0.55 [+0.34 .. +0.83], vocals +0.32 [+0.10 .. +0.55]; test (197 songs) mix
81.30 -> 81.70 % (+0.41 [+0.21 .. +0.61]), vocals 82.51 -> 82.86 % (+0.36 [+0.18 .. +0.54])
(`dev/results/pipeline_compare_{val,test}_2026b.txt`, `pipeline_tuning_policy_val*.txt`). Other runs:
`tdil4b` (four blocks, time dilations 1, 2, 4, 8) +0.16 n.s. on val; ensemble weights unchanged for ch64
(`pipeline_weights_val_ch64.txt`).
### label_quality.py
Rates the timing and pitch labels of every song without ground truth (writes `dev/results/scores.csv`):

    python dev/label_quality.py songs --workers 8 --resume

### make_exclude_list.py / select_training_data.py
Build an exclude list (instrumental/karaoke versions) and select training songs by score, with duplicate detection.
### rmvpe_eval.py / rmvpe_eval_batch.py
Pitch class accuracy of RMVPE (and, with `--pitchnet`, of the packaged classifier) against the note files, for one song or many songs with a summary:

    python dev/rmvpe_eval_batch.py dev/lists/rmvpe_batch_songs.txt --root . --pitchnet --gpu \
        --workers 8 --threads 2 --quality-csv dev/results/scores.csv --out dev/results/rmvpe_batch.csv

### pipeline_eval.py / pipeline_compare.py
End-to-end accuracy of the real `DetectionPipeline` (what the tool writes) in several configurations, also per note length; `+key` = soft key prior only, `+pp` = repetition pooling, key prior and, with a frame model, the note context network (the default), `@file` = without the timing correction, `+pool` = repetition pooling and key prior without the note context network, `--pitchnet-model` = another pitch classifier onnx (e.g. the frame model `pitchnet_2026`). `pipeline_compare.py LOG_A:CONFIG LOG_B:CONFIG` pairs the per-song results of two logs (old model vs new model) with a bootstrap interval.
### context_gru_2026.py
The note context GRU (or a small transformer, `--arch transformer`) on top of the current pipeline with the frame
model: it adds a correction to exactly what the pipeline decides (timing correction, frame model and RMVPE,
repetition pooling, soft key prior). Trained on the caches of `pipeline_weights_val.py extract` of the training
songs, the epoch is chosen on the validation songs, `--test` reports the 197 evaluation songs once. `policy` and
`compare` evaluate pipeline configurations on a cache without the GRU (tuning correction thresholds `NAME@T`,
other frame models), paired per song:

    python dev/pipeline_weights_val.py extract --songs train --shard 0/2 --no-old --sources mix --tuned \
        --model base=models/pitchnet_2026_11_11.onnx --cache dev/results/context_cache/train
    python dev/context_gru_2026.py prepare dev/results/context_cache/train --out dev/results/gru2026_train.npz
    python dev/context_gru_2026.py prepare dev/results/pipeline_weights_cache --out dev/results/gru2026_val.npz
    python dev/context_gru_2026.py train dev/results/gru2026_train.npz dev/results/gru2026_val.npz   # needs torch
    python dev/context_gru_2026.py compare dev/results/pipeline_weights_cache --configs base,base@10,ch64

Results (2026-11-13, 4781 training songs, `--batch 16 --device cuda`, 20 epochs, epoch chosen on val): the GRU
(hidden 48, 36,812 parameters, `dev/results/gru2026_gru48.{pt,onnx}`, 4 ms per song) raises the validation
mix from 85.88 to 88.11 %; wider (96), two layers or the hashed syllables (`--text`) give the same, the
transformer (64 x 2) only 86.86 %. More songs help (529 songs: +1.2 on val). Test, applied once: mix 81.30 ->
82.71 % (+1.41 [+1.09 .. +1.70], 144 songs better, 46 worse), mix + vocals (trained on the mix) 82.51 -> 83.70 %
(+1.19 [+0.88 .. +1.50]); results in `dev/results/gru2026_*.txt`.

Combined with the wider frame model and the tuning correction (GRU hidden 96 retrained on the `ch64@0` scores,
`dev/results/gru2026_ch64t_gru96.{pt,onnx}`, chosen on val), paired against the current default pipeline
(`dev/results/final_config_test_2026b.txt`): test mix 81.30 -> 83.43 % (+2.13 [+1.77 .. +2.51], 160 songs better,
31 worse), mix + vocals 82.51 -> 84.16 % (+1.66 [+1.30 .. +2.03]). In the tool from the morning of 2026-11-13 (now
`models/context_gru_2026_11_13.onnx`, replaced by the average below), run by `ultrastar_pitch/context_model.py` as
the last step of the postprocessing (`--no-context-model` switches it off); `prepare` takes the note features from that module, so
training and tool compute them with the same code. Without RMVPE (frame model scores only) it still gains +2.2
points on the validation mix.

Vocals scores (2026-11-13, afternoon; `pipeline_weights_val.py extract --update --sources mix,vocals` added them to
all training songs, `prepare --vocals`, `merge` joins the mix and the mix + vocals arrays of the same songs; GRU 96,
epoch chosen on val; validation mix / mix + vocals): packaged network (mix only, seed 0) 88.77 / 89.81 %, the same
recipe with seed 1 88.67 / 89.81, trained on mix and mix + vocals (seeds 0, 1) 88.72 / 89.88 and 88.80 / 89.92,
mix + vocals only 88.46 / 89.92, mix + vocals with the single sources as further inputs (`prepare --separate`,
frame model and RMVPE of mix and vocals each, seeds 0, 1) - / 89.83 and - / 89.79. So the vocals scores change
nothing beyond the seed noise (about 0.1). What helps is averaging several networks (`ensemble`, mean of the log
softmax, `dev/results/gru2026_vocals_ensemble_{val,test}.txt`): the packaged one + seed 1 + the two mix and
mix + vocals networks, val +0.25 [+0.13 .. +0.36] / +0.29 [+0.17 .. +0.39]; test, applied once: mix 83.43 ->
83.83 % (+0.40 [+0.25 .. +0.55], 118 songs better, 48 worse), mix + vocals 84.16 -> 84.58 % (+0.42 [+0.26 .. +0.57],
116 better, 47 worse); 22 ms instead of 4 ms per song on one CPU thread. In the tool since the afternoon of
2026-11-13: `combine` joins the four networks into one graph with the same inputs and the mean of their log softmax
as output (`ultrastar_pitch/binaries/context_gru_ensemble_2026_11_13.onnx`, 393,776 parameters; same argmax as the
averaged single networks on all 349,770 val and test notes; real pipeline == cache 3041/3041 notes). End-to-end
(`dev/results/pipeline_eval_2026_11_13b.log`, `pipeline_compare_2026_11_13b.txt`): default 83.8 % (mix + vocals
84.6 %), +0.39 [+0.24 .. +0.54] (118 songs better, 48 worse) / +0.40 [+0.22 .. +0.59] over the single network, the
context step alone +2.12 [+1.82 .. +2.42] / +1.66 [+1.37 .. +1.95]; against the 2.0 pipeline with the model of the
1.x versions +4.97 [+4.37 .. +5.60] (177 better, 19 worse) / +4.06 [+3.51 .. +4.61]:

    python dev/context_gru_2026.py ensemble dev/results/gru2026_val_ch64t.npz dev/results/gru2026_val_ch64t_vocals.npz \
        --models pkg=models/context_gru_2026_11_13.onnx s1=dev/results/gru2026_ch64t_gru96_s1.onnx \
        b0=dev/results/gru2026_both_gru96.onnx b1=dev/results/gru2026_both_gru96_s1.onnx --combos pkg,pkg+s1+b0+b1
    python dev/context_gru_2026.py combine ultrastar_pitch/binaries/context_gru_ensemble_2026_11_13.onnx \
        models/context_gru_2026_11_13.onnx dev/results/gru2026_ch64t_gru96_s1.onnx \
        dev/results/gru2026_both_gru96.onnx dev/results/gru2026_both_gru96_s1.onnx
### remaining_errors_val.py
What is left in the errors of the default pipeline (validation songs, cached scores, only the context network
runs): every wrong note sorted by the six single votes (2020 pitchnet, RMVPE, pitchnet_2026_11_13; mix and
vocals), with a control (the label mirrored at the pipeline's class) and a listening list of the strongest label
error candidates (`dev/results/remaining_errors_listen.csv`, one note per song: `..._top30.csv`):

    CUDA_VISIBLE_DEVICES= python dev/remaining_errors_val.py > dev/results/remaining_errors_val.txt

Results (2026-11-14, `dev/results/remaining_errors_val.txt`): the default pipeline misses 9.9 % of the 100,914
validation notes (mix + vocals). For 3.8 % of all notes some single vote hits the label (control with the
mirrored label: 0.9 %), so about 3 points of information are in the models but not used; for 2.8 % all six
votes agree on another class (label error candidates or an error all models share; 74 % of them one or two
semitones away, most frequent in long notes); 3.3 % have a majority or no agreement. The notes that both the
2020 model and RMVPE miss (15.4 % on the mix) are found by pitchnet_2026 in 31.6 % and by the pipeline in
51.6 % of the cases.

With FCPE as an independent seventh vote (`fcpe_val.py`, run first): on the notes where some of the six votes
hits the label, FCPE (vocals) hits it too in 35.7 % (control 2.4 %), so that information is real; on the notes
where all six agree on another class, FCPE follows them in 97.3 % and hears the label in 0.8 % (control 0.6 %;
notes of at least 300 ms: 97.5 % / 0.3 %), and on the majority and unclear notes it hears the label at chance
level. So about 6 % of the notes match no model, also not one trained on other data: label deviations (written
is not what is sung, or a backing voice dominates) or ambiguous notes, not missing information.
### fcpe_val.py
FCPE ([torchfcpe](https://github.com/CNChTu/FCPE), MIT, trained on other data than our labels) as a third pitch
model on the validation songs: its per-frame pitch distribution is mapped onto the RMVPE bin grid and folded to
12 classes per note like RMVPE (separate cache `dev/results/fcpe_cache/`, about 1.3 s per song with the audio
decoding); `ensemble` adds it as a log-linear vote to every source, before and with the note context network:

    python dev/fcpe_val.py extract                     # needs torch + torchfcpe in the environment
    CUDA_VISIBLE_DEVICES= python dev/fcpe_val.py ensemble > dev/results/fcpe_ensemble_val.txt

Results (2026-11-14, val): FCPE alone 78.9 % exact on the vocals (RMVPE 78.6 %, pitchnet_2026 83.0 %), 58.9 % on
the mix (a monophonic model). As an extra vote it never helps: on the vocals only -0.04 [-0.09 .. +0.01] at
weight 0.1 (with the context network), worse with more weight or on the mix (`dev/results/fcpe_ensemble_val.txt`).
### note_hygiene_eval.py
How often the checks of `ultrastar_pitch/note_hygiene.py` fire on a song database (per group of folders) and how much every fix changes; verifies that the fixes leave nothing fixable, are idempotent and keep notes, pitches and start times, and that the octave of unchanged notes is kept. Reads the note files only (no audio, about 30 s for 13,000 files):

    python dev/note_hygiene_eval.py songs --group "A+B=folderA,folderB" --group C=folderC \
        --workers 8 > dev/results/note_hygiene_eval.txt

### ultrastar-scripts/
A local clone of [barbeque-squared's ultrastar-scripts](https://github.com/barbeque-squared/ultrastar-scripts) (MIT), the source of the rules in `note_hygiene.py`; for reference only, not part of this repository.
### gap_xcorr_eval.py
Time offset between notes and audio without pitches: cross-correlation of the RMVPE voicing with the note on/off signal, compared with the pitch based offset; wrong audio and wrong `#BPM` (drift) tests; accuracy of the tool at file / corrected timing (`extract`, `evaluate`, `accuracy`).
### ultrastar_pitch/tuning_correction.py
Tuning offset, pitch class per segment and global transposition; lives in the package now because the pipeline uses it (`python ultrastar_pitch/tuning_correction.py --selftest`).
### export_demucs_vocals.py
Exports the vocals model of the pretrained HTDemucs (`htdemucs_ft`) to ONNX; with `--vocals-only` this is the
optional `htdemucs_vocals.onnx` of the release `models-v1`.
