#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          detection_pipeline.py
@brief         apply pitch detection with one pipeline
@author        paradigm

How the pitches are determined (end-to-end numbers: 197 evaluation songs, exact pitch class per note, dev/README.md):

  0. Note file hygiene (note_hygiene.py): the structure of the note file is checked (overlapping
     notes, misplaced line breaks, ...) and reported, on request also the style rules of
     barbeque-squared's ultrastar-scripts.
  1. Pitch model (packaged pitchnet_2026_11_13, a frame model): log probabilities of the 12 pitch
     classes per 10 ms frame of the whole song, summed per note. With RMVPE the power spectrum is first
     shifted by the tuning offset of the voice that RMVPE estimates (FRAME_TUNING_CENTS). The model of
     the 1.x versions (pitchnet_2020_12_14, one feature vector per note window) can still be chosen.
  2. RMVPE (optional, rmvpe.onnx): the salience map summed per note and folded to the 12 pitch
     classes (tuning corrected), combined log-linearly with the pitch model (RMVPE weight 0.25 with
     the frame model, 0.6 with the old model: ensemble_weights); without it the pitch model is used alone.
  3. Vocals (optional): with vocals (#VOCALS, a saved vocals.wav or the separation) both models run on
     the mix AND on the vocals and both votes are combined (vocals share 0.5 with the frame model).
  4. Timing check (timing_check.py, needs RMVPE): the RMVPE voicing is cross-correlated with the
     note on/off signal. A clear offset between notes and audio (wrong #GAP) is used for the analysis
     (the file keeps its timing). Notes and audio that hardly fit together and a drifting timing
     (wrong #BPM) are reported as warnings. Measured with the old model: +3.2 points [+2.0 .. +4.4].
  5. Postprocessing:
     a) repetition pooling (repetition_pooling.py): notes of phrases that repeat (same lyrics or
        same rhythm) share their scores, each repetition shifted by its best fitting transposition;
     b) the pseudo key of the song is detected and its pitch class probabilities are added as a
        soft prior to the log scores (StochasticPostprocessor.correct_scores);
     c) with a frame model, the note context network (context_model.py, the average of four small GRUs)
        corrects the result of a) and b) from the neighbouring notes (+2.1 points on the mix, +1.7 with vocals).
     The default pipeline (1. to 5., mix) gets 83.8 % of the notes exact, with vocals 84.6 %; the first
     model of 2.0 (pitchnet_2026_11_11, no tuning correction, no c) 81.3 % / 82.5 %, the 1.x model 78.8 % /
     80.3 %. The old hard correction (correct_pitches) is kept for reference only.
  6. Output: every new pitch class is written in the octave closest to the note in the file (an
     unchanged pitch class leaves the note untouched, the melody keeps its contour; before, every
     note was written as 0 .. 11). The fixes of note_hygiene.py chosen in `fixes` are applied.

check_note_file runs step 0 alone (no audio, no models): it checks a note file in a fraction of a
second and writes the fixed copy <name>_fixed.txt; find_note_files collects the note files of a
folder for it (CLI --check-only, GUI "Check").
"""

import contextlib
import logging
import os
import re
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .project_parser import AudioSeparationError, ProjectParser, ProjectParserError
from .vocal_separator import SeparatorModelError, VocalSeparator
from .audio_preprocessor import AudioPreprocessor
from .pitch_classifier import PitchClassifier
from .stochastic_postprocessor import StochasticPostprocessor
from . import context_model as context_model_mod
from . import note_hygiene
from . import pitch_classifier as pitch_classifier_mod
from . import repetition_pooling
from . import rmvpe as rmvpe_mod
from . import timing_check

logger = logging.getLogger(__name__)

EPS = 1e-6

# --------------------------------------------------------------------------- #
# note file hygiene without the pitch detection
# --------------------------------------------------------------------------- #
FIXED_SUFFIX = "_fixed.txt"
# a .txt file is taken for a note file if its header has #TITLE or #ARTIST (license.txt and the
# like are skipped), utf-16 files as well (the parser rejects them, which is worth reporting)
_NOTE_FILE_TAG_RE = re.compile(rb"^[ \t\xef\xbb\xbf]*#[ \t]*(?:TITLE|ARTIST)[ \t]*:", re.IGNORECASE | re.MULTILINE)


class NoteFileCheck(NamedTuple):
    path: str
    error: Optional[str]                # the tool can't read the file (then there is nothing else)
    parser_warnings: List[str]          # duplicate headers, ignored lines, ...
    issues: List[note_hygiene.Issue]
    fixes: Tuple[str, ...]              # the fixes asked for
    fixed: Dict[str, int]               # what they changed -> count
    written: Optional[str]              # path of the fixed copy (only written if a fix changed something)
    write_error: Optional[str] = None   # the fixed copy couldn't be written

    @property
    def has_errors(self) -> bool:
        return bool(self.error) or any(i.severity == note_hygiene.ERROR for i in self.issues)

    def report(self) -> List[str]:
        """ one line per finding, empty if there is nothing to say """
        lines = []
        if self.error:
            lines.append("error: " + self.error)
        lines += ["parser: " + warning for warning in self.parser_warnings]
        lines += ["note file: " + m for m in note_hygiene.summarize(self.issues, note_hygiene.ERROR, self.fixes)]
        lines += ["style: " + m for m in note_hygiene.summarize(self.issues, note_hygiene.STYLE, self.fixes)]
        if self.fixed:
            lines.append("fixed: {0} -> {1}".format(
                ", ".join("{0} {1}".format(count, what) for what, count in self.fixed.items()),
                self.written or "not written: " + str(self.write_error)))
        return lines


def fixed_file_name(note_file: str) -> str:
    """ <name>_fixed.txt next to the note file """
    return os.path.splitext(note_file)[0] + FIXED_SUFFIX


def find_note_files(folder: str) -> Tuple[List[str], int]:
    """ (note files in the folder and its subfolders, number of other .txt files skipped);
        fixed copies (<name>_fixed.txt) are skipped as well """
    files, skipped = [], 0
    for root, dirs, names in os.walk(folder):
        dirs.sort()
        for name in sorted(names):
            lower = name.lower()
            if not lower.endswith(".txt") or lower.endswith(FIXED_SUFFIX):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "rb") as file:
                    head = file.read(16384)
            except OSError:
                skipped += 1
                continue
            if head.startswith((b"\xff\xfe", b"\xfe\xff")) or _NOTE_FILE_TAG_RE.search(head):
                files.append(path)
            else:
                skipped += 1
    return files, skipped


@contextlib.contextmanager
def _collect_warnings(logger_name: str):
    """ the warnings of a logger as a list instead of the console """
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger(logger_name)
    handler = _Collect(logging.WARNING)
    propagate = log.propagate
    log.addHandler(handler)
    log.propagate = False
    try:
        yield records
    finally:
        log.removeHandler(handler)
        log.propagate = propagate


def check_note_file(note_file: str, style: bool = False, fixes: Sequence[str] = (),
                    output: Optional[str] = None, parser: Optional[ProjectParser] = None) -> NoteFileCheck:
    """ the note file hygiene alone: structure errors (style rules with style), and with fixes a
        fixed copy at output (default <name>_fixed.txt next to the file), written only if a fix
        changed something; the pitches stay as they are """
    parser = parser or ProjectParser()
    fixes = tuple(fixes)
    with _collect_warnings(ProjectParser.__module__) as warnings:
        try:
            parser.load_note_file(note_file)
        except ProjectParserError as err:
            return NoteFileCheck(note_file, str(err), list(warnings), [], fixes, {}, None)
    issues = note_hygiene.check_song(parser, style=style)
    fixed, written, write_error = {}, None, None
    if fixes:
        lines, fixed = note_hygiene.fix_lines(parser.render_lines(), parser.body_start, parser.ms_per_beat, fixes)
        if fixed:
            try:
                parser.save_note_file(output or fixed_file_name(note_file), lines)
                written = output or fixed_file_name(note_file)
            except ProjectParserError as err:
                write_error = str(err)
    return NoteFileCheck(note_file, None, list(warnings), issues, fixes, fixed, written, write_error)


class DetectionPipeline:
    """ pitch detection pipeline with file handling """

    # weights of the ensemble, per pitch model (ensemble_weights): RMVPE against pitchnet in the
    # log-linear ensemble, the separated vocals against the mix when both are available, the soft key
    # prior of the postprocessing. These three are the ones of the old pitchnet (pitchnet_2020_12_14)
    RMVPE_WEIGHT = 0.6
    VOCALS_WEIGHT = 0.35
    PRIOR_WEIGHT = StochasticPostprocessor.PRIOR_WEIGHT
    # the frame model (pitchnet_2026) is much better on its own and wants less RMVPE, more vocals and a
    # weaker key prior; measured on the 249 validation songs of its training (dev/pipeline_weights_val.py:
    # +1.6 points exact on the mix, +1.5 with the vocals over the weights of the old model)
    FRAME_MODEL_WEIGHTS = {"rmvpe": 0.25, "vocals": 0.5, "prior": 0.15}
    # RMVPE voicing threshold for the tuning estimate
    RMVPE_THRESHOLD = 0.15
    # the frame model is run on the spectrum shifted by the RMVPE tuning estimate of the source when the estimate
    # is reliable and at least this far off (cents); None disables the correction. Measured on the validation
    # songs, thresholds 0 .. 20 cents are equal (dev/results/pipeline_tuning_policy_val_ch64.txt): always
    FRAME_TUNING_CENTS = 0.0  # type: Optional[float]
    # weight of the repeated phrases in the postprocessing, 0 disables the repetition pooling
    REPETITION_WEIGHT = 1.0

    def __init__(
        self,
        project_parser: ProjectParser,
        audio_preprocessor: AudioPreprocessor,
        pitch_classifier: PitchClassifier,
        stochastic_postprocessor: StochasticPostprocessor,
        vocal_model: Optional[str] = None,
        rmvpe_model: Optional[str] = None,
        use_rmvpe: bool = True,
        correct_timing: bool = True,
        fixes: Sequence[str] = (),
        style_check: bool = False,
        keep_octave: bool = True,
        use_context_model: bool = True,
        context_model: Optional[str] = None,
    ):
        self.project_parser = project_parser
        self.audio_preprocessor = audio_preprocessor
        self.pitch_classifier = pitch_classifier
        self.stochastic_postprocessor = stochastic_postprocessor
        # path of the onnx model for vocal separation, loaded on first use
        self.vocal_model = vocal_model
        self._separator = None
        self._separator_path = None
        # RMVPE model: None = look it up (environment variable, user data folder); loaded on first use
        self.rmvpe_model = rmvpe_model
        self.use_rmvpe = use_rmvpe
        self._rmvpe = None
        self._rmvpe_path = None
        self._rmvpe_missing_logged = False
        # shift the notes by the measured offset for the analysis (timing_check), needs RMVPE
        self.correct_timing = correct_timing
        # note context network (context_model.py) as the last step of the postprocessing, only with a frame
        # model (it was trained on its scores); context_model: path of another network (default: packaged)
        self.use_context_model = use_context_model
        self.context_model = context_model
        self._context = None
        self._context_path = None
        # what the last transform() used, for the user interface: "pitchnet", "pitchnet+rmvpe" and
        # the audio sources ("mix", "vocals", "mix+vocals")
        self.last_models = ""
        self.last_sources = ""
        # result of the timing check of the last transform() (None without RMVPE)
        self.last_timing = None  # type: Optional[timing_check.TimingCheck]
        # note file hygiene (note_hygiene.py): fixes applied to the written file (names of
        # note_hygiene.FIXES), style rules reported in addition to the structure errors
        self.fixes = tuple(fixes)
        self.style_check = style_check
        # write a new pitch class in the octave closest to the note in the file (False: 0 .. 11)
        self.keep_octave = keep_octave
        # issues found in the last transform() and the fixes done (description -> count)
        self.last_hygiene = []  # type: List[note_hygiene.Issue]
        self.last_fixes = {}  # type: Dict[str, int]
        # voicing per 10 ms frame of the last audio source the frame model (pitchnet_2026) saw, None otherwise
        self.last_frame_voicing = None  # type: Optional[np.ndarray]

    # ----------------------------------------------------------------------- #
    # models
    # ----------------------------------------------------------------------- #
    def ensemble_weights(self) -> Tuple[float, float, float]:
        """ (RMVPE, vocals, key prior) weights for the current pitch classifier """
        if self.pitch_classifier.frame_based:
            w = self.FRAME_MODEL_WEIGHTS
            return w["rmvpe"], w["vocals"], w["prior"]
        return self.RMVPE_WEIGHT, self.VOCALS_WEIGHT, self.PRIOR_WEIGHT

    def _get_separator(self) -> VocalSeparator:
        """ load the vocal separation model on demand (it needs a couple of seconds) """
        if not self.vocal_model:
            raise AudioSeparationError("no vocal separation model selected")
        if self._separator is None or self._separator_path != self.vocal_model:
            try:
                self._separator = VocalSeparator(self.vocal_model)
            except SeparatorModelError as err:
                raise AudioSeparationError(str(err)) from err
            self._separator_path = self.vocal_model
        return self._separator

    def _get_rmvpe(self) -> Optional[rmvpe_mod.Rmvpe]:
        """ the RMVPE session or None, if disabled or the model is not installed """
        if not self.use_rmvpe:
            return None
        path = self.rmvpe_model or rmvpe_mod.default_model_path()
        if not path:
            if not self._rmvpe_missing_logged:
                logger.warning(
                    "rmvpe.onnx not found, using pitchnet only (set $%s or copy the model into %s "
                    "for better results)", rmvpe_mod.RMVPE_MODEL_ENV_VAR, rmvpe_mod.user_model_dir()
                )
                self._rmvpe_missing_logged = True
            return None
        if self._rmvpe is None or self._rmvpe_path != path:
            self._rmvpe = rmvpe_mod.Rmvpe(path)
            self._rmvpe_path = path
        return self._rmvpe

    def _get_context_model(self) -> Optional[context_model_mod.ContextModel]:
        """ the note context network or None (disabled, not a frame model, or the file is missing) """
        if not self.use_context_model or not self.pitch_classifier.frame_based:
            return None
        path = self.context_model or context_model_mod.default_model_path()
        if self._context is None or self._context_path != path:
            if not os.path.exists(path):
                logger.warning("note context network %s not found, postprocessing without it", path)
                return None
            self._context = context_model_mod.ContextModel(path)
            self._context_path = path
        return self._context

    # ----------------------------------------------------------------------- #
    # features per note
    # ----------------------------------------------------------------------- #
    def pitchnet_scores(self, samples: np.ndarray, bounds: List[Tuple[int, int]]) -> np.ndarray:
        """ log probabilities of the 12 pitch classes per note [n, 12]: the frame probabilities
            of a note are summed (as before) and normalised """
        features, lengths = [], []
        for start, end in bounds:
            frames = self.audio_preprocessor.transform(samples[start:end])
            features.append(frames)
            lengths.append(len(frames))
        # predict all block features at once (increases performance a lot!)
        prob = self.pitch_classifier.predict(np.vstack(features))
        scores = np.zeros((len(bounds), 12))
        idx = 0
        for i, length in enumerate(lengths):
            p = prob[idx:idx + length].sum(axis=0)
            scores[i] = np.log(p / max(p.sum(), EPS) + EPS)
            idx += length
        return scores

    @staticmethod
    def salience(rmvpe: rmvpe_mod.Rmvpe, samples: np.ndarray) -> np.ndarray:
        """ RMVPE salience [frames, 360] of 16 bit samples """
        return rmvpe.salience(samples.astype(np.float32) / 32768.0)

    def rmvpe_scores(self, rmvpe: rmvpe_mod.Rmvpe, samples: np.ndarray, segments,
                     salience: Optional[np.ndarray] = None, tuning=None) -> np.ndarray:
        """ log of the RMVPE chroma per note [n, 12] (salience folded to pitch classes); tuning: the
            estimate of rmvpe.estimate_tuning if already known """
        if salience is None:
            salience = self.salience(rmvpe, samples)
        if tuning is None:
            tuning = self.estimate_tuning(salience, segments)
        return np.log(rmvpe_mod.chroma_per_note(salience, segments, tuning.offset_cents) + EPS)

    def estimate_tuning(self, salience: np.ndarray, segments):
        """ RMVPE tuning estimate of the voice (cents) """
        tuning = rmvpe_mod.estimate_tuning(salience, segments, self.RMVPE_THRESHOLD)
        logger.info("RMVPE: tuning offset %+.0f cents (reliable: %s)", tuning.offset_cents, tuning.reliable)
        return tuning

    def frame_tuning(self, tuning) -> float:
        """ the tuning offset (cents) the frame model corrects, 0 for none (FRAME_TUNING_CENTS) """
        if tuning is None or self.FRAME_TUNING_CENTS is None or not tuning.reliable:
            return 0.0
        return float(tuning.offset_cents) if abs(tuning.offset_cents) >= self.FRAME_TUNING_CENTS else 0.0

    def frame_model_scores(self, samples: np.ndarray, segments, tuning_cents: float = 0.0) -> np.ndarray:
        """ log probabilities of the 12 pitch classes per note [n, 12] from the frame model (pitchnet_2026):
            the model runs once over the whole audio, the frame probabilities are summed per note;
            tuning_cents: tuning offset corrected in the spectrum """
        logprob, voicing = self.pitch_classifier.predict_frames(samples, tuning_cents)
        self.last_frame_voicing = voicing
        return self.pitch_classifier.frame_scores_per_note(logprob, segments)

    def note_scores(self, samples: np.ndarray, bounds, segments, salience: Optional[np.ndarray] = None) -> np.ndarray:
        """ combined log scores of one audio source: pitchnet, plus RMVPE if available (salience:
            precomputed RMVPE salience of these samples) """
        rmvpe = self._get_rmvpe()
        tuning = None
        if rmvpe is not None:
            if salience is None:
                salience = self.salience(rmvpe, samples)
            tuning = self.estimate_tuning(salience, segments)
        if self.pitch_classifier.frame_based:
            cents = self.frame_tuning(tuning)
            if cents:
                logger.info("frame model: tuning corrected by %+.0f cents", cents)
            scores = self.frame_model_scores(samples, segments, cents)
        else:
            scores = self.pitchnet_scores(samples, bounds)
        name = "pitchnet" if self.pitch_classifier.model_name == pitch_classifier_mod.MODEL_NAME else \
            os.path.splitext(self.pitch_classifier.model_name)[0]
        if rmvpe is None:
            self.last_models = name
            return scores
        self.last_models = name + "+rmvpe"
        w = self.ensemble_weights()[0]
        return w * self.rmvpe_scores(rmvpe, samples, segments, salience, tuning) + (1 - w) * scores

    # ----------------------------------------------------------------------- #
    # pipeline
    # ----------------------------------------------------------------------- #
    def transform(
        self,
        notes_org: str,
        notes_new: str,
        postproc: bool = True,
        separate_vocals: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
        use_vocals: Optional[bool] = None,
    ) -> (np.ndarray, np.ndarray):
        """ load note file, extract audio, detect pitches, and write to new notes.txt.
            separate_vocals isolates the vocals first (slow) and saves them as vocals.wav next to the
            notes. use_vocals: None = a #VOCALS header or an existing vocals.wav is used when present
            (free), True = like None but separate if there is none (same as separate_vocals), False =
            never. With vocals the models run on the mix and on the vocals and both votes are
            combined. progress(done, total) reports the separation """
        parser = self.project_parser
        parser.load_note_file(notes_org)
        # structure of the note file (and style rules on request), reported before the analysis
        self.last_fixes = {}
        self.last_hygiene = note_hygiene.check_song(parser, style=self.style_check)
        for message in note_hygiene.summarize(self.last_hygiene, note_hygiene.ERROR, self.fixes):
            logger.warning("note file: %s", message)
        for message in note_hygiene.summarize(self.last_hygiene, note_hygiene.STYLE, self.fixes):
            logger.info("style: %s", message)
        pitches_org = parser.dump_pitches()
        segments = parser.note_segments()
        sample_rate = rmvpe_mod.SR
        bounds = parser.note_bounds(sample_rate)
        n_empty = sum(1 for start, end in bounds if end <= start)
        if n_empty:
            logger.warning("%d of %d notes have zero length, check #BPM and #VERSION!", n_empty, len(bounds))
        # the mix (never the saved vocals)
        mix = parser.load_main_audio(sample_rate)
        if any(start >= len(mix) for start, end in bounds):
            logger.warning("some notes lie beyond the end of the audio file, check #GAP and the audio file!")
        # timing check on the RMVPE voicing of the mix; a clear offset is used for the analysis
        self.last_timing = None
        salience = None
        rmvpe = self._get_rmvpe()
        if rmvpe is not None:
            salience = self.salience(rmvpe, mix)
            check = timing_check.check_timing(salience.max(axis=1), segments, correct=self.correct_timing)
            self.last_timing = check
            logger.info("timing: offset %+.0f ms (peak %.2f), applied %+.0f ms, tempo %+.2f %%",
                        1000 * check.offset_s, check.height, 1000 * check.applied_s, 100 * check.tempo)
            for warning in check.warnings:
                logger.warning(warning)
            if check.applied_s:
                segments = [(start + check.applied_s, end + check.applied_s) for start, end in segments]
                bounds = parser.note_bounds(sample_rate, check.applied_s)
        scores = self.note_scores(mix, bounds, segments, salience)
        self.last_sources = "mix"
        # vocals: a #VOCALS header or a saved vocals.wav are used whenever present (free), the
        # separation only on request
        if use_vocals is None:
            use_vocals = separate_vocals or parser.has_vocal_audio()
        if use_vocals:
            separator = None if parser.has_vocal_audio() else self._get_separator()
            vocals = parser.load_audio(sample_rate, separator=separator, progress=progress, use_vocals_file=True)
            if parser.audio_source in ("#VOCALS", "vocals.wav", "separated"):
                logger.info("vocals from %s", parser.audio_source)
                v = self.ensemble_weights()[1]
                scores = (1 - v) * scores + v * self.note_scores(vocals, bounds, segments)
                self.last_sources = "mix+vocals"
        if postproc:
            context = parser.note_context()
            pooled = scores
            if self.REPETITION_WEIGHT > 0:
                logger.info("repeated phrases: %.0f %% of the notes", 100 * repetition_pooling.pooled_fraction(**context))
                pooled = repetition_pooling.pool_repetitions(scores, weight=self.REPETITION_WEIGHT, **context)
            key = self.stochastic_postprocessor.detect_key(pooled.argmax(axis=1).tolist())
            logger.info("Song was written in key: %d", key)
            weight = self.ensemble_weights()[2]
            network = self._get_context_model()
            if network is None:
                pitches_new = self.stochastic_postprocessor.correct_scores(pooled, weight=weight)
            else:
                # the soft key prior as in correct_scores, then the context network corrects the result
                key_logprior = np.log(self.stochastic_postprocessor.key_prior(key))[None].repeat(len(pooled), axis=0)
                decided = pooled + weight * key_logprior
                corrected = network.correct(scores, pooled, key_logprior, decided, [end - start for start, end in segments],
                                            context["beats"], context["lengths"], context["phrases"])
                pitches_new = corrected.argmax(axis=1)
                logger.info("note context network: %d of %d notes changed", int(np.sum(pitches_new != decided.argmax(axis=1))),
                            len(pitches_new))
                self.last_models += "+context"
        else:
            pitches_new = scores.argmax(axis=1)
        pitches_new = [int(p) for p in pitches_new]
        parser.update_pitches(pitches_new, keep_octave=self.keep_octave)
        lines = parser.render_lines()
        if self.fixes:
            lines, self.last_fixes = note_hygiene.fix_lines(lines, parser.body_start, parser.ms_per_beat, self.fixes)
            for what, count in self.last_fixes.items():
                logger.info("fixed: %d %s", count, what)
        parser.save_note_file(notes_new, lines)
        return np.array(pitches_org), np.array(pitches_new)
