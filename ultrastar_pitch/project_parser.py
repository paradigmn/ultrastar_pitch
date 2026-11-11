#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          project_parser.py
@brief         parse an USDX note file and the corresponding audio file
               for further processing
@author        paradigm

Supported file format versions (https://github.com/UltraStar-Deluxe/format):

    unversioned  no #VERSION header (legacy files), same timing as v1
    1.x.x        #BPM is implicitly quadrupled -> 1 beat = 15000 / BPM ms
    2.x.x        #BPM is the real beat rate    -> 1 beat = 60000 / BPM ms

All other major versions are rejected with an UnsupportedVersionError.
Every problem found while parsing is reported via a subclass of
ProjectParserError, so callers can simply catch the base class.
"""
import contextlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import PurePosixPath, PureWindowsPath
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .vocal_separator import VocalSeparator, VocalSeparatorError

logger = logging.getLogger(__name__)

# separated vocals, which are saved next to the note file (see ProjectParser.load_audio)
VOCALS_FILE_NAME = "vocals.wav"


# --------------------------------------------------------------------------- #
# exceptions
# --------------------------------------------------------------------------- #
class ProjectParserError(Exception):
    """ base class of all errors raised by the project parser """

    def __init__(self, message: str, line_no: Optional[int] = None):
        self.line_no = line_no
        if line_no is not None:
            message = "line {0}: {1}".format(line_no, message)
        super().__init__(message)


class NoteFileError(ProjectParserError):
    """ the note file can't be read, decoded or written """


class HeaderError(ProjectParserError):
    """ base class for header problems """


class MissingHeaderError(HeaderError):
    """ a header which is required for processing is missing or empty """


class InvalidHeaderError(HeaderError):
    """ a header exists, but its value is malformed """


class UnsupportedVersionError(ProjectParserError):
    """ the #VERSION of the file is not supported by this parser """


class UnsupportedFeatureError(ProjectParserError):
    """ the file uses a (legacy) feature which can't be processed correctly """


class NoteSyntaxError(ProjectParserError):
    """ a note line doesn't follow the file format """


class EmptySongError(ProjectParserError):
    """ the file contains no notes with pitch information (or none is loaded) """


class PitchUpdateError(ProjectParserError):
    """ the new pitches can't be applied to the loaded song """


class AudioError(ProjectParserError):
    """ base class for audio problems """


class UnsafeFileReferenceError(AudioError):
    """ an audio reference is an absolute path (forbidden by the specification) """


class AudioFileNotFoundError(AudioError):
    """ the referenced audio file doesn't exist """


class FFmpegNotFoundError(AudioError):
    """ the ffmpeg executable could not be started """


class AudioConversionError(AudioError):
    """ ffmpeg failed or the converted audio is unusable """


class AudioSeparationError(AudioError):
    """ the vocals couldn't be separated from the audio """


# --------------------------------------------------------------------------- #
# format definitions
# --------------------------------------------------------------------------- #
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
# "#  KEY : value", the key is everything up to the first colon
_HEADER_RE = re.compile(r"^\s*#\s*(?P<key>[^:]*?)\s*:\s*(?P<value>.*?)\s*$")
# v1/legacy allow a comma as decimal separator
_NUMBER_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
# only regular (:) and golden (*) notes carry pitch information
_NOTE_RE = re.compile(
    r"^\s*(?P<type>[:*])"
    r"\s+(?P<start>\d{1,10})"
    r"\s+(?P<duration>\d{1,10})"
    r"\s+(?P<pitch>[+-]?\d{1,9})(?=\s|$)"
)
_VOICE_RE = re.compile(r"^\s*P\s*(?P<voice>\d)\s*$")
# legacy bpm change: "B <beat> <bpm>"
_BPM_CHANGE_RE = re.compile(r"^\s*B\s+\d+(?:[.,]\d+)?\s+\d+(?:[.,]\d+)?\s*$")
_ENCODING_HEADER_RE = re.compile(r"^\s*#\s*ENCODING\s*:", re.IGNORECASE)


class ProjectParser:
    """ parse an usdx note file and the corresponding audio file  """

    SUPPORTED_MAJOR_VERSIONS = (1, 2)
    # newest minor version known per major version (newer ones only trigger a warning)
    _LATEST_MINOR = {1: 1, 2: 0}

    def __init__(self):
        """ init state variables """
        # header of the note file; keys are upper case incl. hash, e.g. "#BPM"
        # (#BPM and #GAP are converted to float)
        self.meta = {}  # type: Dict[str, object]
        # list of dictionarys with singable notes
        self.singable = []  # type: List[dict]
        # buffer for note_file content (one entry per line, incl. line break)
        self.file_buffer = []  # type: List[str]
        # absolute project path derived from note file
        self.proj_dir = ""
        # file format version as tuple or None for unversioned files
        self.version = None  # type: Optional[Tuple[int, int, int]]
        # duration of a single beat in milliseconds
        self.ms_per_beat = None  # type: Optional[float]
        # encoding the note file was read with
        self.encoding = "utf-8"
        # index of the first line after the header
        self._body_start = 0
        # numbers of the voices, which have notes (P1, P2, ...), more than one means a duet
        self.voices = []  # type: List[int]
        # header the audio was taken from by load_audio (#VOCALS, #AUDIO, #MP3, "vocals.wav" or "separated")
        self.audio_source = None  # type: Optional[str]
        # sample rate of the saved vocals.wav (mono, 16 bit), 16 kHz is what the analysis uses
        self.vocals_sample_rate = 16000
        # change the ffmpeg path depending on using script or executable
        if getattr(sys, "frozen", False):
            self.ffmpeg = os.path.join(sys._MEIPASS, "ffmpeg.exe")
        else:
            self.ffmpeg = "ffmpeg"

    # ----------------------------------------------------------------------- #
    # loading
    # ----------------------------------------------------------------------- #
    def _reset(self):
        """ discard everything from a previously loaded file """
        self.meta.clear()
        self.singable.clear()
        self.file_buffer.clear()
        self.proj_dir = ""
        self.version = None
        self.ms_per_beat = None
        self.encoding = "utf-8"
        self._body_start = 0
        self.voices = []
        self.audio_source = None

    @property
    def is_duet(self) -> bool:
        """ True, if notes of more than one voice (P1, P2, ...) are in the song """
        return len(self.voices) > 1

    def load_note_file(self, note_file: str):
        """ load and parse usdx project file """
        self._reset()
        note_file = os.fspath(note_file)
        lines, encoding = self._read_lines(note_file)
        meta, body_start = self._parse_header(lines)
        version = self._parse_version(meta)
        legacy = version is None
        bpm = self._get_number(meta, "#BPM", version, required=True)
        if bpm <= 0:
            raise InvalidHeaderError("header #BPM has to be greater than zero")
        gap = self._get_number(meta, "#GAP", version) or 0.0
        if legacy and meta.get("#RELATIVE", "").lower() == "yes":
            raise UnsupportedFeatureError(
                "relative mode (#RELATIVE:YES) is not supported"
            )
        # v1 and unversioned files implicitly quadruple the bpm value
        ms_per_beat = (15000.0 if version is None or version[0] < 2 else 60000.0) / bpm
        singable, voices = self._parse_body(lines, body_start, legacy, gap, ms_per_beat)
        if not singable:
            raise EmptySongError("the file contains no notes with pitch (':' or '*')")
        # everything went fine -> commit state
        meta["#BPM"] = bpm
        if "#GAP" in meta:
            meta["#GAP"] = gap
        self.meta.update(meta)
        self.singable.extend(singable)
        self.file_buffer.extend(lines)
        self.proj_dir = os.path.dirname(os.path.abspath(note_file))
        self.version = version
        self.ms_per_beat = ms_per_beat
        self.voices = sorted(voices)
        self.encoding = encoding
        self._body_start = body_start

    @staticmethod
    def _read_lines(note_file: str) -> Tuple[List[str], str]:
        """ read the file and split it into lines (CR, LF and CRLF are line breaks) """
        try:
            with open(note_file, "rb") as file:
                raw = file.read()
        except OSError as err:
            raise NoteFileError("can't read '{0}': {1}".format(note_file, err)) from err
        if b"\x00" in raw[:4096] or raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise NoteFileError(
                "'{0}' is not an utf-8 text file (binary or utf-16 content)".format(
                    note_file
                )
            )
        text, encoding = ProjectParser._decode(raw)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        # keep the line breaks, so the file can be written back unchanged
        buffer = [line + "\n" for line in lines[:-1]]
        if lines[-1]:
            buffer.append(lines[-1])
        return buffer, encoding

    @staticmethod
    def _decode(raw: bytes) -> Tuple[str, str]:
        """ decode as utf-8 (with or without bom), fall back to legacy encodings """
        try:
            return raw.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            pass
        # latin-1 never fails and keeps the ascii header readable
        probe = raw.decode("latin-1")
        candidates = []
        for line in probe.split("\n")[:100]:
            if _ENCODING_HEADER_RE.match(line):
                candidates.append(line.split(":", 1)[1].strip().rstrip("\r"))
                break
        candidates.append("cp1252")
        for candidate in candidates:
            try:
                return raw.decode(candidate), candidate.lower()
            except (UnicodeDecodeError, LookupError):
                continue
        return probe, "latin-1"

    @staticmethod
    def _parse_header(lines: List[str]) -> Tuple[Dict[str, object], int]:
        """ read all header lines, returns header dict and index of first body line """
        meta = {}  # type: Dict[str, object]
        body_start = len(lines)
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            if not line.lstrip().startswith("#"):
                body_start = idx
                break
            match = _HEADER_RE.match(line.rstrip("\n"))
            if match is None or not match.group("key"):
                # if it was an important header, a "missing header" error follows
                logger.warning("line %d: ignoring malformed header", idx + 1)
                continue
            key = "#" + match.group("key").upper()
            value = match.group("value")
            if not value:
                # an empty value is equivalent to an absent header
                continue
            if key in meta:
                logger.warning("line %d: duplicate header %s", idx + 1, key)
            meta[key] = value
        return meta, body_start

    def _parse_version(self, meta: dict) -> Optional[Tuple[int, int, int]]:
        """ evaluate #VERSION, None means unversioned file """
        raw = meta.get("#VERSION")
        if raw is None:
            return None
        match = _VERSION_RE.match(raw)
        if match is None:
            raise InvalidHeaderError(
                "header #VERSION '{0}' doesn't match MAJOR.MINOR.PATCH".format(raw)
            )
        version = tuple(int(x) for x in match.groups())
        if version[0] not in self.SUPPORTED_MAJOR_VERSIONS:
            raise UnsupportedVersionError(
                "file format version {0} is not supported (supported major versions: {1})".format(
                    raw, ", ".join(str(x) for x in self.SUPPORTED_MAJOR_VERSIONS)
                )
            )
        if version[1] > self._LATEST_MINOR[version[0]]:
            logger.warning(
                "file format version %s is newer than known, some features may be ignored", raw
            )
        return version

    @staticmethod
    def _get_number(
        meta: dict, key: str, version: Optional[tuple], required: bool = False
    ) -> Optional[float]:
        """ convert a numeric header to float, accepts comma as decimal separator """
        raw = meta.get(key)
        if raw is None:
            if required:
                raise MissingHeaderError("required header {0} is missing".format(key))
            return None
        if not _NUMBER_RE.match(raw):
            raise InvalidHeaderError(
                "header {0} has no valid numeric value: '{1}'".format(key, raw)
            )
        if "," in raw and version is not None and version[0] >= 2:
            logger.warning("header %s uses a decimal comma, forbidden since v2", key)
        value = float(raw.replace(",", "."))
        if not math.isfinite(value):
            raise InvalidHeaderError("header {0} is out of range: '{1}'".format(key, raw))
        return value

    @staticmethod
    def _parse_body(
        lines: List[str], body_start: int, legacy: bool, gap: float, ms_per_beat: float
    ) -> Tuple[List[dict], set]:
        """ collect all notes with pitch information and the numbers of all voices with notes """
        singable = []
        voices = set()
        voice = 1  # without a voice change the notes belong to the first voice
        for idx in range(body_start, len(lines)):
            line = lines[idx].rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            kind = stripped[0]
            if stripped == "E":
                # everything after the end marker has to be ignored
                break
            if kind in ":*":
                match = _NOTE_RE.match(line)
                if match is None:
                    raise NoteSyntaxError(
                        "invalid note, expected '{0} <start> <duration> <pitch> <text>': {1!r}".format(
                            kind, stripped[:60]
                        ),
                        idx + 1,
                    )
                voices.add(voice)
                start = int(match.group("start"))
                duration = int(match.group("duration"))
                # start = gap + start_beat * beat_length
                # end = gap + (start_beat + duration) * beat_length
                singable.append(
                    {
                        "t_start": gap + start * ms_per_beat,
                        "t_end": gap + (start + duration) * ms_per_beat,
                        # pitch is relative to C4 and can be negative, python's modulo is safe
                        "pitch": int(match.group("pitch")) % 12,
                        "voice": voice,
                        "line": idx,
                        "span": match.span("pitch"),
                    }
                )
            elif kind in "FRG":
                # freestyle, rap and golden rap carry no usable pitch, but they belong to a voice
                voices.add(voice)
            elif kind == "-":
                continue  # end of phrase
            elif kind == "P" and _VOICE_RE.match(line):
                voice = int(_VOICE_RE.match(line).group("voice"))
            elif kind == "B" and legacy and _BPM_CHANGE_RE.match(line):
                raise UnsupportedFeatureError(
                    "bpm changes ('B' lines) are not supported", idx + 1
                )
            elif kind == "#":
                logger.warning("line %d: header inside the song body ignored", idx + 1)
            else:
                logger.warning("line %d: ignoring unknown line %r", idx + 1, stripped[:40])
        return singable, voices

    # ----------------------------------------------------------------------- #
    # pitches
    # ----------------------------------------------------------------------- #
    def _require_loaded(self):
        if not self.singable:
            raise EmptySongError("no note file loaded")

    def dump_pitches(self) -> list:
        """ returns singable pitches """
        return [singable["pitch"] for singable in self.singable]

    def update_pitches(self, new_pitches: list):
        """ replace old singable pitches by newly calculated ones """
        if len(new_pitches) != len(self.singable):
            raise PitchUpdateError(
                "pitches can't be updated, array size doesn't match ({0} != {1})".format(
                    len(new_pitches), len(self.singable)
                )
            )
        try:
            converted = [int(pitch) for pitch in new_pitches]
        except (TypeError, ValueError) as err:
            raise PitchUpdateError("new pitches have to be integers: {0}".format(err)) from err
        for singable, new_pitch in zip(self.singable, converted):
            singable["pitch"] = new_pitch

    def save_note_file(self, note_file: str):
        """ save updated note file under a new name """
        self._require_loaded()
        lines = list(self.file_buffer)
        # only the pitch column is replaced, everything else stays untouched
        for singable in self.singable:
            idx = singable["line"]
            begin, end = singable["span"]
            lines[idx] = lines[idx][:begin] + str(int(singable["pitch"])) + lines[idx][end:]
        if self.encoding != "utf-8":
            # the output is always utf-8, a stale #ENCODING header would corrupt it
            for idx in range(self._body_start):
                if _ENCODING_HEADER_RE.match(lines[idx]):
                    lines[idx] = "#ENCODING:UTF-8\n"
        note_file = os.fspath(note_file)
        tmp_path = None
        try:
            # write to a temporary file first, so a failure can't leave a broken output
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(os.path.abspath(note_file)), suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
                file.writelines(lines)
            os.replace(tmp_path, note_file)
            tmp_path = None
        except OSError as err:
            raise NoteFileError("can't write '{0}': {1}".format(note_file, err)) from err
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ----------------------------------------------------------------------- #
    # audio
    # ----------------------------------------------------------------------- #
    def _audio_reference(self) -> Tuple[str, str]:
        """ pick the audio header: vocals first (best for pitch detection), then full song """
        if self.version is not None and self.version[0] >= 2:
            keys = ("#VOCALS", "#AUDIO")  # #MP3 doesn't exist in v2
        else:
            # in v1 #AUDIO supersedes #MP3
            keys = ("#VOCALS", "#AUDIO", "#MP3")
        for key in keys:
            if self.meta.get(key):
                return key, str(self.meta[key])
        raise MissingHeaderError("no audio header found (expected {0})".format(" or ".join(keys)))

    def _find_audio_file(self, key: str, reference: str) -> str:
        """ resolve a file reference relative to the project folder """
        reference = reference.replace("\\", "/")
        if (
            "\x00" in reference
            or PurePosixPath(reference).is_absolute()
            or PureWindowsPath(reference).drive
        ):
            raise UnsafeFileReferenceError(
                "{0} references an absolute path, which is not allowed: '{1}'".format(
                    key, reference
                )
            )
        # linux and mac are case sensitive, but songs are mostly created on windows
        current = self.proj_dir
        for part in PurePosixPath(reference).parts:
            candidate = os.path.join(current, part)
            if not os.path.exists(candidate):
                try:
                    matches = [x for x in os.listdir(current) if x.lower() == part.lower()]
                except OSError:
                    matches = []
                if len(matches) != 1:
                    raise AudioFileNotFoundError(
                        "audio file referenced by {0} not found: '{1}'".format(key, reference)
                    )
                candidate = os.path.join(current, matches[0])
            current = candidate
        if not os.path.isfile(current):
            raise AudioFileNotFoundError(
                "audio file referenced by {0} not found: '{1}'".format(key, reference)
            )
        return current

    def _run_ffmpeg(self, args: List[str], source: str):
        """ run ffmpeg with the given arguments """
        try:
            subprocess.run(
                [self.ffmpeg, "-nostdin", "-v", "error", "-y"] + args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                shell=False,
            )
        except FileNotFoundError as err:
            raise FFmpegNotFoundError(
                "can't execute ffmpeg ('{0}'), is it installed?".format(self.ffmpeg)
            ) from err
        except subprocess.CalledProcessError as err:
            details = (err.stderr or b"").decode("utf-8", "replace").strip()
            raise AudioConversionError(
                "ffmpeg can't convert '{0}': {1}".format(
                    os.path.basename(source), details[-300:] or "unknown error"
                )
            ) from err

    def _to_mono_wav(self, input_args: List[str], source: str, tmp_dir: str, sample_rate: int) -> np.ndarray:
        """ convert the input to mono, 16 bit samples at the requested sample rate """
        wav_path = os.path.join(tmp_dir, "audio.wav")
        self._run_ffmpeg(
            input_args
            + ["-vn", "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le", wav_path],
            source,
        )
        try:
            with wave.open(wav_path, "rb") as wav_file:
                if wav_file.getsampwidth() != 2 or wav_file.getnchannels() != 1:
                    raise AudioConversionError("unexpected wav format after conversion")
                frames = wav_file.readframes(wav_file.getnframes())
        except (wave.Error, EOFError, OSError) as err:
            raise AudioConversionError("converted wav can't be read: {0}".format(err)) from err
        samples = np.frombuffer(frames, dtype="<i2")
        if len(samples) == 0:
            raise AudioConversionError("the audio file contains no samples")
        return samples

    def _convert_audio(self, audio_path: str, sample_rate: int) -> np.ndarray:
        """ convert audio to mono, 16 bit samples via ffmpeg """
        with tempfile.TemporaryDirectory(prefix="ultrastar_pitch_") as tmp_dir:
            return self._to_mono_wav(["-i", audio_path], audio_path, tmp_dir, sample_rate)

    # ----------------------------------------------------------------------- #
    # vocals.wav next to the note file
    # ----------------------------------------------------------------------- #
    def _main_audio(self) -> Tuple[str, str]:
        """ header and path of the audio the vocals are separated from (never the #VOCALS file) """
        self._require_loaded()
        keys = ("#AUDIO",) if self.version is not None and self.version[0] >= 2 else ("#AUDIO", "#MP3")
        for key in keys:
            if self.meta.get(key):
                return key, self._find_audio_file(key, str(self.meta[key]))
        raise MissingHeaderError("no audio header found (expected {0})".format(" or ".join(keys)))

    def _folder_files(self) -> Dict[str, str]:
        """ lower case name -> real name of all files in the folder of the note file """
        try:
            return {name.lower(): name for name in os.listdir(self.proj_dir)}
        except OSError:
            return {}

    def find_vocals_file(self) -> Optional[str]:
        """ path of the saved vocals of this song in the folder of the note file, None if there is none.
            It is vocals.wav, or vocals_<audio name>.wav if vocals.wav belongs to another audio file
            of the folder. A file with a vocals.wav.json next to it is only accepted, if the json
            names the audio file of this song, a vocals.wav without json (made by hand) is trusted """
        self._require_loaded()
        try:
            audio = self._main_audio()[1]
        except ProjectParserError:
            audio = None  # the vocals can't be checked then
        names = self._folder_files()
        candidates = [VOCALS_FILE_NAME]
        if audio:
            candidates.append("vocals_{0}.wav".format(os.path.splitext(os.path.basename(audio))[0].lower()))
        for candidate in candidates:
            if candidate not in names:
                continue
            path = os.path.join(self.proj_dir, names[candidate])
            origin = names.get(candidate + ".json")
            if origin is None:
                if candidate == VOCALS_FILE_NAME:
                    return path
                continue
            try:
                with open(os.path.join(self.proj_dir, origin), encoding="utf-8") as file:
                    info = json.load(file)
                if audio is None or (
                    str(info.get("audio", "")).lower() == os.path.basename(audio).lower()
                    and info.get("size") == os.path.getsize(audio)
                ):
                    return path
            except (OSError, ValueError, AttributeError):
                continue
        return None

    def has_vocal_audio(self, use_vocals_file: bool = True) -> bool:
        """ True, if the vocals are available without a separation: a #VOCALS header or a saved vocals file """
        return bool(self.meta.get("#VOCALS")) or (use_vocals_file and self.find_vocals_file() is not None)

    @contextlib.contextmanager
    def _vocals_lock(self, timeout: float = 60.0):
        """ one process at a time writes vocal files into a folder (parallel analysis processes may
            work on several note files of one folder), locks of crashed processes expire """
        lock = os.path.join(self.proj_dir, ".vocals.lock")
        deadline = time.time() + timeout
        while True:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                break
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(lock) > 120:
                        os.remove(lock)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise OSError("the folder is locked by another process: {0}".format(lock))
                time.sleep(0.1)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                os.remove(lock)

    def _save_vocals(self, raw_path: str, audio_path: str, separator: VocalSeparator):
        """ write the separated vocals (raw float stereo) as vocals.wav (or vocals_<audio name>.wav,
            if vocals.wav belongs to another audio) with a json, which names the audio they come from """
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        dest = None
        try:
            with self._vocals_lock():
                names = self._folder_files()  # after the lock: another process may just have written
                target = VOCALS_FILE_NAME
                if VOCALS_FILE_NAME in names:
                    # vocals.wav of the same audio file is outdated (the audio changed) and gets replaced,
                    # vocals.wav of another audio file stays
                    try:
                        with open(os.path.join(self.proj_dir, names.get(VOCALS_FILE_NAME + ".json", "")), encoding="utf-8") as file:
                            same_audio = str(json.load(file).get("audio", "")).lower() == os.path.basename(audio_path).lower()
                    except (OSError, ValueError, AttributeError):
                        same_audio = False
                    if not same_audio:
                        target = "vocals_{0}.wav".format(stem.lower())
                dest = os.path.join(self.proj_dir, names.get(target, target))
                # the json first: a vocals file without its json would be trusted for any audio of the folder
                with open(dest + ".json.tmp", "w", encoding="utf-8") as file:
                    json.dump({"audio": os.path.basename(audio_path), "size": os.path.getsize(audio_path),
                               "separator": getattr(separator, "model_name", ""),
                               "sample_rate": self.vocals_sample_rate}, file)
                os.replace(dest + ".json.tmp", dest + ".json")
                self._run_ffmpeg(
                    ["-f", "f32le", "-ac", "2", "-ar", str(VocalSeparator.SAMPLE_RATE), "-i", raw_path, "-vn",
                     "-ac", "1", "-ar", str(self.vocals_sample_rate), "-acodec", "pcm_s16le", "-f", "wav",
                     dest + ".tmp"],
                    audio_path,
                )
                os.replace(dest + ".tmp", dest)  # atomic: no other process ever sees a half written file
            logger.info("saved vocals to %s", dest)
        except (OSError, AudioError) as err:  # e.g. read-only folder, the analysis goes on without it
            logger.warning("can't save the vocals next to the note file: %s", err)
            for leftover in ((dest + ".tmp", dest + ".json.tmp") if dest else ()):
                if os.path.isfile(leftover):
                    os.remove(leftover)

    def _separate_and_convert(
        self,
        audio_path: str,
        sample_rate: int,
        separator: VocalSeparator,
        progress: Optional[Callable[[int, int], None]],
        save: bool = False,
    ) -> np.ndarray:
        """ isolate the vocals, then convert them exactly like the unseparated audio """
        rate = VocalSeparator.SAMPLE_RATE
        with tempfile.TemporaryDirectory(prefix="ultrastar_pitch_") as tmp_dir:
            raw_path = os.path.join(tmp_dir, "audio.f32")
            # decode to stereo float samples, the format the separation model works with
            self._run_ffmpeg(
                ["-i", audio_path, "-vn", "-f", "f32le", "-ac", "2", "-ar", str(rate), raw_path],
                audio_path,
            )
            stereo = np.fromfile(raw_path, dtype="<f4")
            if len(stereo) < 2:
                raise AudioConversionError("the audio file contains no samples")
            stereo = stereo[: len(stereo) // 2 * 2].reshape(-1, 2).T
            try:
                vocals = separator.separate(stereo, progress)
            except VocalSeparatorError as err:
                raise AudioSeparationError("vocal separation failed: {0}".format(err)) from err
            del stereo
            np.ascontiguousarray(vocals.T, dtype="<f4").tofile(raw_path)
            del vocals
            samples = self._to_mono_wav(
                ["-f", "f32le", "-ac", "2", "-ar", str(rate), "-i", raw_path],
                audio_path,
                tmp_dir,
                sample_rate,
            )
            if save:
                self._save_vocals(raw_path, audio_path, separator)
            return samples

    def find_audio_file(self) -> Tuple[str, str]:
        """ header (#VOCALS, #AUDIO or #MP3) and absolute path of the audio file the song references """
        self._require_loaded()
        key, reference = self._audio_reference()
        return key, self._find_audio_file(key, reference)

    def load_audio(
        self,
        sample_rate: int = 16000,
        separator: Optional[VocalSeparator] = None,
        progress: Optional[Callable[[int, int], None]] = None,
        use_vocals_file: Optional[bool] = None,
    ) -> np.ndarray:
        """ decode the whole audio of the song to mono, 16 bit samples. Where the audio comes from:
              1. the #VOCALS header
              2. a saved vocals.wav in the folder of the note file (see find_vocals_file)
              3. the vocals separated from #AUDIO / #MP3 by the separator, they are saved as vocals.wav
                 then, so the (slow) separation is only done once
              4. #AUDIO / #MP3 as it is
            use_vocals_file: None = use and write vocals.wav only together with a separator,
            True = also use an existing one without a separator, False = never use or write it.
            progress(done, total) reports the separation. self.audio_source tells the origin
            afterwards (#VOCALS, vocals.wav, separated, #AUDIO or #MP3) """
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate has to be an positive integer value")
        use_file = use_vocals_file if use_vocals_file is not None else separator is not None
        key, audio_path = self.find_audio_file()
        vocals = self.find_vocals_file() if use_file and key != "#VOCALS" else None
        if vocals is not None:
            logger.info("using the saved vocals %s", vocals)
            samples = self._convert_audio(vocals, sample_rate)
            self.audio_source = "vocals.wav"
        elif separator is not None and key != "#VOCALS":
            logger.info("separating vocals from %s", os.path.basename(audio_path))
            samples = self._separate_and_convert(audio_path, sample_rate, separator, progress, save=use_file)
            self.audio_source = "separated"
        else:
            samples = self._convert_audio(audio_path, sample_rate)
            self.audio_source = key
        return samples

    def note_segments(self) -> List[Tuple[float, float]]:
        """ (start, end) of every singable note in seconds """
        self._require_loaded()
        return [(segment["t_start"] / 1000.0, segment["t_end"] / 1000.0) for segment in self.singable]

    def note_bounds(self, sample_rate: int, offset_s: float = 0.0) -> List[Tuple[int, int]]:
        """ (start, end) of every singable note in samples, shifted by offset_s (negative values
            would index from the end of the array, so the bounds are clipped at 0) """
        self._require_loaded()
        bounds = []
        for segment in self.singable:
            start = max(0, int(round((segment["t_start"] / 1000.0 + offset_s) * sample_rate)))
            end = max(start, int(round((segment["t_end"] / 1000.0 + offset_s) * sample_rate)))
            bounds.append((start, end))
        return bounds

    def load_main_audio(self, sample_rate: int = 16000) -> np.ndarray:
        """ decode the mix (#AUDIO / #MP3, never the #VOCALS file or a saved vocals.wav) to mono,
            16 bit samples; self.audio_source tells the header afterwards """
        key, audio_path = self._main_audio()
        samples = self._convert_audio(audio_path, sample_rate)
        self.audio_source = key
        return samples

    def process_audio(
        self,
        sample_rate: int = 16000,
        separator: Optional[VocalSeparator] = None,
        progress: Optional[Callable[[int, int], None]] = None,
        use_vocals_file: Optional[bool] = None,
    ) -> Iterator[np.ndarray]:
        """ convert, resample and divide audio file into audio segments (see load_audio) """
        samples = self.load_audio(sample_rate, separator, progress, use_vocals_file)
        bounds = self.note_bounds(sample_rate)
        n_empty = sum(1 for start, end in bounds if start >= len(samples) or end <= start)
        if n_empty:
            logger.warning(
                "%d of %d notes contain no audio (outside of the audio file or zero length). "
                "check #BPM, #GAP, #VERSION and the audio file!",
                n_empty,
                len(bounds),
            )
        return (samples[start:end] for start, end in bounds)
