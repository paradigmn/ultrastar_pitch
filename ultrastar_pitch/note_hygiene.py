#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          note_hygiene.py
@brief         check the structure of a note file and optionally fix it (everything but the pitches)
@author        paradigm

The pitch detection only replaces the pitch column. This module looks at the rest of the file.

Checks (check_lines, check_header):
  structure, always: notes that start before the end of the previous note or before the previous
      note, line breaks before the end of the previous note or after the start of the next note,
      two line breaks in a row, a line break without a note before it, notes without length or
      text, lines after the end marker E, a missing E. The game shows such files wrongly or not
      at all, the CLI and the GUI report them as warnings;
  style, on request: the conventions of barbeque-squared's ultrastar-scripts (inspired by Yass and
      the requirements of My Little Karaoke): line breaks at fixed positions in the pause, no note
      on the beat of its line break, word boundaries as a space at the start of a syllable, golden
      notes about 1/17 of the beats, #YEAR, #LANGUAGE and #GENRE set; and own ones: #VERSION set,
      UTF-8, the files of #COVER, #BACKGROUND, #VIDEO and #INSTRUMENTAL exist.

Fixes (fix_lines, opt-in, applied to the written file only, in this order):
  "structure"   own rules: overlapping notes are shortened to the start of the next note, line
                breaks are moved into the pause between their notes, double line breaks and line
                breaks without a note before them are removed, lines after E are removed, a
                missing E is added. Notes out of order and notes without length or text are only
                reported;
  "linebreaks"  every line break between two notes is put where barbeque's fix-linebreaks puts it
                (optimal_linebreak);
  "whitespace"  barbeque's fix-whitespace: a space at the end of a syllable is moved to the start of
                the next syllable, spaces at the end of a line are removed.

Measured on 13,422 readable note files (dev/note_hygiene_eval.py, results/note_hygiene_eval.txt;
songs/: 2,395 by barbeque and My Little Karaoke / 10,482 from usdb / 545 others), share of the files:
  - structure: 0.0 to 3.3 % per check and group, e.g. overlapping notes 0.8 / 0.0 / 1.1 %, line
    break before the end of the previous note 3.3 / 0.0 / 1.1 %;
  - style: line breaks not at barbeque's position: 9.7 / 5.0 / 70.9 % of the checked line breaks
    (usdb follows the same rules); spaces at the end of syllables in 2.7 / 100 / 2.6 % of the files
    (usdb writes them that way, the format specification leaves it open); golden notes outside
    1/34 .. 2/17 of the beats or none at all in 31 / 39 / 27 %; #GENRE missing in 17 / 43 / 45 %;
  - fixes: afterwards no fixable issue is left, a second run changes nothing, every fixed file is
    read again with the same notes, pitches and start times. The structure fix changes 5.0 / 0.0 /
    3.5 % of the files, linebreaks 44 / 42 / 91 %, whitespace 2.9 / 100 / 2.8 %.

Credits and license: the line break rules (optimal_linebreak), the whitespace fix and the style
checks are adapted from ultrastar-scripts by barbeque-squared
(https://github.com/barbeque-squared/ultrastar-scripts: check.py, fix_linebreaks.py,
fix_whitespace.py); the adapted parts are marked with "adapted from ultrastar-scripts". They are
used under the MIT license of ultrastar-scripts (see also THIRD_PARTY_NOTICES.md):

    Copyright (c) 2020 barbeque.squared@gmail.com

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
import re
from collections import Counter, OrderedDict
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

FIXES = ("structure", "linebreaks", "whitespace")
ERROR = "error"
STYLE = "style"

# golden notes (adapted from ultrastar-scripts, check.py): ideally 1/17 of the beats are golden
# (8000 points for the normal notes and 1000 for the golden ones, which count twice). The exact
# value is met by almost no file (97 to 99 %), so only a share outside this band or no golden
# note at all is reported
GOLDEN_IDEAL = 1 / 17
GOLDEN_MIN = 1 / 34
GOLDEN_MAX = 2 / 17
YEAR_MIN, YEAR_MAX = 1900, 2100
# files of the header checked in the style report (the audio is checked by the parser)
FILE_HEADERS = ("#COVER", "#BACKGROUND", "#VIDEO", "#INSTRUMENTAL")
# line numbers listed per kind of issue in summarize
MAX_LINES_SHOWN = 5

# kind -> (severity, message); the order is the order of the report
KINDS = OrderedDict([
    ("overlap", (ERROR, "note starts before the end of the previous note")),
    ("order", (ERROR, "note starts before the previous note (notes out of order)")),
    ("break_early", (ERROR, "line break before the end of the previous note")),
    ("break_late", (ERROR, "line break after the start of the next note")),
    ("double_break", (ERROR, "two line breaks in a row")),
    ("break_without_note", (ERROR, "line break without a note before it")),
    ("zero_length", (ERROR, "note without length")),
    ("no_text", (ERROR, "note without text")),
    ("after_end", (ERROR, "lines after the end marker E (ignored by the game)")),
    ("no_end", (ERROR, "the end marker E is missing")),
    ("break_position", (STYLE, "line break not where barbeque's rules put it")),
    ("note_at_break", (STYLE, "note starts on the beat of the line break before it")),
    ("trailing_space", (STYLE, "syllable ends with a space (barbeque: a space at the start of the next syllable)")),
    ("golden", (STYLE, "golden notes")),
    ("no_year", (STYLE, "#YEAR is not set")),
    ("year", (STYLE, "#YEAR is not a plausible year")),
    ("no_language", (STYLE, "#LANGUAGE is not set")),
    ("no_genre", (STYLE, "#GENRE is not set")),
    ("no_version", (STYLE, "#VERSION is not set (legacy file, required by the format since 1.0.0)")),
    ("encoding", (STYLE, "the file is not UTF-8 (the output is written as UTF-8)")),
    ("file_missing", (STYLE, "file not found")),
])

_NOTE_RE = re.compile(
    r"^\s*(?P<type>[:*FRG])\s+(?P<start>-?\d+)\s+(?P<length>-?\d+)\s+(?P<pitch>[+-]?\d+)(?=\s|$)"
)
_BREAK_RE = re.compile(r"^\s*-\s*(?P<beat>-?\d+)?")
_VOICE_RE = re.compile(r"^\s*P\s*(?P<voice>\d+)\s*$")


class Issue(NamedTuple):
    kind: str                   # key of KINDS
    severity: str               # ERROR or STYLE
    line: Optional[int]         # line number in the file (1 = first line), None for the whole file
    message: str
    fix: Optional[str]          # the fix that resolves it (one of FIXES) or None


def _issue(kind: str, line_idx: Optional[int], fix: Optional[str] = None, message: Optional[str] = None) -> Issue:
    severity, text = KINDS[kind]
    return Issue(kind, severity, None if line_idx is None else line_idx + 1, message or text, fix)


# --------------------------------------------------------------------------- #
# the body of a note file as elements
# --------------------------------------------------------------------------- #
class _Note:
    kind = "note"

    def __init__(self, idx: int, line: str, match):
        self.idx = idx
        self.line = line
        self.match = match
        self.type = match.group("type")
        self.start = int(match.group("start"))
        self.length = int(match.group("length"))
        rest = line[match.end("pitch"):]
        self.sep = rest[:1] or " "
        self.text = rest[1:]
        self._text = self.text
        self._length = self.length

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def changed(self) -> bool:
        return self.text != self._text or self.length != self._length

    def render(self) -> str:
        line = self.line
        if self.text != self._text:
            head = line[:self.match.end("pitch")]
            line = head + (self.sep + self.text if self.text else "")
        if self.length != self._length:
            line = line[:self.match.start("length")] + str(self.length) + line[self.match.end("length"):]
        return line


class _Break:
    kind = "break"

    def __init__(self, idx: int, line: str, match):
        self.idx = idx
        self.line = line
        self.match = match
        self.beat = int(match.group("beat")) if match.group("beat") is not None else None
        self._beat = self.beat
        self.removed = False

    @property
    def changed(self) -> bool:
        return self.beat != self._beat

    def render(self) -> str:
        if self._beat is None:
            return "- {0}".format(self.beat)
        return self.line[:self.match.start("beat")] + str(self.beat) + self.line[self.match.end("beat"):]


class _Block:
    """ the notes and line breaks of one voice between two voice changes (P1, P2, ...) """

    def __init__(self, voice: int):
        self.voice = voice
        self.elements = []  # type: List[object]


def _scan(lines: Sequence[str], body_start: int) -> Tuple[List[_Block], Optional[int], List[int]]:
    """ (blocks, index of the line with E or None, indices of the non-empty lines after E).
        Lines that are no notes, line breaks or voice changes are skipped (the parser reports them) """
    blocks = [_Block(1)]
    end = None
    after_end = []
    for idx in range(body_start, len(lines)):
        line = lines[idx].rstrip("\r\n")
        stripped = line.strip()
        if end is not None:
            if stripped:
                after_end.append(idx)
            continue
        if not stripped:
            continue
        if stripped == "E":
            end = idx
            continue
        kind = stripped[0]
        if kind in ":*FRG":
            match = _NOTE_RE.match(line)
            if match is not None:
                blocks[-1].elements.append(_Note(idx, line, match))
        elif kind == "-":
            blocks[-1].elements.append(_Break(idx, line, _BREAK_RE.match(line)))
        elif kind == "P":
            match = _VOICE_RE.match(line)
            if match is not None:
                blocks.append(_Block(int(match.group("voice"))))
    return blocks, end, after_end


# --------------------------------------------------------------------------- #
# line breaks
# --------------------------------------------------------------------------- #
def optimal_linebreak(prev_end: int, start: int, ms_per_beat: float) -> int:
    """ beat of the line break in the pause from prev_end (end of the last note of a line) to start
        (first note of the next line).
        Adapted from ultrastar-scripts (barbeque-squared, MIT), fix_linebreaks.py _optimal_linebreak.
        Changes: the seconds come from the beat duration, so the rules hold for every #VERSION (the
        original reads #BPM in the meaning of version 1); a pause of one beat gets the line break at
        its start (the original puts it on the start beat of the next note, which its own check
        reports) """
    pause = start - prev_end
    seconds = pause * ms_per_beat / 1000.0
    if 2 <= pause <= 8:
        return start - 2
    if 9 <= pause <= 12:
        return start - 3
    if 13 <= pause <= 16:
        return start - 4
    if seconds > 3.96:      # more than four seconds (0.066 minutes): two seconds after the end
        return prev_end + round(2000.0 / ms_per_beat)
    if seconds > 1.98:      # more than two seconds (0.033 minutes): one second after the end
        return prev_end + math.floor(1000.0 / ms_per_beat)
    if pause > 16:          # more than 16 beats, but less than two seconds
        return prev_end + 10
    return prev_end         # pause of 0 or 1 beat


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_lines(lines: Sequence[str], body_start: int, ms_per_beat: float, style: bool = False) -> List[Issue]:
    """ structure (and with style the style rules) of the body of a note file. lines: the lines of
        the file, body_start: index of the first line after the header, ms_per_beat: duration of a
        beat (ProjectParser.ms_per_beat) """
    issues = []
    blocks, end, after_end = _scan(lines, body_start)
    beats = Counter()   # voice -> beats of all notes but freestyle
    golden = Counter()  # voice -> beats of golden notes
    for block in blocks:
        prev = None        # previous note of the block
        brk = None         # line break after prev
        early = None       # index of the break_early issue of brk
        for el in block.elements:
            if el.kind == "break":
                if brk is not None:
                    issues.append(_issue("double_break", el.idx, "structure"))
                elif prev is None:
                    issues.append(_issue("break_without_note", el.idx, "structure"))
                else:
                    brk = el
                    if el.beat is not None and el.beat < prev.end:
                        early = len(issues)
                        issues.append(_issue("break_early", el.idx, "structure"))
                continue
            # a note
            if el.type != "F":
                beats[block.voice] += el.length
                if el.type in "*G":
                    golden[block.voice] += el.length
            if el.length <= 0:
                issues.append(_issue("zero_length", el.idx))
            if not el.text:
                issues.append(_issue("no_text", el.idx))
            if prev is not None:
                # the structure fix can shorten the previous note only if this one starts after it, a
                # line break between them can't be placed otherwise either
                fixable = "structure" if el.start > prev.start else None
                if el.start < prev.start:
                    issues.append(_issue("order", el.idx))
                elif el.start < prev.end:
                    issues.append(_issue("overlap", el.idx, fixable))
                if early is not None and fixable is None:
                    issues[early] = issues[early]._replace(fix=None)
                if brk is not None and brk.beat is not None:
                    pause = el.start - prev.end
                    if el.start < brk.beat:
                        issues.append(_issue("break_late", brk.idx, fixable))
                    elif style and pause >= 0 and brk.beat >= prev.end:
                        # adapted from ultrastar-scripts (check.py): position of the line break; a note on
                        # the beat of its line break is only reported if the pause leaves room for it
                        if pause >= 2 and brk.beat != optimal_linebreak(prev.end, el.start, ms_per_beat):
                            issues.append(_issue("break_position", brk.idx, "linebreaks"))
                        if pause >= 1 and el.start == brk.beat:
                            issues.append(_issue("note_at_break", el.idx, "linebreaks"))
                elif style and brk is None and prev.text.strip() and prev.text != prev.text.rstrip():
                    # adapted from ultrastar-scripts (fix_whitespace.py)
                    issues.append(_issue("trailing_space", prev.idx, "whitespace"))
            prev = el
            brk = None
            early = None
    if end is None:
        issues.append(_issue("no_end", None, "structure"))
    elif after_end:
        issues.append(_issue("after_end", after_end[0], "structure",
                             "{0} line(s) after the end marker E (ignored by the game)".format(len(after_end))))
    if style:
        duet = len([voice for voice in beats if beats[voice] > 0]) > 1
        for voice in sorted(beats):
            if beats[voice] <= 0:
                continue
            share = golden[voice] / beats[voice]
            if golden[voice] == 0 or not GOLDEN_MIN <= share <= GOLDEN_MAX:
                issues.append(_issue("golden", None, None, "{0}golden notes are {1:.1f} % of the beats ({2} of {3}), "
                                     "barbeque's rule: about 1/17 = {4:.1f} % ({5} beats)".format(
                                         "P{0}: ".format(voice) if duet else "", 100 * share, golden[voice],
                                         beats[voice], 100 * GOLDEN_IDEAL, round(beats[voice] * GOLDEN_IDEAL))))
    return issues


def check_header(meta: Dict[str, object], version: Optional[tuple], encoding: str = "utf-8",
                 find_file: Optional[Callable[[str], object]] = None) -> List[Issue]:
    """ style of the header: meta and version as ProjectParser.meta / .version; find_file(key)
        resolves the file of a header and raises an exception if it doesn't exist """
    issues = []
    # adapted from ultrastar-scripts (check.py): #YEAR, #LANGUAGE, #GENRE
    year = meta.get("#YEAR")
    if year is None:
        issues.append(_issue("no_year", None))
    elif not (str(year).isdigit() and len(str(year)) == 4 and YEAR_MIN <= int(str(year)) <= YEAR_MAX):
        issues.append(_issue("year", None, None, "#YEAR is not a plausible four-digit year: {0!r}".format(year)))
    if not meta.get("#LANGUAGE"):
        issues.append(_issue("no_language", None))
    if not meta.get("#GENRE"):
        issues.append(_issue("no_genre", None))
    # own
    if version is None:
        issues.append(_issue("no_version", None))
    if encoding.lower() not in ("utf-8", "utf8"):
        issues.append(_issue("encoding", None, None,
                             "the file is encoded as {0}, not UTF-8 (the output is written as UTF-8)".format(encoding)))
    if find_file is not None:
        for key in FILE_HEADERS:
            if meta.get(key):
                try:
                    find_file(key)
                except Exception:  # noqa: BLE001 - any error means: the file can't be used
                    issues.append(_issue("file_missing", None, None,
                                         "{0}: file not found: {1}".format(key, meta[key])))
    return issues


def check_song(parser, style: bool = False) -> List[Issue]:
    """ all checks of a loaded ProjectParser (structure, with style also the style rules) """
    issues = check_lines(parser.file_buffer, parser.body_start, parser.ms_per_beat, style)
    if style:
        issues += check_header(parser.meta, parser.version, parser.encoding, parser.find_referenced_file)
    return issues


def summarize(issues: Iterable[Issue], severity: str, applied: Iterable[str] = ()) -> List[str]:
    """ one line per kind of issue and severity with the number and the first line numbers;
        applied: the fixes used for the output """
    applied = set(applied)
    groups = OrderedDict()  # type: Dict[tuple, List[Issue]]
    order = list(KINDS)
    for issue in sorted((i for i in issues if i.severity == severity), key=lambda i: order.index(i.kind)):
        groups.setdefault((issue.kind, issue.message), []).append(issue)
    lines = []
    for (kind, message), group in groups.items():
        numbers = [i.line for i in group if i.line is not None]
        where = ""
        if numbers:
            shown = ", ".join(str(n) for n in numbers[:MAX_LINES_SHOWN])
            more = ", ..." if len(numbers) > MAX_LINES_SHOWN else ""
            where = " ({0}{1} {2}{3})".format("{0}x, ".format(len(group)) if len(group) > 1 else "",
                                              "lines" if len(numbers) > 1 else "line", shown, more)
        fixes = {i.fix for i in group}
        if fixes == {None}:
            hint = ""
        elif None not in fixes and fixes <= applied:
            hint = " - fixed in the output"
        else:
            hint = " - fix: " + ", ".join(sorted(f for f in fixes if f))
        lines.append(message + where + hint)
    return lines


# --------------------------------------------------------------------------- #
# fixes
# --------------------------------------------------------------------------- #
def parse_fixes(text: str) -> Tuple[str, ...]:
    """ "structure,whitespace" or "all" -> the fixes in the order of FIXES """
    names = {name.strip().lower() for name in text.split(",") if name.strip()}
    if "all" in names:
        return FIXES
    unknown = names - set(FIXES)
    if unknown:
        raise ValueError("unknown fix {0}, expected {1} or all".format(
            ", ".join(sorted(unknown)), ", ".join(FIXES)))
    return tuple(name for name in FIXES if name in names)


def _move_break(brk: _Break, prev_end: int, start: Optional[int]) -> bool:
    """ structure: put the line break into the pause from prev_end to start (start None: no next
        note); False if nothing changed """
    if start is not None and prev_end > start:
        return False  # notes overlap, nothing sensible to do
    beat = brk.beat
    if beat is None or beat < prev_end:
        beat = prev_end
    elif start is not None and beat > start:
        beat = max(prev_end, start - 1)
    if beat == brk.beat:
        return False
    brk.beat = beat
    return True


def fix_lines(lines: Sequence[str], body_start: int, ms_per_beat: float,
              fixes: Iterable[str]) -> Tuple[List[str], Dict[str, int]]:
    """ (fixed lines, what was done -> count). lines as in check_lines; fixes: names of FIXES """
    fixes = set(fixes)
    unknown = fixes - set(FIXES)
    if unknown:
        raise ValueError("unknown fix {0}, expected {1}".format(", ".join(sorted(unknown)), ", ".join(FIXES)))
    blocks, end, after_end = _scan(lines, body_start)
    done = Counter()
    lines = list(lines)
    if "structure" in fixes:
        for block in blocks:
            prev = None
            brk = None
            for el in block.elements:
                if el.kind == "break":
                    if brk is not None or prev is None:
                        el.removed = True
                        done["line breaks removed (double or without a note before them)"] += 1
                    else:
                        brk = el
                    continue
                if prev is not None and prev.start < el.start < prev.end:
                    prev.length = el.start - prev.start
                    done["overlapping notes shortened"] += 1
                if prev is not None and brk is not None and _move_break(brk, prev.end, el.start):
                    done["line breaks moved into the pause"] += 1
                prev = el
                brk = None
            if prev is not None and brk is not None and _move_break(brk, prev.end, None):
                done["line breaks moved into the pause"] += 1
    if "linebreaks" in fixes:
        for block in blocks:
            prev = None
            brk = None
            for el in block.elements:
                if el.kind == "break":
                    if not el.removed and brk is None and prev is not None:
                        brk = el
                    continue
                if prev is not None and brk is not None and el.start >= prev.end:
                    beat = optimal_linebreak(prev.end, el.start, ms_per_beat)
                    if beat != brk.beat:
                        brk.beat = beat
                        done["line breaks moved (barbeque's rules)"] += 1
                prev = el
                brk = None
    if "whitespace" in fixes:
        # adapted from ultrastar-scripts (fix_whitespace.py): a space at the end of a syllable goes to
        # the start of the next one, spaces at the end of a line are removed. Changes: a syllable that
        # is only a space stays as it is, no second space is added to a syllable that starts with one
        for block in blocks:
            prev = None
            for el in block.elements:
                if el.kind == "break":
                    if not el.removed and prev is not None:
                        if prev.text.strip() and prev.text != prev.text.rstrip():
                            prev.text = prev.text.rstrip()
                            done["spaces removed at the end of a line"] += 1
                        prev = None
                    continue
                if prev is not None and prev.text.strip() and prev.text != prev.text.rstrip():
                    prev.text = prev.text.rstrip()
                    if not el.text[:1].isspace():
                        el.text = " " + el.text
                    done["spaces moved to the start of the next syllable"] += 1
                prev = el
            if prev is not None and prev.text.strip() and prev.text != prev.text.rstrip():
                prev.text = prev.text.rstrip()
                done["spaces removed at the end of a line"] += 1
    # write the lines
    elements = {el.idx: el for block in blocks for el in block.elements}
    out = []
    for idx, line in enumerate(lines):
        if "structure" in fixes and end is not None and idx > end:
            continue
        el = elements.get(idx)
        if el is not None and el.kind == "break" and el.removed:
            continue
        if el is not None and el.changed:
            newline = line[len(line.rstrip("\r\n")):]
            line = el.render() + newline
        out.append(line)
    if "structure" in fixes:
        if end is not None and after_end:
            done["lines after the end marker E removed"] += len(after_end)
        if end is None:
            if out and not out[-1].endswith("\n"):
                out[-1] += "\n"
            out.append("E\n")
            done["end marker E added"] += 1
    return out, dict(done)


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #
def _selftest():
    """ the checks and fixes on a small example """
    text = (
        "#TITLE:Test\n#ARTIST:Test\n#BPM:300\n#GAP:0\n"
        ": 0 4 0 Hel\n"
        ": 4 4 2 lo \n"          # space at the end of a syllable
        ": 8 4 4 world \n"       # space at the end of a line
        "- 12\n"                 # pause 1 beat: line break ok
        ": 13 4 0 a\n"
        ": 15 4 0 b\n"           # overlaps the note before
        "- 18\n"                 # before the end of the previous note (19)
        "- 30\n"                 # double line break
        ": 30 2 0 c\n"           # note on the beat of its line break, pause 11 beats
        "* 40 4 0 d\n"
        "E\n"
        "comment\n"
    )
    lines = text.splitlines(keepends=True)
    issues = check_lines(lines, 4, 50.0, style=True)
    kinds = Counter(i.kind for i in issues)
    assert kinds == Counter({"trailing_space": 1, "overlap": 1, "break_early": 1, "double_break": 1,
                             "after_end": 1, "golden": 1}), kinds
    fixed, done = fix_lines(lines, 4, 50.0, FIXES)
    expected = (
        "#TITLE:Test\n#ARTIST:Test\n#BPM:300\n#GAP:0\n"
        ": 0 4 0 Hel\n: 4 4 2 lo\n: 8 4 4  world\n- 12\n"
        ": 13 2 0 a\n: 15 4 0 b\n- 27\n: 30 2 0 c\n* 40 4 0 d\nE\n"
    )
    assert "".join(fixed) == expected, "".join(fixed)
    assert done == {"overlapping notes shortened": 1, "line breaks removed (double or without a note before them)": 1,
                    "line breaks moved into the pause": 1, "line breaks moved (barbeque's rules)": 1,
                    "spaces moved to the start of the next syllable": 1, "spaces removed at the end of a line": 1,
                    "lines after the end marker E removed": 1}, done
    # nothing fixable is left and a second run changes nothing
    assert all(i.fix is None for i in check_lines(fixed, 4, 50.0, style=True))
    assert fix_lines(fixed, 4, 50.0, FIXES) == (fixed, {})
    # barbeque's line break rules (v1, #BPM 300 -> 50 ms per beat)
    assert optimal_linebreak(100, 105, 50.0) == 103       # 2-8 beats: 2 before
    assert optimal_linebreak(100, 112, 50.0) == 109       # 9-12 beats: 3 before
    assert optimal_linebreak(100, 116, 50.0) == 112       # 13-16 beats: 4 before
    assert optimal_linebreak(100, 130, 50.0) == 110       # > 16 beats, < 2 s: 10 after
    assert optimal_linebreak(100, 150, 50.0) == 120       # > 2 s: 1 s after
    assert optimal_linebreak(100, 200, 50.0) == 140       # > 4 s: 2 s after
    # a missing E is added
    fixed, done = fix_lines(lines[:-2], 4, 50.0, ("structure",))
    assert fixed[-1] == "E\n" and done["end marker E added"] == 1
    print("selftest ok")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("usage: python ultrastar_pitch/note_hygiene.py --selftest")
