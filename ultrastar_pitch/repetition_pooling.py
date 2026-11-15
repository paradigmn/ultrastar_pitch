#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file          repetition_pooling.py
@brief         share the evidence of repeated phrases (chorus, repeated lines) between their notes
@author        paradigm

Songs repeat lines: the chorus, repeated verses, call and response. A phrase with the same lyrics
or the same rhythm as another phrase is very likely sung to the same melody, possibly
transposed (key change in the last chorus). Measured on the labels of 197 songs: 53 % of all notes
lie in a phrase whose lyrics occur again, 86 % of the note pairs at the same position of two such
phrases have the same pitch class, 93 % after the best transposition of the whole phrase.

pool_repetitions adds to the log scores of every note the mean log scores of the notes at the same
position in its partner phrases (same lyrics or same rhythm), each partner shifted by the
transposition that fits the audio evidence of both phrases best. A short note that the models
can't resolve gets the evidence of the longer or clearer sung repetitions this way.

Measured (197 songs, mix, timing of the file, see dev/README.md): together with the soft key
prior 75.3 % exact instead of 73.6 % (+1.6 points [+1.2 .. +2.0], paired per song), notes shorter
than 60 ms 56.1 % instead of 53.6 %.
"""

import itertools
import re
from typing import Dict, List, Sequence

import numpy as np

# minimum number of notes of a phrase to be matched by lyrics resp. by rhythm (shorter phrases
# repeat by chance)
MIN_NOTES_TEXT = 3
MIN_NOTES_RHYTHM = 4
# minimum number of real syllables (not empty, not "~") of a phrase matched by lyrics
MIN_SYLLABLES = 2

_NON_WORD_RE = re.compile(r"[^\w~]+")


def normalize_syllable(text: str) -> str:
    """ lower case without spaces and punctuation, "~" (continued syllable) is kept """
    return _NON_WORD_RE.sub("", text.lower())


def phrase_groups(phrases: Sequence[int], texts: Sequence[str], beats: Sequence[int],
                  lengths: Sequence[int], kind: str) -> List[List[np.ndarray]]:
    """ groups of phrases that repeat, every phrase as an array of note indices. kind "text":
        same syllables, "rhythm": same relative onsets and lengths (in beats) """
    phrases = np.asarray(phrases)
    groups = {}  # type: Dict[tuple, List[np.ndarray]]
    for phrase in np.unique(phrases):
        idx = np.flatnonzero(phrases == phrase)
        if kind == "text":
            if len(idx) < MIN_NOTES_TEXT:
                continue
            key = tuple(normalize_syllable(texts[i]) for i in idx)
            if sum(1 for k in key if k and k != "~") < MIN_SYLLABLES:
                continue
        elif kind == "rhythm":
            if len(idx) < MIN_NOTES_RHYTHM:
                continue
            first = beats[idx[0]]
            key = tuple((beats[i] - first, lengths[i]) for i in idx)
        else:
            raise ValueError("unknown kind {0!r}, expected 'text' or 'rhythm'".format(kind))
        groups.setdefault(key, []).append(idx)
    return [occurrences for occurrences in groups.values() if len(occurrences) > 1]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def pool_repetitions(scores: np.ndarray, phrases: Sequence[int], texts: Sequence[str],
                     beats: Sequence[int], lengths: Sequence[int], weight: float = 1.0,
                     kinds: Sequence[str] = ("text", "rhythm"), rotate: bool = True) -> np.ndarray:
    """ scores [n_notes, 12] log probabilities; returns the pooled scores: every note of a
        repeated phrase gets weight * mean of the scores of the same note position in its partner
        phrases. With rotate every partner is shifted by the transposition r that maximises
        sum over the notes of P_a(c) * P_b(c - r) (P = softmax of the scores), so transposed
        repetitions (key change) help as well. Notes without a partner keep their scores """
    scores = np.asarray(scores, dtype=np.float64)
    if weight == 0 or len(scores) == 0:
        return scores.copy()
    # every ordered pair of phrases once, even if they match by lyrics and by rhythm
    partners = {}
    for kind in kinds:
        for occurrences in phrase_groups(phrases, texts, beats, lengths, kind):
            for a, b in itertools.permutations(range(len(occurrences)), 2):
                partners[(int(occurrences[a][0]), int(occurrences[b][0]))] = (occurrences[a], occurrences[b])
    if not partners:
        return scores.copy()
    prob = _softmax(scores)
    acc = np.zeros_like(scores)
    count = np.zeros(len(scores))
    for idx_a, idx_b in partners.values():
        shift = 0
        if rotate:
            fit = [np.sum(prob[idx_a] * np.roll(prob[idx_b], r, axis=1)) for r in range(12)]
            shift = int(np.argmax(fit))
        acc[idx_a] += np.roll(scores[idx_b], shift, axis=1)
        count[idx_a] += 1
    pooled = scores.copy()
    has = count > 0
    pooled[has] += weight * acc[has] / count[has][:, None]
    return pooled


def pooled_fraction(phrases: Sequence[int], texts: Sequence[str], beats: Sequence[int],
                    lengths: Sequence[int], kinds: Sequence[str] = ("text", "rhythm")) -> float:
    """ share of the notes that have at least one partner phrase (for logging) """
    if len(phrases) == 0:
        return 0.0
    covered = np.zeros(len(phrases), dtype=bool)
    for kind in kinds:
        for occurrences in phrase_groups(phrases, texts, beats, lengths, kind):
            for idx in occurrences:
                covered[idx] = True
    return float(covered.mean())
