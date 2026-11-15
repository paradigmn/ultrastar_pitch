#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file          make_exclude_list.py
@brief         generate an exclude list for label_quality.py from an index of note files
@author        paradigm

The index is a text file with one path per line, e.g. created with
    cd /path/to/songs && find . -name "*.txt" > index.txt

Files with one of the skip words in their file name are written to the exclude list, the same
rules as --skip-names of label_quality.py use. Files, which only match one of the review words
(by default the whole word "inst"), are added as comments: remove the "# " in front of a line
to exclude it too.

    python dev/make_exclude_list.py index.txt -o exclude.txt
    python dev/label_quality.py /path/to/songs -o scores.csv --exclude exclude.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_quality import DEFAULT_SKIP_WORDS, compile_skip_words, is_skipped  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="generate an exclude list from an index of note files")
    parser.add_argument("index", help="text file with one path per line")
    parser.add_argument("-o", "--output", default="exclude.txt")
    parser.add_argument("--skip-names", default=DEFAULT_SKIP_WORDS,
                        help="words in the file name, which exclude a file (default: %(default)s)")
    parser.add_argument("--review-names", default="=inst",
                        help="words, which only get a commented entry for review (default: %(default)s, empty: none)")
    args = parser.parse_args()

    skip, review = compile_skip_words(args.skip_names), compile_skip_words(args.review_names)
    with open(args.index, encoding="utf-8", errors="replace") as file:
        entries = [line.strip() for line in file if line.strip()]
    excluded = [entry for entry in entries if is_skipped(entry, skip)]
    candidates = [entry for entry in entries if not is_skipped(entry, skip) and is_skipped(entry, review)]

    with open(args.output, "w", encoding="utf-8", newline="\n") as file:
        file.write("# exclude list for label_quality.py, generated from {0} ({1} files)\n".format(
            os.path.basename(args.index), len(entries)))
        file.write("# file names with one of these words: {0}\n".format(args.skip_names))
        file.writelines(entry + "\n" for entry in excluded)
        if candidates:
            file.write("\n# --- candidates for review ({0} files, words: {1}) ---\n".format(
                len(candidates), args.review_names))
            file.write("# remove the '# ' in front of a line to exclude the file as well\n")
            file.writelines("# " + entry + "\n" for entry in candidates)
    print("{0} of {1} files excluded, {2} candidates to review -> {3}".format(
        len(excluded), len(entries), len(candidates), args.output))


if __name__ == "__main__":
    main()
