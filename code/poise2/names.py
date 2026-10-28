# Invented names, split so that train, dev and test never share one.
from __future__ import annotations

import itertools
import numpy as np

_ONSETS = ["b", "br", "c", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "z", "th", "st"]
_VOWELS = ["a", "e", "i", "o", "u", "a", "e", "o"]
_CODAS = ["", "", "n", "r", "l", "s", "m", "nd", "rn", "ck"]

FIRST_NAMES = [
    "Ada", "Bram", "Cela", "Dov", "Esme", "Faro", "Gil", "Hana", "Ilsa", "Jorn", "Kai", "Lior",
    "Mira", "Nils", "Oona", "Pell", "Quin", "Rhea", "Sol", "Tamsin", "Ulla", "Vern", "Wren",
    "Yara", "Zane", "Arlo", "Bea", "Cass", "Dela", "Emil", "Fenna", "Greer", "Hollis", "Ines",
    "Joss", "Kit", "Lena", "Mael", "Noor", "Orin", "Pia", "Reid", "Sage", "Teo", "Una", "Vida",
    "Wynn", "Xan", "Yuri", "Zia", "Anouk", "Boden", "Ciro", "Dagny", "Elio", "Freya", "Gus",
    "Hedda", "Ivo", "Juno", "Kaspar", "Liv", "Mads", "Nell", "Otis", "Petra", "Rafe", "Siri",
]
ORG_SUFFIXES = ["Labs", "Robotics", "Freight", "Analytics", "Foundry", "Optics", "Systems",
                "Mills", "Partners", "Biotech", "Instruments", "Logistics"]
CITY_PATTERNS = ["{n}", "Port {n}", "{n}ford", "{n}ton", "New {n}", "{n} Falls", "{n}by", "{n} Hill"]


def _syllable_words(seed: int, n: int, min_syl=2, max_syl=2):
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        k = int(rng.integers(min_syl, max_syl + 1))
        w = ""
        for j in range(k):
            w += _ONSETS[int(rng.integers(len(_ONSETS)))] + _VOWELS[int(rng.integers(len(_VOWELS)))]
            if j == k - 1:
                w += _CODAS[int(rng.integers(len(_CODAS)))]
        w = w.capitalize()
        if 4 <= len(w) <= 7 and w not in seen:
            seen.add(w)
            out.append(w)
    return out


_SPLIT_FRACTIONS = {"train": 0.6, "dev": 0.2, "test": 0.2}


def _partition(items, seed):
    rng = np.random.default_rng(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    a = int(n * _SPLIT_FRACTIONS["train"])
    b = a + int(n * _SPLIT_FRACTIONS["dev"])
    return {"train": items[:a], "dev": items[a:b], "test": items[b:]}


_STEMS = _syllable_words(20261004, 2400)
_STEM_SPLITS = _partition(_STEMS, 7)
_FIRST_SPLITS = _partition(FIRST_NAMES, 11)


class NamePool:

    def __init__(self, split: str, rng):
        self.rng = rng
        self.stems = _STEM_SPLITS[split]
        self.first = _FIRST_SPLITS[split]

    def _stem(self, used):
        for _ in range(1000):
            s = self.stems[int(self.rng.integers(len(self.stems)))]
            if s not in used:
                used.add(s)
                return s
        raise RuntimeError("name pool exhausted")

    def person(self, used):
        f = self.first[int(self.rng.integers(len(self.first)))]
        return f"{f} {self._stem(used)}"

    def org(self, used):
        return f"{self._stem(used)} {ORG_SUFFIXES[int(self.rng.integers(len(ORG_SUFFIXES)))]}"

    def city(self, used):
        pat = CITY_PATTERNS[int(self.rng.integers(len(CITY_PATTERNS)))]
        return pat.format(n=self._stem(used))

    def word(self, used):
        return self._stem(used)
