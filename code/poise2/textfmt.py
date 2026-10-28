# Sentence templates that can be parsed back into the facts they were filled from.
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

FIELD_PATTERNS = {
    "year": r"(?P<year>\d{4})",
    "pop": r"(?P<pop>\d{1,3}(?:,\d{3})+|\d+)",
    "num": r"(?P<num>\d+)",
    "x": r"(?P<x>-?\d+)",
    "y": r"(?P<y>-?\d+)",
    "i": r"(?P<i>\d+)",
}
_FIELD = re.compile(r"\{(\w+)\}")


def compile_template(tpl: str) -> re.Pattern:
    out, pos, seen = [], 0, {}
    for m in _FIELD.finditer(tpl):
        out.append(re.escape(tpl[pos:m.start()]))
        name = m.group(1)
        base = re.sub(r"\d+$", "", name)
        if name in seen:
            out.append(f"(?P={name})")
        else:
            pat = FIELD_PATTERNS.get(base)
            out.append(pat.replace(f"?P<{base}>", f"?P<{name}>") if pat else f"(?P<{name}>.+?)")
            seen[name] = True
        pos = m.end()
    out.append(re.escape(tpl[pos:]))
    return re.compile("^" + "".join(out) + "$")


class TemplateSet:

    def __init__(self, table: Dict[str, List[str]]):
        self.table = table
        self.compiled: List[Tuple[str, str, re.Pattern]] = []
        for key, tpls in table.items():
            for t in tpls:
                self.compiled.append((key, t, compile_template(t)))

    def render(self, key: str, idx: int, **fields) -> str:
        return self.table[key][idx % len(self.table[key])].format(**fields)

    def parse(self, text: str) -> Optional[Tuple[str, Dict[str, str]]]:
        hits = []
        for key, t, rx in self.compiled:
            m = rx.match(text)
            if m:
                literal = len(_FIELD.sub("", t))
                hits.append((literal, key, m.groupdict()))
        if not hits:
            return None
        hits.sort(key=lambda h: -h[0])
        best = hits[0]
        for h in hits[1:]:
            if h[0] == best[0] and (h[1], h[2]) != (best[1], best[2]):
                raise ValueError(f"ambiguous text: {text!r} -> {hits}")
        return best[1], best[2]
