"""Metrics for formula recognition (image -> LaTeX). Pure Python, no extra packages.

- exact_match        : prediction == reference (whitespace ignored)
- cer                : character error rate (edit distance / reference length, whitespace ignored)
- token_edit_sim     : 1 - token edit distance / max length (LaTeX tokens split on spaces)
- bleu4              : corpus BLEU-4 on LaTeX tokens (with brevity penalty)
"""
from __future__ import annotations

import math
from collections import Counter


def _lev(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _nospace(s: str) -> str:
    return "".join(s.split())


def _bleu4(preds_tok, refs_tok) -> float:
    match, total = [0] * 4, [0] * 4
    hyp_len = ref_len = 0
    for p, r in zip(preds_tok, refs_tok):
        hyp_len += len(p)
        ref_len += len(r)
        for n in range(1, 5):
            pc = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            match[n - 1] += sum(min(c, rc[g]) for g, c in pc.items())
            total[n - 1] += max(len(p) - n + 1, 0)
    if hyp_len == 0 or min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / 4
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * math.exp(log_p)


def formula_scores(preds: list[str], refs: list[str]) -> dict:
    n = max(len(refs), 1)
    em = sum(_nospace(p) == _nospace(r) for p, r in zip(preds, refs)) / n
    char_err = sum(_lev(_nospace(p), _nospace(r)) for p, r in zip(preds, refs))
    char_tot = sum(max(len(_nospace(r)), 1) for r in refs)
    pt, rt = [p.split() for p in preds], [r.split() for r in refs]
    tok_sim = sum(1 - _lev(a, b) / max(len(a), len(b), 1) for a, b in zip(pt, rt)) / n
    return {
        "exact_match": em,
        "cer": char_err / max(char_tot, 1),
        "token_edit_sim": tok_sim,
        "bleu4": _bleu4(pt, rt),
    }
