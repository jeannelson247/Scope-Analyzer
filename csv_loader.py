"""
csv_loader.py — Robust loader for oscilloscope CSV/TXT/TSV exports.

Handles the messy reality of scope files:
  * Instrument preamble lines (Tektronix / Rigol / LeCroy headers)
  * Comma, semicolon, or tab delimiters
  * Header row detection (or auto-generated CH names)
  * Large files: C-engine parsing + optional float32 to halve memory
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LoadedData:
    path: str
    df: pd.DataFrame
    delimiter: str
    skiprows: int
    columns: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)     # preamble key → row values
    units: dict = field(default_factory=dict)    # column name → unit string

    @property
    def n_rows(self) -> int:
        return len(self.df)


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        # fall back: pick the delimiter that appears most
        counts = {d: sample.count(d) for d in (",", ";", "\t")}
        return max(counts, key=counts.get) or ","


def _is_float(tok: str) -> bool:
    tok = tok.strip()
    if not tok:
        return False
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _is_data_line(line: str, delim: str) -> tuple[bool, int]:
    """A data line: >=2 non-empty tokens, the FIRST token is numeric, and
    ALL non-empty tokens are numeric. Instrument metadata lines such as
    'Probe Attenuation,1,,100,,1,,1,' start with text, so they fail the
    first-token test even when most of their fields are numbers."""
    toks = line.rstrip("\r\n").split(delim)
    vals = [t for t in toks if t.strip() != ""]
    if len(vals) < 2 or not _is_float(toks[0]):
        return False, len(toks)
    return all(_is_float(t) for t in vals), len(toks)


def _find_data_start(lines: list[str], delim: str) -> tuple[int, list[str] | None]:
    """Return (index of first data line, header names or None).

    The header is the closest previous line with the same field count whose
    tokens are NOT mostly numeric. To avoid one odd metadata line being
    mistaken for data, the line after the candidate must also be numeric
    (or the candidate must be the last sniffed line).
    """
    for i, line in enumerate(lines):
        ok, ntoks = _is_data_line(line, delim)
        if not ok:
            continue
        if i + 1 < len(lines) and lines[i + 1].strip():
            ok2, ntoks2 = _is_data_line(lines[i + 1], delim)
            if not (ok2 and ntoks2 == ntoks):
                continue
        # look back for a header with the same field count
        for j in range(i - 1, max(-1, i - 4), -1):
            htoks = [t.strip() for t in lines[j].rstrip("\r\n").split(delim)]
            hvals = [t for t in htoks if t]
            if len(htoks) == ntoks and hvals and \
                    sum(_is_float(t) for t in hvals) / len(hvals) < 0.5:
                return i, [t.strip() or f"col{k}" for k, t in enumerate(htoks)]
        return i, None
    raise ValueError("No numeric data rows found in file.")


def _parse_meta(lines: list[str], delim: str) -> dict:
    """Parse 'Key,v1,v2,…' preamble lines into {key: [values]}."""
    meta = {}
    for line in lines:
        toks = [t.strip() for t in line.rstrip("\r\n").split(delim)]
        if toks and toks[0] and not _is_float(toks[0]):
            meta[toks[0]] = toks[1:]
    return meta


def load_csv(path: str, float32: bool = True, max_sniff_lines: int = 60) -> LoadedData:
    with open(path, "r", errors="replace") as f:
        head = [f.readline() for _ in range(max_sniff_lines)]
    head = [h for h in head if h]
    delim = _sniff_delimiter("".join(head[:20]))
    start, header = _find_data_start(head, delim)
    meta = _parse_meta(head[:start], delim)

    read_kw = dict(sep=delim, skiprows=start, header=None, engine="c",
                   na_values=["", " ", "NaN", "nan"], on_bad_lines="skip",
                   skip_blank_lines=True)
    try:
        df = pd.read_csv(path, dtype=np.float64, **read_kw)
    except ValueError:
        # stray text somewhere in the body — coerce instead of crashing
        df = pd.read_csv(path, **read_kw)
        df = df.apply(pd.to_numeric, errors="coerce")
    # drop fully-empty trailing columns (some scopes add a dangling delimiter)
    df = df.dropna(axis=1, how="all")

    if header and len(header) >= df.shape[1]:
        df.columns = header[: df.shape[1]]
    else:
        names = ["Time"] + [f"CH{i}" for i in range(1, df.shape[1])]
        df.columns = names[: df.shape[1]]
    df = df.dropna(axis=1, how="all")

    # de-duplicate column names (scopes sometimes repeat labels)
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        c = str(c)
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    df.columns = cols

    if float32:
        df = df.astype(np.float32)

    # map units onto columns (Tektronix 'Vertical Units,V,,A,…' style:
    # values align with the data columns after the time column)
    units: dict[str, str] = {}
    vu = meta.get("Vertical Units")
    if vu:
        cols_after_time = list(df.columns)[1:]
        for c, u in zip(cols_after_time, vu):
            if u:
                units[c] = u

    return LoadedData(path=path, df=df, delimiter=delim,
                      skiprows=start, columns=list(df.columns),
                      meta=meta, units=units)


def minmax_decimate(x: np.ndarray, y: np.ndarray, target: int):
    """Min–max envelope decimation: preserves spikes/peaks while reducing
    point count for export. Returns (x_d, y_d)."""
    n = len(x)
    if n <= target or target < 8:
        return x, y
    nbins = max(4, target // 2)
    edges = np.linspace(0, n, nbins + 1).astype(np.int64)
    xs = np.empty(nbins * 2, dtype=x.dtype)
    ys = np.empty(nbins * 2, dtype=y.dtype)
    for b in range(nbins):
        s, e = edges[b], max(edges[b] + 1, edges[b + 1])
        seg = y[s:e]
        finite = np.isfinite(seg)
        if not np.any(finite):
            # Preserve the time span of a bad segment without letting an
            # all-NaN/all-inf bin crash the loader. QC will still flag it.
            i_min, i_max = s, e - 1
        else:
            local = np.flatnonzero(finite)
            vals = seg[finite]
            i_min = s + int(local[int(np.argmin(vals))])
            i_max = s + int(local[int(np.argmax(vals))])
        a, b2 = sorted((i_min, i_max))
        xs[2 * b], xs[2 * b + 1] = x[a], x[b2]
        ys[2 * b], ys[2 * b + 1] = y[a], y[b2]
    return xs, ys
