
from __future__ import annotations

import os
import re
import math
import random
import argparse
import itertools
from collections import Counter
# ── Heavy-dependency guard: --preview also works without torch ──────────────────
_RUNTIME_OK, _IMPORT_ERR = True, None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from models.text_latent import SimpleTextEncoder, LLMTextEncoder
    from models.text_alignment import (VICRegAlign, _dominant_period,
                                       _jump_score)
except Exception as _e:                                    # pragma: no cover
    _RUNTIME_OK, _IMPORT_ERR = False, _e
    from types import SimpleNamespace

    class Dataset:                                         # placeholder
        pass

    class _ShimModule:                                     # only lets class definitions pass;
        pass                                               # instantiation is guarded by _RUNTIME_OK

    class _ShimNoGrad:                                     # @torch.no_grad() placeholder
        def __call__(self, f):
            return f

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    nn = SimpleNamespace(Module=_ShimModule)
    torch = SimpleNamespace(no_grad=lambda: _ShimNoGrad())
    SimpleTextEncoder = LLMTextEncoder = VICRegAlign = None  # [preview placeholder]
    _dominant_period = _jump_score = None
import sys
_argv = list(sys.argv)
sys.argv = _argv[:1]                      # at import time, adapt_z only sees the script name
try:
    from adapt_z import set_seed, calculate_metrics, settings
except (Exception, SystemExit):           # SystemExit derives from BaseException and must be caught explicitly
    settings = {}

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if _RUNTIME_OK:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def calculate_metrics(preds, truths):
        mae = float(np.mean(np.abs(preds - truths)))
        mse = float(np.mean((preds - truths) ** 2))
        return mae, mse
finally:
    sys.argv = _argv                      # restore the real command line whether or not it succeeded

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# ============================================================
# Unified plotting style
# ============================================================

DARK_BLUE = "#2457A7"   # main baseline
DARK_RED = "#8B1E3F"    # iAdapter / ours

# The other ablation methods are de-emphasized relative to ours/baseline, but stay saturated enough that every line is clearly visible,
# never so faint that it looks "not drawn".
MUTED_COLORS = [
    "#6E6E6E",
    "#B3773D",
    "#4F8A68",
    "#7C6FA3",
    "#A85F5F",
]


def _find_chinese_font():
    """Prefer Microsoft YaHei; fall back automatically if it is missing."""
    available_fonts = {font.name for font in fm.fontManager.ttflist}

    candidates = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
    ]

    for font_name in candidates:
        if font_name in available_fonts:
            return font_name

    print("[viz] Microsoft YaHei not found, using DejaVu Sans")
    return "DejaVu Sans"


CN_FONT = _find_chinese_font()

LEGEND_FONT = fm.FontProperties(
    family=CN_FONT,
    weight="bold",
    size=14,
)



plt.rcParams.update({
    "font.family": CN_FONT,
    "axes.unicode_minus": False,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})


def _get_line_style(mode, index, baseline_mode):
    """Highlight ours and the main baseline; de-emphasize the other methods."""
    if _is_ours(mode):
        return {
            "color": DARK_RED,
            "lw": 2.8,
            "ls": "-",
            "marker": "o",
            "markersize": 7,
            "markeredgecolor": "white",
            "markeredgewidth": 0.8,
            "alpha": 1.0,
            "zorder": 8,
        }

    if mode == baseline_mode:
        return {
            "color": DARK_BLUE,
            "lw": 2.4,
            "ls": "--",
            "marker": "s",
            "markersize": 6,
            "markeredgecolor": "white",
            "markeredgewidth": 0.7,
            "alpha": 1.0,
            "zorder": 7,
        }

    return {
        "color": MUTED_COLORS[index % len(MUTED_COLORS)],
        "lw": 1.7,
        "ls": "--",
        "marker": "s",
        "markersize": 5.5,
        "markeredgecolor": "white",
        "markeredgewidth": 0.4,
        "alpha": 0.95,
        "zorder": 3,
    }


def _style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.28,
    )

    # Fewer vertical grid lines give a cleaner figure
    ax.grid(
        axis="x",
        linestyle="-",
        linewidth=0.35,
        alpha=0.12,
    )

    ax.tick_params(
        axis="both",
        direction="out",
        width=0.8,
        length=3.5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")


def _save_figure(fig, save_path):
    """Also save a high-resolution PNG; PDF is recommended for the paper."""
    save_path = Path(save_path)

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    # Vector version, stays sharp when placed in the paper
    fig.savefig(
        save_path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )


# ══════════════════════════════════════════════════════════════
# SECTION 1 — DOMAIN REGISTRY (Time-MMD, 9 domains)
# ══════════════════════════════════════════════════════════════

DOMAINS = ["Agriculture", "Climate", "Economy", "Energy", "Environment",
           "Health_US", "Security", "SocialGood", "Traffic", "Weather"]

_DOMAIN_ALIASES = {d.lower(): d for d in DOMAINS}
_DOMAIN_ALIASES.update({"health": "Health_US", "health_us": "Health_US",
                        "socialgood": "SocialGood", "social_good": "SocialGood",
                        "new_metr": "Weather"})

_DOMAIN_TOPIC = {
    "Agriculture": "agriculture",
    "Climate": "climate and drought",
    "Economy": "economy and international trade",
    "Energy": "energy and fuel prices",
    "Environment": "environment and air quality",
    "Health_US": "public health and influenza-like illness",
    "Security": "security and disaster assistance",
    "SocialGood": "social good and unemployment",
    "Traffic": "traffic and transportation",
    "Weather": "weather conditions",
}



# Heuristic default windows (seq_len, pred_len) by domain sampling frequency; the CLI can override them, and they shrink automatically.
# pred_len can be a single int or a list[int] (a default combination of several horizons).
_DOMAIN_WINDOW = {
    "Agriculture": (36, [6,8,10,12]),   "Climate": (36, [6,8,10,12]),   "Economy": (36, [6,8,10,12]),
    "Energy": (48, [6,8,10,12]),        "Environment": (96, [48, 96, 192, 336]),
    "Health_US": (96, [12,24,36,48]),    "Security": (36, [6,8,10,12]),
    "SocialGood": (36, [6,8,10,12]),    "Traffic": (36, [6,8,10,12]),
    "Weather": (96, [48, 96, 192, 336]),
}

# _DOMAIN_WINDOW = {
#     "Agriculture": (96, 6),   "Climate": (96, 6),   "Economy": (96, 6),
#     "Energy": (96, 8),        "Environment": (96, 12),
#     "Health_US": (96, 12),    "Security": (96, 6),
#     "SocialGood": (96, 6),    "Traffic": (96, 6),
# }


def canonical_domain(name: str) -> str:
    key = os.path.splitext(str(name).strip())[0].lower()
    if key in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[key]
    raise ValueError(f"Unknown domain '{name}'. Choose from {DOMAINS} "
                     f"(alias: Health -> Health_US).")


# ══════════════════════════════════════════════════════════════
# SECTION 2 — TEXT CLEANING (pure Python; per clause, removes NA / pipeline boilerplate)
# ══════════════════════════════════════════════════════════════

_DATE_SEG = re.compile(r"(\d{4}-\d{2}-\d{2})\s*:\s*")

_DROP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"^\s*(?:na|n/?a)\b",
    r"\bno (?:objective facts?|relevant|direct|specific|useful) ",
    r"\bthere (?:is|are|was|were) no\b",
    r"\bno (?:predictions?|information|basis|data)\b",
    r"\bit is impossible to make\b",
    r"\bunable to (?:extract|make|provide|glean)\b",
    r"\bcould(?: not|n't) (?:provide|extract|be gleaned)\b",
    r"\bsearch results (?:do not|did not|don't|didn't|seem to be|appear to be|are (?:mostly )?unrelated|primarily focused|were not relevant)\b",
    r"\bresults (?:are|appear) (?:mostly )?(?:unrelated|irrelevant)\b",
    r"\bunfortunately\b",
    r"\bno useful information\b",
    r"\bnot relevant to (?:making )?predict",
    r"^\s*(?:long|short)[- ]?term predictions?\s*:?\s*$",
    r"^\s*objective facts about [^:]{0,120}:?\s*$",
    r"^\s*predictions you can make for the (?:long|short)[- ]?term future[^:]*:?\s*$",
]]

_STRIP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"here is (?:the |my )?[^:;]{0,140}:",
    r"note\s*:[^;]*",
    r"after reviewing the search results[^:;,.]*[,:.]?",
    r"based on the (?:provided )?search results[^:;,.]*[,:.]?",
    r"since (?:the search results|there is no relevant information)[^;]*",
    r"the (?:other|majority of the|provided|first) [^;]{0,60}(?:results?|title)[^;]{0,120}(?:relevant|related|irrelevant|discarded|summary)[^;]*",
    r"therefore,? (?:the predictions and analysis|my output will be|no useful information)[^;]*",
    r"as a result,? i must conclude[^;]*",
    r"i (?:was able to|have) (?:extract|found)[^;]{0,160}(?::|\.)",
    r"predictions you can make for the (?:long|short)[- ]?term future\s*(?:\([^)]*\))?\s*:\s*",
    r"^\s*(?:my )?(?:long|short)[- ]?term predictions?\s*(?:are)?\s*:\s*",
    r"^\s*the objective facts about [^:]{0,120}(?:are|situation)\s*:\s*",
    r"\(no relevant information found\)",
    r"\(next (?:few months|1-4 weeks)\)",
]]

_WS = re.compile(r"\s+")


def _clean_clause(s: str) -> str:
    """Clean a single clause; return '' if it carries no information."""
    if not s:
        return ""
    s = re.sub(r"\bNA(?=[A-Z(])", " ", s)          # fix "NAHere is" concatenation errors
    for pat in _DROP_PATTERNS:
        if pat.search(s):
            return ""
    for pat in _STRIP_PATTERNS:
        s = pat.sub(" ", s)
    s = _WS.sub(" ", s).strip(" ;,.-")
    if not s:
        return ""
    for pat in _DROP_PATTERNS:                     # re-check after stripping
        if pat.search(s):
            return ""
    return s


def split_date_segments(text: str):
    """'d1: seg1; d2: seg2 ...' -> [(date|None, body), ...] (in order of appearance)."""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = _DATE_SEG.split(text)
    out = []
    if parts[0].strip():
        out.append((None, parts[0]))
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i], parts[i + 1]))
    return out


def clean_timemmd_text(raw, max_clauses: int = 6, min_clause_chars: int = 20):
    """
    Clean one fact/preds cell -> (clean_text, n_kept, n_total).
    Keeps the "YYYY-MM-DD:" date anchors; when budget is limited, the **latest** date segments are kept first
    (most relevant for online forecasting); output is in chronological order.
    """
    segs = split_date_segments(raw)
    n_total, kept = 0, []
    for pos, (d, body) in enumerate(segs):
        for clause in body.split(";"):
            n_total += 1
            c = _clean_clause(clause)
            if c and len(c) >= min_clause_chars:
                kept.append((pos, d, c))
    kept = kept[-max_clauses:] if max_clauses > 0 else kept
    pieces, last_d = [], None
    for _pos, d, c in kept:
        if d and d != last_d:
            pieces.append(f"{d}: {c}")
            last_d = d
        else:
            pieces.append(c)
    return " ".join(pieces), len(kept), n_total


# ══════════════════════════════════════════════════════════════
# SECTION 3 — RealTextProvider (numeric + real text, two layouts, leakage-resistant)
# ══════════════════════════════════════════════════════════════

_TEXT_LIKE_COLS = {"date", "Date", "start_date", "end_date", "fact", "pred",
                   "preds", "Final_Output", "Final_Search_8", "Final_Search_16"}
_HISTORY_STAT_COLS = {"prior_history_avg", "prior_history_std"}


class RealTextProvider:
    """
    Supplies cleaned real text in numeric-row order (replaces the old TrafficTextGenerator):
      · self.numeric  : (n_rows, C) float32, NaN interpolated/filled, not standardized;
      · self.row_text : cleaned text of each numeric row;
      · self.row_conf : [C1] per-row text confidence = causal EMA of row_has_info,
        momentum CONF_MOMENTUM (can be overridden via the class attribute before construction); floored at
        0.25 when the current row has information. Downstream soft-gates with emb*conf instead of a hard 0/1 mask;
      · text_for_row(r): aggregated text of row r and its preceding text_lookback-1 rows
        (all on the history side); when matching the original layout, enforces text.end_date <= numeric row.end_date
        — consistent with MM-TSFlib's anti-leakage constraint; text of future target rows is never visible.
    """

    SIMPLE_MAX_LEN = 48
    _VOCAB_CAP = 4000
    _TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")
    CONF_MOMENTUM = 0.8            # [C1] confidence EMA momentum (larger = smoother)
    CONF_FLOOR = 0.25              # [C1] confidence floor when the current row has information

    def __init__(self, domain: str, data_dir: str = "./data",
                 text_source: str = "both", text_lookback: int = 2,
                 max_clauses: int = 6, max_chars: int = 1200,
                 min_clause_chars: int = 20, raw_text_topk: int = 6,
                 raw_lookback_days: int = 60,
                 keep_history_stats: bool = False, ot_only: bool = False,
                 verbose: bool = True):
        self.domain = canonical_domain(domain)
        self.data_dir = data_dir
        if text_source not in ("fact", "preds", "both"):
            raise ValueError("text_source must be fact/preds/both")
        self.text_source = text_source
        self.text_lookback = max(1, text_lookback)
        self.max_clauses = max_clauses
        self.max_chars = max_chars
        self.min_clause_chars = min_clause_chars
        self.raw_text_topk = raw_text_topk
        self.raw_lookback_days = raw_lookback_days
        self.keep_history_stats = keep_history_stats
        self.ot_only = ot_only
        self.verbose = verbose
        self._fallback = (f"No notable {_DOMAIN_TOPIC[self.domain]} "
                          f"developments reported in this period.")

        layout = self._resolve_paths()
        if layout[0] == "merged":
            self._load_merged(layout[1])
        else:
            self._load_raw(*layout[1:])
        self._build_row_texts()
        self._word2idx = None
        # [O3] frozen encoder cache: {cache_key: {global_row: pooled/ids}}
        self.enc_cache: dict = {}
        if verbose:
            self.coverage_report()

    # ── Path discovery ──
    def _resolve_paths(self):
        d, dd = self.domain, self.data_dir
        names = ([d] + (["Health"] if d == "Health_US" else [])
                 + (["new_metr"] if d == "Weather" else []))
        for nm in names:
            for p in (os.path.join(dd, f"{nm}.csv"),
                      os.path.join(dd, nm, f"{nm}.csv")):
                if os.path.isfile(p):
                    if self.verbose:
                        print(f"[provider] layout=merged  {p}")
                    return ("merged", p)
        for nm in names:
            num = os.path.join(dd, "numerical", nm, f"{nm}.csv")
            rep = os.path.join(dd, "textual", nm, f"{nm}_report.csv")
            sea = os.path.join(dd, "textual", nm, f"{nm}_search.csv")
            if os.path.isfile(num) and (os.path.isfile(rep) or os.path.isfile(sea)):
                if self.verbose:
                    print(f"[provider] layout=raw  num={num}")
                return ("raw", num,
                        rep if os.path.isfile(rep) else None,
                        sea if os.path.isfile(sea) else None)
        raise FileNotFoundError(
            f"No data for domain '{d}' under '{dd}'. Expected '{d}.csv' "
            f"(MM-TSFlib merged) or numerical/{d}/{d}.csv + textual/{d}/*.csv "
            f"(Time-MMD raw). Data: https://github.com/AdityaLab/Time-MMD")

    # ── Numeric column selection (linear interpolation of NaN, same as TSlib uea.interpolate_missing) ──
    def _pick_numeric(self, df: pd.DataFrame):
        drop = set(_TEXT_LIKE_COLS)
        if not self.keep_history_stats:
            drop |= _HISTORY_STAT_COLS
        cols = [c for c in df.columns
                if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
        if self.ot_only and "OT" in cols:
            cols = ["OT"]
        elif self.ot_only:
            print("[provider] WARN: ot_only=True but no 'OT' column; using all.")
        if not cols:
            raise ValueError(f"No numeric columns for {self.domain}")
        sub = df[cols].apply(
            lambda y: y.interpolate(method="linear", limit_direction="both")
            if y.isna().any() else y)
        vals = sub.values.astype(np.float32)
        if np.isnan(vals).any():
            mu = np.nanmean(vals, axis=0)
            for j in range(vals.shape[1]):
                m = np.isnan(vals[:, j])
                if m.any():
                    vals[m, j] = 0.0 if np.isnan(mu[j]) else mu[j]
        return vals, cols

    @staticmethod
    def _parse_dates(df, prefer=("end_date", "date", "Date", "start_date")):
        for c in prefer:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce")
                if s.notna().any():
                    return s
        return None

    # ── Layout A: MM-TSFlib merged ──
    def _load_merged(self, path):
        import csv
        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError:
            df = pd.read_csv(path, quoting=csv.QUOTE_NONE,
                             engine="python", on_bad_lines="warn")
        dates = self._parse_dates(df)
        if dates is not None:
            order = np.argsort(dates.values.astype("datetime64[ns]"),
                               kind="stable")
            if not np.all(order == np.arange(len(df))):
                df = df.iloc[order].reset_index(drop=True)
                dates = dates.iloc[order].reset_index(drop=True)
        self.df, self.dates = df, dates
        self.numeric, self.channels = self._pick_numeric(df)
        fc = "fact" if "fact" in df.columns else None
        pc = next((c for c in ("preds", "pred") if c in df.columns), None)
        self._raw_fact = (df[fc].astype(str).where(df[fc].notna(), "")
                          if fc else pd.Series([""] * len(df)))
        self._raw_pred = (df[pc].astype(str).where(df[pc].notna(), "")
                          if pc else pd.Series([""] * len(df)))
        if fc is None and pc is None:
            print(f"[provider] WARN: no fact/preds column in {path}; "
                  f"neutral fallback text will be used for all rows.")

    # ── Layout B: original Time-MMD (interval matching + anti-leakage: text.end <= numeric row.end) ──
    def _load_raw(self, num_path, rep_path, sea_path):
        num = pd.read_csv(num_path)
        num["_end"] = self._parse_dates(num, ("end_date", "date", "Date"))
        num = num.sort_values("_end", kind="stable").reset_index(drop=True)
        self.df, self.dates = num, num["_end"]
        self.numeric, self.channels = self._pick_numeric(num)

        frames = []
        for p in (rep_path, sea_path):
            if p is None:
                continue
            t = pd.read_csv(p)
            t["_end"] = self._parse_dates(t, ("end_date", "date", "Date"))
            pc = next((c for c in ("pred", "preds") if c in t.columns), None)
            frames.append(pd.DataFrame({
                "_end": t["_end"],
                "fact": (t["fact"].astype(str).where(t["fact"].notna(), "")
                         if "fact" in t.columns else ""),
                "pred": (t[pc].astype(str).where(t[pc].notna(), "")
                         if pc else ""),
            }))
        txt = (pd.concat(frames, ignore_index=True)
               .dropna(subset=["_end"]).sort_values("_end", kind="stable")
               .reset_index(drop=True)) if frames else pd.DataFrame(
                   columns=["_end", "fact", "pred"])

        n, facts, preds = len(num), [""] * len(num), [""] * len(num)
        ends, j = txt["_end"].values, 0
        lb = pd.Timedelta(days=self.raw_lookback_days)
        for i in range(n):
            e = num["_end"].iloc[i]
            if pd.isna(e):
                continue
            while j < len(txt) and ends[j] <= np.datetime64(e):
                j += 1
            fp, pp = [], []
            for k in range(max(0, j - self.raw_text_topk), j):
                te = txt["_end"].iloc[k]
                if e - te > lb:
                    continue
                dstr = pd.Timestamp(te).strftime("%Y-%m-%d")
                if txt["fact"].iloc[k].strip():
                    fp.append(f"{dstr}: {txt['fact'].iloc[k]}")
                if txt["pred"].iloc[k].strip():
                    pp.append(f"{dstr}: {txt['pred'].iloc[k]}")
            facts[i], preds[i] = "; ".join(fp), "; ".join(pp)
        self._raw_fact, self._raw_pred = pd.Series(facts), pd.Series(preds)

    # ── Per-row cleaning + [C1] confidence EMA ──
    def _build_row_texts(self):
        n = len(self.numeric)
        self.row_text, self.row_has_info = [], np.zeros(n, dtype=bool)
        half = max(2, self.max_clauses // 2)
        for i in range(n):
            parts = []
            if self.text_source in ("fact", "both"):
                fc, k, _ = clean_timemmd_text(
                    self._raw_fact.iloc[i],
                    self.max_clauses if self.text_source == "fact" else half,
                    self.min_clause_chars)
                if k:
                    parts.append("Facts. " + fc)
            if self.text_source in ("preds", "both"):
                pc, k, _ = clean_timemmd_text(
                    self._raw_pred.iloc[i],
                    self.max_clauses if self.text_source == "preds" else half,
                    self.min_clause_chars)
                if k:
                    parts.append("Outlook. " + pc)
            if parts:
                self.row_text.append(" ".join(parts))
                self.row_has_info[i] = True
            else:
                self.row_text.append(self._fallback)

        # [C1] text confidence = causal EMA of row_has_info (history only, no leakage):
        #   conf[i] = m*conf[i-1] + (1-m)*has_info[i]
        # smooths out 0/1 steps; consecutive informative rows -> conf approaches 1, long uninformative stretches -> exponential
        # decay towards 0. Floored at CONF_FLOOR when the current row has information, to avoid fully masking at cold start.
        m = float(self.CONF_MOMENTUM)
        conf, c = np.zeros(n, dtype=np.float32), 0.0
        for i in range(n):
            c = m * c + (1.0 - m) * float(self.row_has_info[i])
            conf[i] = c
        self.row_conf = np.clip(
            np.maximum(conf, self.CONF_FLOOR * self.row_has_info), 0.0, 1.0)

    # ── Public interface ──
    @property
    def n_rows(self):
        return len(self.row_text)

    def text_for_row(self, r: int) -> str:
        r = int(min(max(r, 0), self.n_rows - 1))
        parts, seen = [], set()
        for rr in range(max(0, r - self.text_lookback + 1), r + 1):
            if self.row_has_info[rr] and self.row_text[rr] not in seen:
                parts.append(self.row_text[rr])
                seen.add(self.row_text[rr])
        joined = " ".join(parts) if parts else self._fallback

        # [NEW] inject the date of the row index as a temporal anchor.
        # Several rows often aggregate to the same text (overlapping lookback windows), and a frozen LLM outputs nearly
        # identical pooled vectors for nearly identical strings, another cause of text-embedding collapse.
        # With a date-label prefix, even when the text content is similar, the pooled vectors spread out
        # in embedding space because of the differing position tokens.
        date_tag = ""
        if self.dates is not None and pd.notna(self.dates.iloc[r]):
            date_tag = f"[t={self.dates.iloc[r].date()}] "
        joined = date_tag + joined

        if len(joined) > self.max_chars:                    # truncate from the left, keep the latest
            joined = joined[-self.max_chars:]
            sp = joined.find(" ")
            if 0 < sp < 80:
                joined = joined[sp + 1:]
        return joined

    def conf_for_row(self, r: int) -> float:
        """[C1] EMA confidence of row r (clamped to the valid range)."""
        r = int(min(max(r, 0), self.n_rows - 1))
        return float(self.row_conf[r])

    # ── Vocabulary of the simple encoder (real corpus) ──
    def _ensure_vocab(self):
        if self._word2idx is not None:
            return
        cnt = Counter()
        for i, t in enumerate(self.row_text):
            if self.row_has_info[i]:
                cnt.update(self._TOKEN.findall(t.lower()))
        common = [w for w, _ in cnt.most_common(self._VOCAB_CAP - 2)]
        self._word2idx = {"<pad>": 0, "<unk>": 1}
        self._word2idx.update({w: i + 2 for i, w in enumerate(common)})

    @property
    def vocab_size(self):
        self._ensure_vocab()
        return len(self._word2idx)

    def tokenize(self, text: str, max_len: int | None = None):
        self._ensure_vocab()
        L = max_len or self.SIMPLE_MAX_LEN
        ids = [self._word2idx.get(t, 1)
               for t in self._TOKEN.findall(text.lower())][:L]
        return ids + [0] * (L - len(ids))

    # ── Diagnostics ──
    def coverage_report(self):
        n, info = self.n_rows, int(self.row_has_info.sum())
        lens = np.array([len(t) for t in self.row_text])
        freq = ""
        if self.dates is not None and self.dates.notna().sum() > 2:
            freq = (f"  median_step="
                    f"{self.dates.dropna().diff().dt.days.median():.0f}d")
        print(f"[provider] {self.domain}: rows={n}  "
              f"channels={len(self.channels)} {self.channels[:5]}"
              f"{'...' if len(self.channels) > 5 else ''}  "
              f"informative={info}/{n} ({100 * info / max(n, 1):.1f}%)  "
              f"conf_ema mean={self.row_conf.mean():.2f} "
              f"(m={self.CONF_MOMENTUM})  "
              f"clean_chars mean={lens.mean():.0f} max={lens.max()}{freq}  "
              f"source={self.text_source}")

    def preview(self, k: int = 3):
        for r in np.linspace(0, self.n_rows - 1,
                             num=min(k, self.n_rows), dtype=int):
            d = (self.dates.iloc[r].date() if self.dates is not None
                 and pd.notna(self.dates.iloc[r]) else "?")
            raw = " | ".join(s for s in (str(self._raw_fact.iloc[r])[:150],
                                         str(self._raw_pred.iloc[r])[:150])
                             if s and s != "nan")
            print(f"\n──[{self.domain}] row {r}  date={d}  "
                  f"informative={bool(self.row_has_info[r])}  "
                  f"conf={self.row_conf[r]:.2f}")
            print(f"  RAW   : {raw[:320]}{'...' if len(raw) > 320 else ''}")
            print(f"  CLEAN : {self.row_text[r][:320]}"
                  f"{'...' if len(self.row_text[r]) > 320 else ''}")


# ══════════════════════════════════════════════════════════════
# SECTION 4 — DATA (60/20/20 split; returns split row offsets for text alignment)
# ══════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    """Sliding-window dataset. Returns (x, y, x_mark_placeholder, ts_start_idx)."""
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data.astype(np.float32)
        self.seq_len, self.pred_len = seq_len, pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.zeros(self.seq_len, 4), idx)


def auto_window(n_rows: int, seq_len: int, pred_len: int,
                min_train: int = 20, min_test: int = 5):
    """Shrink windows automatically by data length so that every split has enough samples."""
    s, p = seq_len, pred_len
    while True:
        t_end = int(n_rows * 0.6)
        train_n = t_end - s - p + 1
        test_n = (n_rows - max(0, int(n_rows * 0.8) - s)) - s - p + 1
        if (train_n >= min_train and test_n >= min_test) or (s <= 8 and p <= 2):
            break
        if s > 8:
            s = max(8, s // 2)
        elif p > 2:
            p = max(2, p // 2)
        else:
            break
    if (s, p) != (seq_len, pred_len):
        print(f"[data] auto-shrink window: seq_len {seq_len}->{s}, "
              f"pred_len {pred_len}->{p} (n_rows={n_rows})")
    return s, p


def load_real_data_mm(provider: RealTextProvider, seq_len: int, pred_len: int):
    """
    Same 60/20/20 split and scaler logic as the original load_real_data, and additionally returns each
    split's raw row offsets — the global row index of the last input row of window index w
    = offsets[split] + w + seq_len - 1 (the key to aligning real text).
    """
    values = provider.numeric
    n = len(values)
    t_end, v_end = int(n * 0.6), int(n * 0.8)
    scaler = StandardScaler().fit(values[:t_end])
    data = scaler.transform(values).astype(np.float32)

    val_start = max(0, t_end - seq_len)
    test_start = max(0, v_end - seq_len)
    train = TimeSeriesDataset(data[:t_end], seq_len, pred_len)
    val = TimeSeriesDataset(data[val_start:v_end], seq_len, pred_len)
    test = TimeSeriesDataset(data[test_start:], seq_len, pred_len)
    offsets = {"train": 0, "val": val_start, "test": test_start}
    print(f"[data] {provider.domain}  rows={n}  channels={values.shape[1]}  "
          f"train={len(train)}  val={len(val)}  test={len(test)}  "
          f"offsets={offsets}")
    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Dataset '{provider.domain}' too short (rows={n}) for "
            f"seq_len={seq_len}, pred_len={pred_len}. Use the full Time-MMD "
            f"csv or reduce --seq_len/--pred_len.")
    return train, val, test, scaler, offsets


_DEFAULT_LRS = dict(finetune_lr=1e-3, adapter_lr=1e-3,
                    online_lr=1e-4, adp_online_lr=1e-3)


def ensure_settings(model_name: str, data_key: str, overrides=None):
    """Inject the lr configuration for new domains into settings (injected at run time, adapt_z.py is not modified). [O8]"""
    tbl = settings.setdefault(model_name, {})
    if data_key not in tbl:
        src = tbl.get("traffic") or next(iter(tbl.values()), None)
        entry = dict(src) if isinstance(src, dict) else {}
        for k, v in _DEFAULT_LRS.items():
            entry.setdefault(k, v)
        tbl[data_key] = entry
        print(f"[settings] injected '{model_name}/{data_key}': {tbl[data_key]}")
    if overrides:
        applied = {k: v for k, v in overrides.items() if v is not None}
        if applied:
            tbl[data_key].update(applied)
            print(f"[settings] overrides applied: {applied}")


def compute_offset_features(x):
    """
    Distribution-shift statistics (B, 8): let the MoE gate perceive concept drift.
    x is already normalized by the training-set StandardScaler, so the training distribution ≈ N(0,1).
    """
    B, L, N = x.shape
    flat = x.reshape(B, -1)
    mean_all = flat.mean(dim=1)
    std_all = flat.std(dim=1, unbiased=False)
    q90 = torch.quantile(flat, 0.90, dim=1)
    q10 = torch.quantile(flat, 0.10, dim=1)
    q = max(1, L // 4)
    trend = x[:, -q:, :].mean(dim=(1, 2)) - x[:, :q, :].mean(dim=(1, 2))
    diff_std = (x[:, 1:, :] - x[:, :-1, :]).std(dim=(1, 2), unbiased=False)
    last_level = x[:, -1, :].mean(dim=1)
    return torch.stack([
        mean_all, mean_all.abs(), std_all, (std_all - 1.0).abs(),
        q90 - q10, trend, trend.abs(), diff_std + last_level.abs() * 0.1,
    ], dim=-1)


# ══════════════════════════════════════════════════════════════
# SECTION 5 — TEXT ENCODERS (frozen LLM, pool/project split to support caching [O3])
# ══════════════════════════════════════════════════════════════

# SimpleTextEncoder / LLMTextEncoder have been moved to models/text_latent.py
# (shared by adapt_z_multimodal.py / GPT4MTS_z.py / VoT_z.py); here they are only imported, in the
# try block at the top of the file, via `from models.text_latent import SimpleTextEncoder,
# LLMTextEncoder`, under the same names; behaviour is identical to the original inline definitions.


# ══════════════════════════════════════════════════════════════
# SECTION 6 — ADAPTERS (the method itself; [C2] no text-statistics/RevIN branch at all)
#   (a) AddFusionAdapter : additive fusion (M1-M3)
#   (b) IGSoftMoEFusionAdapter : adaptive fusion of heterogeneous experts (M4-M5, ours)
# ══════════════════════════════════════════════════════════════

class AddFusionAdapter(nn.Module):
    """Additive fusion adapter: h = h_feat + h_grad (+ h_text)."""
    def __init__(self, d_model, z_shape, text_dim=64, hidden=64, dropout=0.0,
                 use_text=True, use_text_grad=True, device="cpu"):
        super().__init__()
        self.use_text, self.use_text_grad = use_text, use_text_grad
        D_z = z_shape[-1]
        self.linear = nn.Linear(D_z, hidden)
        self.linear_ = nn.Linear(d_model, hidden)
        # [O9] built directly as (N, D), same semantics as the original torch.ones(z_shape)[0]
        self.lr = nn.Parameter(torch.ones(z_shape[1:], dtype=torch.float32))
        self.drop = nn.Dropout(dropout)
        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden)
            self.txt_lr = nn.Parameter(torch.ones(text_dim) * 0.3)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, D_z)
        self.last_gate = None
        self.aux_loss = torch.zeros(())

    def forward(self, x, his_grad, text_emb=None, his_text_grad=None,
                offset_feat=None):
        B = x.size(0)
        g = (his_grad * self.lr).unsqueeze(0).expand(B, -1, -1)
        h_g = self.linear_(g)
        h_f = self.linear(x)
        if self.use_text and text_emb is not None:
            N = x.size(1)
            if self.use_text_grad and his_text_grad is not None:
                scale = 1.0 + self.txt_lr * his_text_grad
                text_emb = text_emb * scale.unsqueeze(0)
            h_t = self.text_proj(text_emb.unsqueeze(1).expand(B, N, -1))
            h = h_f + h_g + h_t
        else:
            h = h_f + h_g
        h = self.drop(torch.relu(h))
        h = self.drop(torch.relu(self.linear2(h)))
        return self.linear3(h)


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    sparsemax(z)_i = max(z_i - tau, 0), with tau chosen so the outputs sum to 1 (Martins & Astudillo 2016).
    Unlike softmax, it can output **exact zeros**, which adaptively switches off some experts
    — the number of effective experts is decided by the data, with no fixed top-k.
    """
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device,
                     dtype=logits.dtype)
    shape = [1] * logits.dim()
    shape[dim] = -1
    k = k.view(shape)
    support = (1 + k * z_sorted) > z_cumsum
    k_support = support.to(logits.dtype).sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (z_cumsum.gather(dim, (k_support.long() - 1)) - 1.0) / k_support
    return torch.clamp(z - tau, min=0.0)


# ── heterogeneous experts (per-variate embedding axis D_z) ──

class IdentityExpert(nn.Module):
    """Zero-initialized residual expert: lets the gate choose "no correction"."""
    def __init__(self, hidden, d_z):
        super().__init__()
        self.out = nn.Linear(hidden, d_z)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, feature, g_h, txt_mod, f_h):
        return self.out(h)


class TrendExpert(nn.Module):
    """Low-rank bottleneck → smooth trend-type correction."""
    def __init__(self, hidden, d_z, rank=2):
        super().__init__()
        self.pre = nn.Linear(hidden, d_z)
        self.down = nn.Linear(d_z, rank, bias=False)
        self.up = nn.Linear(rank, d_z, bias=False)

    def forward(self, h, feature, g_h, txt_mod, f_h):
        return self.up(self.down(self.pre(h)))


class SeasonalityExpert(nn.Module):
    """Truncation to low Fourier modes → periodic correction."""
    def __init__(self, hidden, d_z, num_modes=8):
        super().__init__()
        self.d_z = d_z
        self.n_freq = d_z // 2 + 1
        self.num_modes = min(num_modes, self.n_freq)
        self.gain = nn.Parameter(torch.zeros(self.num_modes, 2))
        self.mix = nn.Linear(hidden, d_z)

    def forward(self, h, feature, g_h, txt_mod, f_h):
        Xf = torch.fft.rfft(feature, dim=-1)
        gain = torch.view_as_complex(self.gain.contiguous())
        mask = torch.zeros(self.n_freq, dtype=Xf.dtype, device=Xf.device)
        mask[: self.num_modes] = 1.0 + gain
        seasonal = torch.fft.irfft(Xf * mask, n=self.d_z, dim=-1)
        return seasonal + self.mix(h)


class FluctuationExpert(nn.Module):
    """High-pass + latest Z gradient → error-driven high-frequency correction."""
    def __init__(self, hidden, d_z):
        super().__init__()
        self.hp = nn.Linear(d_z, hidden)
        self.body = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                                  nn.Linear(hidden, d_z))

    def forward(self, h, feature, g_h, txt_mod, f_h):
        highpass = feature - feature.mean(dim=-1, keepdim=True)
        return self.body(torch.cat([torch.relu(self.hp(highpass)), g_h],
                                   dim=-1))


class TextFiLMExpert(nn.Module):
    """The only expert holding the text channel: text → FiLM(γ, β) modulation of the features."""
    def __init__(self, hidden, d_z, text_dim):
        super().__init__()
        self.film = nn.Linear(text_dim, hidden * 2)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_z))

    def forward(self, h, feature, g_h, txt_mod, f_h):
        gamma, beta = self.film(txt_mod).unsqueeze(1).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        return self.out(f_h * (1.0 + gamma) + beta)


# ══════════════════════════════════════════════════════════════
# SECTION 6b — CLIP-style TS–text contrastive alignment (a simplified VoT Endogenous Text
# Alignment: no trend/seasonal-decomposition contrast; a single symmetric InfoNCE directly on the pooled
# TS representation and the text representation, forcing both into one shared space)
# ══════════════════════════════════════════════════════════════

# VICRegAlign (+ the helper functions _dominant_period/_jump_score) has been moved to
# models/text_alignment.py (the same file also adds the full
# MultiComponentVICRegAlign alignment used by VoT_z); here it is imported under the same name via
# `from models.text_alignment import VICRegAlign` in the try block at the top of the file.


class IGSoftMoEFusionAdapter(nn.Module):
    """
    Importance-Gated Soft MoE (IG-SMoE) fusion adapter — the core contribution of this work.
    Compared with the fixed top-k MoEFusionAdapter:
      · removes every fixed k / top-k restriction; all experts are fused softly;
      · the gate g(·) takes [per-variate features + pooled_x + pooled_text +
        his_grad statistics + offset distribution shift], reusing the O2/O3/O4/O5 caches and snapshots;
      · s = g(...) + α·grad_modulation;  w = softmax(s/τ) or sparsemax(s);
      · z_fused = Σ_i w_i·Expert_i  (weighted sum; the weights are the importances);
      · adaptive number of effective experts N_eff = 1/Σ_i w_i²  → self.eff_experts;
      · auxiliary loss (the opposite of the old "load balancing, encourage uniformity"; it now encourages **decisiveness**):
          aux_mode='entropy': L_aux = λ·H(w)  (negative entropy)
          aux_mode='l2'     : L_aux = -λ·Σw²
          aux_mode='none'   : 0
    [C1] text confidence does not enter this module — the caller has already soft-gated with emb*conf,
         so the adapter's response to "weak text" decays naturally, with no extra logic needed.
    """

    def __init__(self, d_model, z_shape, text_dim=64, hidden=64,
                 offset_dim=8, num_modes=8, aux_weight=0.02,
                 gate_activation="softmax", aux_mode="entropy",
                 temperature=1.0, dropout=0.1, device="cpu",
                 use_text=True, use_text_grad=True):
        super().__init__()
        assert gate_activation in ("softmax", "sparsemax")
        assert aux_mode in ("entropy", "l2", "none")
        D_z = z_shape[-1]
        self.D_z, self.text_dim = D_z, text_dim
        self.offset_dim = offset_dim
        self.temperature = temperature
        self.gate_activation = gate_activation
        self.aux_mode, self.aux_weight = aux_mode, aux_weight
        self.use_text, self.use_text_grad = use_text, use_text_grad

        self.lr = nn.Parameter(torch.ones(z_shape[1:]))
        self.txt_lr = nn.Parameter(torch.ones(text_dim))

        self.f_proj = nn.Linear(D_z, hidden)
        self.g_proj = nn.Linear(d_model, hidden)
        self.t_proj = nn.Linear(text_dim, hidden)
        self.trunk = nn.Linear(hidden * 3, hidden)
        self.drop = nn.Dropout(dropout)

        self.experts = nn.ModuleList([
            IdentityExpert(hidden, D_z),
            TrendExpert(hidden, D_z, rank=2),
            SeasonalityExpert(hidden, D_z, num_modes=num_modes),
            FluctuationExpert(hidden, D_z),
            TextFiLMExpert(hidden, D_z, text_dim),
        ])
        self.num_experts = len(self.experts)

        # Importance gate g(pooled_x, pooled_text, his_grad_stats, offset)
        self.offset_norm = nn.LayerNorm(offset_dim)
        self.gate_f = nn.Linear(D_z, hidden)  # per-variate local features
        self.gate_p = nn.Linear(D_z, hidden)  # pooled_x global features
        self.gate_g = nn.Linear(d_model, hidden)  # his_grad statistics
        self.gate_t = nn.Linear(text_dim, hidden)  # pooled_text
        self.gate_o = nn.Linear(offset_dim, hidden)  # distribution shift (concept drift)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden * 5, hidden), nn.ReLU(),
            nn.Linear(hidden, self.num_experts))

        # grad_modulation: the dual-gradient direction directly modulates the gate logits
        self.grad_mod_z = nn.Linear(d_model, self.num_experts)
        self.grad_mod_t = nn.Linear(text_dim, self.num_experts)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.last_gate = None
        self.eff_experts = float(self.num_experts)
        self.aux_loss = torch.zeros((), device=torch.device(device))

    def _activate(self, s):
        if self.gate_activation == "sparsemax":
            return sparsemax(s, dim=-1)
        return torch.softmax(s / max(self.temperature, 1e-6), dim=-1)

    def forward(self, feature, his_grad, text_emb=None, his_text_grad=None,
                offset_feat=None):
        B, N, _ = feature.shape
        device = feature.device
        if text_emb is None or not self.use_text:
            text_emb = torch.zeros(B, self.text_dim, device=device)
        if offset_feat is None:
            offset_feat = torch.zeros(B, self.offset_dim, device=device)

        g = (his_grad * self.lr).unsqueeze(0).expand(B, -1, -1)  # (B,N,d_model)
        f_h = torch.relu(self.f_proj(feature))
        g_h = torch.relu(self.g_proj(g))

        if self.use_text_grad and his_text_grad is not None:
            direction = his_text_grad / (his_text_grad.norm() + 1e-6)
            txt_mod = text_emb * (1.0 + self.txt_lr * direction).unsqueeze(0)
        else:
            txt_mod = text_emb
        txt_exp = txt_mod.unsqueeze(1).expand(-1, N, -1)
        t_h = torch.relu(self.t_proj(txt_exp))

        h = self.drop(torch.relu(self.trunk(
            torch.cat([f_h, g_h, t_h], dim=-1))))

        outs = [e(h, feature, g_h, txt_mod, f_h) for e in self.experts]
        experts = torch.stack(outs, dim=2)  # (B,N,E,D_z)

        # ── Importance gate (no top-k) ──
        pooled_x = feature.mean(dim=1, keepdim=True).expand(-1, N, -1)
        o_exp = self.offset_norm(offset_feat).unsqueeze(1).expand(-1, N, -1)
        gate_h = torch.cat([
            torch.relu(self.gate_f(feature)),
            torch.relu(self.gate_p(pooled_x)),
            torch.relu(self.gate_g(g)),
            torch.relu(self.gate_t(txt_exp)),
            torch.relu(self.gate_o(o_exp)),
        ], dim=-1)
        s = self.gate_net(gate_h)  # (B,N,E)

        grad_bias = self.grad_mod_z(g.mean(dim=1))  # (B,E)
        if self.use_text_grad and his_text_grad is not None:
            grad_bias = grad_bias + self.grad_mod_t(
                his_text_grad.unsqueeze(0).expand(B, -1))
        s = s + self.alpha * grad_bias.unsqueeze(1)  # broadcast (B,1,E)

        w = self._activate(s)  # (B,N,E)
        self.last_gate = w.detach()

        eff = 1.0 / (w.pow(2).sum(dim=-1) + 1e-9)
        self.eff_experts = float(eff.mean().detach())

        if self.aux_mode == "entropy":
            H = -(w * (w + 1e-9).log()).sum(dim=-1).mean()
            self.aux_loss = self.aux_weight * H
        elif self.aux_mode == "l2":
            self.aux_loss = -self.aux_weight * w.pow(2).sum(dim=-1).mean()
        else:
            self.aux_loss = torch.zeros((), device=device)

        return (experts * w.unsqueeze(-1)).sum(dim=2)  # (B,N,D_z)


def _position_candidates(backbone, e_layers):
    """Set of candidate injection layers for PosGate (iAdapter multi-position adaptive fusion), sorted and deduplicated.

    The number of SOFTS encoder layers is hard-coded to 3 in models/SOFTS_z.py::Model.__init__
    (independent of the e_layers passed in), so 3 is used for it; the other backbones that support PosGate
    (iTransformer/Transformer/TimesNet) use the actual e_layers. Three positions are taken: {first layer, middle layer,
    last layer}; when e_layers<=2 the set naturally reduces to 1–2 positions after deduplication.

    VoT_z is not indexed by layer (it has a single frozen GPT2 trunk and no layer-by-layer injectable
    encoder stack); instead it has 4 shape-compatible candidate injection points in models/VoT_z.py
    (POS_TREND/POS_SEASONAL/POS_PRE_GPT2/POS_POST_GPT2), and all 4 are always returned.
    """
    if backbone == "VoT_z":
        return [1, 2, 3, 4]
    max_layers = 3 if backbone == "SOFTS" else max(int(e_layers), 1)
    mid = max(1, max_layers // 2)
    return sorted(set([1, mid, max_layers]))


class PositionAdaptiveMoEFusionAdapter(nn.Module):
    """The "multi-position adaptive fusion" optimization of iAdapter (design decision 6).

    Background: each existing backbone exposes only one feature snapshot, at the single configured z_loc
    (there is no interface exposing a hidden state per layer), so this module does not "compute a separate
    correction per layer"; it "uses the same signal for adaptive selection at several points":
      1. reuse IGSoftMoEFusionAdapter (`self.base`) on this single feature snapshot
         to compute one base correction z_base = Σ_i w_i·Expert_i (B,N,D_z);
      2. an extra position-gate network gate = sigmoid(GateNet(pooled_feature,
         text_emb)) (B, len(positions)); each position decides independently "whether to inject
         here and how much" (∈[0,1], without competing for a shared budget);
      3. forward returns {layer_idx: gate[:,i]·z_base}, and the backbone, in its
         `isinstance(z, dict)` branch, additively injects it into each matching layer (layers that do not match
         receive no correction).

    [FIX] Previously the gate used softmax (the weights of the len(positions) positions always summed to 1),
    which is equivalent to spreading the strength of the single z_base correction over the positions (with 3 candidate positions each
    gets only ~1/3 of the strength), whereas the single-position iAdapter (fusion="moe") injects
    z_base at full strength at its only injection point — this is the direct reason why, in the PosGate ablation, more adaptivity meant weaker results, not
    a fault of "adaptive position selection" itself. After switching to an independent sigmoid per position, the gate is initialized at
    ~0.5 (instead of being forced down to ~1/N), and several positions may approach full strength 1.0 at the same time,
    so the training objective is purely "should this position correct or not" rather than "how to split the budget".

    `last_gate`/`aux_loss` are passed through from `self.base` (expert gate / decisiveness auxiliary loss),
    keeping the same downstream reading contract as IGSoftMoEFusionAdapter (in OnlineMM.online(),
    `getattr(self.adapter, "last_gate", None)` is used for the gate_history record).
    """

    def __init__(self, d_model, z_shape, positions, text_dim=64, hidden=64,
                 offset_dim=8, num_modes=8, aux_weight=0.02,
                 gate_activation="softmax", aux_mode="entropy",
                 temperature=1.0, dropout=0.1, device="cpu",
                 use_text=True, use_text_grad=True):
        super().__init__()
        assert len(positions) >= 1
        self.positions = list(positions)
        self.text_dim = text_dim
        self.base = IGSoftMoEFusionAdapter(
            d_model=d_model, z_shape=z_shape, text_dim=text_dim,
            hidden=hidden, offset_dim=offset_dim, num_modes=num_modes,
            aux_weight=aux_weight, gate_activation=gate_activation,
            aux_mode=aux_mode, temperature=temperature, dropout=dropout,
            device=device, use_text=use_text, use_text_grad=use_text_grad)

        D_z = z_shape[-1]
        self.pos_gate_net = nn.Sequential(
            nn.Linear(D_z + text_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, len(self.positions)))
        self.last_pos_gate = None
        self.aux_loss = torch.zeros((), device=torch.device(device))

    @property
    def last_gate(self):
        return self.base.last_gate

    @property
    def eff_experts(self):
        return self.base.eff_experts

    def forward(self, feature, his_grad, text_emb=None, his_text_grad=None,
                offset_feat=None):
        B = feature.size(0)
        device = feature.device

        z_base = self.base(feature, his_grad, text_emb, his_text_grad,
                           offset_feat)                     # (B,N,D_z)
        self.aux_loss = self.base.aux_loss

        txt = (text_emb if text_emb is not None
              else torch.zeros(B, self.text_dim, device=device))
        pooled = feature.mean(dim=1)                        # (B, D_z)
        # [FIX] independent sigmoid per position ∈[0,1], instead of a softmax across positions (summing to 1).
        # softmax forcibly spreads z_base over the number of positions (average weight per position
        # only 1/len(positions)), so "adaptive positions" were always weaker than the single-point, full-strength
        # iAdapter; sigmoid lets each position decide its injection strength independently, and several can approach 1.0 at once.
        gate = torch.sigmoid(
            self.pos_gate_net(torch.cat([pooled, txt], dim=-1))
        )                                                    # (B, len(positions))
        self.last_pos_gate = gate.detach()

        return {pos: gate[:, i].view(B, 1, 1) * z_base
                for i, pos in enumerate(self.positions)}


# ══════════════════════════════════════════════════════════════
# SECTION 7 — ONLINE MULTIMODAL ADAPTATION (iAdapter, optimized OnlineMM)
# ══════════════════════════════════════════════════════════════

class OnlineMM:
    """
    Multimodal online forecasting (native real-text version).

    Key design (method unchanged):
      · update_backbone=False (default) — the backbone and projection head are not updated online;
        adaptation comes entirely from the adaptive expert-fusion adapter, strictly distinct from "online fine-tuning";
      · dual-gradient feedback: his_grad = ∂L/∂z, his_text_grad = ∂L/∂text_emb;
      · few-shot anti-overfitting: dropout / weight_decay / grad clip / patience.

    Changes in this version:
      [C1] _encode_text returns an (emb, conf) pair, where conf is the provider's
           EMA confidence (B,1); every call site soft-gates with emb*conf before entering the adapter and the
           text-gradient path — a smooth replacement for a hard 0/1 mask.
      [C2] no text-statistics branch: text does not directly shift/scale the output distribution.
      [C3] self.reslog (optional ResonanceLogger): the online main loop records, per batch,
           text energy and counterfactual |Δpred| for resonance visualization; default None, zero overhead.
      [C4] self.clip_align (optional VICRegAlign): invariance/variance/covariance three-term
           loss aligning the pooled TS representation with the text representation (replaces the original CLIP-style symmetric InfoNCE,
           whose gradient vanished when all negatives were similar), added to the main loss (self.clip_weight,
           adapted to the in-batch text variance, see _adaptive_clip_weight); self.embed_log
           (optional EmbedAlignLogger) collects shared-space vectors for the
           plot_clip_alignment_space visualization; both enabled by default with zero extra forward cost
           (they reuse the already computed feature/text_emb).

    Implementation optimizations (see [O1]-[O6] at the top of the file):
      · both gradient feedbacks now use torch.autograd.grad — no pollution of parameter gradients,
        no retain_graph, and his_grad no longer accumulates without bound (--his_mode selectable);
      · the feature-extraction forward is wrapped in no_grad (adapter parameter gradients do not depend on the feature graph);
      · frozen LLM pooling is cached by global row index and shared across ablation modes;
      · the buffer stores only the needed number of batches + his snapshots (N,D); text indices are strictly aligned with numeric slices.
    """

    def __init__(self, backbone, d_model, args,
                 provider: RealTextProvider, split_offsets: dict,
                 fusion="add",                       # "moe" | "add"
                 text_encoder_type="BERT",           # "BERT"|"GPT2"|"GPT2M"|"MODERNBERT"|"T5"|"BART"|"simple"
                 text_dim=64, llm_layers=6,
                 use_text=True, use_text_grad=True,
                 update_backbone=False,
                 dropout=0.1, weight_decay=1e-4, grad_clip=1.0, patience=3,
                 top_k=2, aux_weight=0.02,
                 gate_activation="softmax", aux_mode="entropy", temperature=1.0,
                 his_mode="ema", his_momentum=0.9, use_text_cache=True,
                 enc_in=1,
                 use_clip_align=True, clip_dim=64, clip_weight=0.1):

        self.backbone = backbone
        # Backbones such as GPT4MTS_z/VoT_z consume text_emb directly (USES_TEXT=True),
        # which is independent of the external use_text (whether IGSoftMoEFusionAdapter/AddFusionAdapter
        # also use text) — even when the external use_text=False (see
        # default_modes_for_backbone), text_enc must still be built and (text_emb, conf) obtained
        # to feed the backbone.
        self.backbone_uses_text = getattr(backbone, "USES_TEXT", False)
        self.args = args
        self.device = args.device
        self.provider = provider
        self._offsets = dict(split_offsets)
        self._cur_offset = self._offsets.get("val", 0)
        self.fusion = fusion
        self.use_text = use_text
        self.use_text_grad = use_text_grad
        self.update_backbone = update_backbone
        self.grad_clip = grad_clip
        self.patience = patience
        assert his_mode in ("ema", "latest", "sum")
        self.his_mode, self.his_momentum = his_mode, his_momentum
        self.use_text_cache = use_text_cache
        self.loss_func = nn.MSELoss()

        z_shape_list = list(args.z_shape)
        z_shape_list[0] = max(args.batch_size, args.batch_size2)
        args.z_shape = tuple(z_shape_list)
        self.z = nn.Parameter(torch.zeros(args.z_shape, requires_grad=True,
                                          device=self.device))

        if fusion == "moe":
            self.adapter = IGSoftMoEFusionAdapter(
                d_model=d_model, z_shape=args.z_shape, text_dim=text_dim,
                hidden=64, dropout=dropout,
                gate_activation=gate_activation, temperature=temperature,
                aux_mode=aux_mode, aux_weight=aux_weight,
                use_text=use_text, use_text_grad=use_text_grad,
                device=self.device).to(self.device)
        elif fusion == "moe_pos":
            # iAdapter+PosGate (optimization): requires a backbone that supports multi-position dict-z injection
            # (ITransformerZ/TransformerZ/TimesNet_z/SOFTS_z/VoT_z); PatchTST/
            # GPT4MTS_z is explicitly excluded by design decision 7; fail fast when unsupported.
            if not getattr(backbone, "SUPPORTS_POS_GATE", False):
                raise ValueError(
                    f"fusion='moe_pos' (iAdapter+PosGate) requires a backbone that supports "
                    f"multi-position dict-z injection; the current backbone="
                    f"{type(backbone).__name__} does not (PatchTST/GPT4MTS_z "
                    f"are explicitly excluded by design; see design decision 7).")
            positions = getattr(args, "pos_candidates", None) or [args.z_loc]
            self.adapter = PositionAdaptiveMoEFusionAdapter(
                d_model=d_model, z_shape=args.z_shape, positions=positions,
                text_dim=text_dim, hidden=64, dropout=dropout,
                gate_activation=gate_activation, temperature=temperature,
                aux_mode=aux_mode, aux_weight=aux_weight,
                use_text=use_text, use_text_grad=use_text_grad,
                device=self.device).to(self.device)
        else:
            self.adapter = AddFusionAdapter(
                d_model=d_model, z_shape=args.z_shape, text_dim=text_dim,
                hidden=64, dropout=dropout, use_text=use_text,
                use_text_grad=use_text_grad,
                device=self.device).to(self.device)

        # ── Text encoder (real text; simple uses the provider's real-corpus vocabulary) [O1] ──
        self.text_enc = None
        self._cache_key = None
        if use_text or self.backbone_uses_text:
            if text_encoder_type == "simple":
                self.text_enc = SimpleTextEncoder(
                    provider.vocab_size, embed_dim=text_dim,
                    output_dim=text_dim, max_len=provider.SIMPLE_MAX_LEN,
                    dropout=dropout).to(self.device)
                self._cache_key = ("simple_ids", provider.SIMPLE_MAX_LEN)
            else:
                self.text_enc = LLMTextEncoder(
                    llm_tag=text_encoder_type, output_dim=text_dim,
                    llm_layers=llm_layers, dropout=dropout,
                    device=self.device).to(self.device)
                self._cache_key = ("pooled", text_encoder_type, llm_layers)

        # ── VICReg-style TS–text alignment (invariance + variance + covariance three-term loss, replacing the original CLIP-
        #    style symmetric InfoNCE — frozen LLM pooled vectors are generally highly clustered, so InfoNCE gradients vanish
        #    when all negatives are similar, while the VICReg variance term directly prevents collapse) ──
        self.clip_weight = clip_weight
        self.clip_align = (VICRegAlign(d_z=d_model, text_dim=text_dim,
                                       d_clip=clip_dim,
                                       lam=25.0, mu=25.0, nu=1.0).to(self.device)
                           if (use_text and use_clip_align) else None)

        self._adapter_params = list(self.adapter.parameters())
        if self.clip_align is not None:
            self._adapter_params += list(self.clip_align.parameters())
        self._text_params = ([p for p in self.text_enc.parameters()
                              if p.requires_grad]
                             if self.text_enc is not None else [])
        self._weight_decay = weight_decay
        self._settings_data = args.data.lower()
        self.text_dim = text_dim

        # [C3] resonance-logging hook (finetune_mm attaches a ResonanceLogger to M5)
        self.reslog = None
        # CLIP-alignment semantic-space visualization hook (finetune_mm attaches it in clip_align modes)
        self.embed_log = None

        # training traces, for visualization
        self.gate_history = []
        self.text_grad_history = []

    # ── helpers ──────────────────────────────────────────────

    def _rows_of_windows(self, window_idxs):
        """Window start index -> global row index of the last input row (clamped to the valid range). [O1]"""
        last = self.provider.n_rows - 1
        s = self.args.seq_len - 1
        return [min(max(self._cur_offset + int(w) + s, 0), last)
                for w in window_idxs]

    def _dup_text_mask(self, rows):
        """Rows whose text contents are pairwise identical -> (B,B) bool mask (True = duplicate).
        Inherited from the CLIPAlign era as a mutual-exclusion mask for negatives; VICRegAlign does no negative
        contrast, so it is now only passed through as a compatibility argument (dup_mask is not used by VICRegAlign)."""
        texts = [self.provider.text_for_row(r) for r in rows]
        B = len(texts)
        mask = torch.zeros(B, B, dtype=torch.bool, device=self.device)
        for i in range(B):
            for j in range(i + 1, B):
                if texts[i] == texts[j]:
                    mask[i, j] = mask[j, i] = True
        return mask

    def _adaptive_clip_weight(self, text_emb: torch.Tensor,
                              base_weight: float = 0.1) -> float:
        """Measure the in-batch text-embedding variance and adjust the VICReg alignment weight dynamically.
        The lower the variance (the worse the collapse) → the larger the weight, pushing harder to spread out; in normal operation the weight
        stays near base_weight and does not interfere with the main task loss.
        Empirical threshold: var < 0.05 counts as collapse; normal training is about 0.3-1.0."""
        with torch.no_grad():
            var = text_emb.var(dim=0).mean().item()
        collapse_signal = max(0.0, 1.0 - var / 0.3)
        return base_weight * (1.0 + 4.0 * collapse_signal)  # at most 5× the base weight

    def _encode_text(self, window_idxs):
        """Real text -> (emb, conf); the frozen part is cached by global row index [O3].
        [C1] conf: the provider's EMA confidence tensor (B,1). The caller soft-gates with emb*conf
        — exponential decay during uninformative periods instead of hard zeroing, smooth rise during informative ones.
        Modes without text always return (None, None)."""
        if not (self.use_text or self.backbone_uses_text) or self.text_enc is None:
            return None, None
        rows = self._rows_of_windows(window_idxs)
        conf = torch.tensor(
            [self.provider.conf_for_row(r) for r in rows],
            device=self.device, dtype=torch.float32).unsqueeze(-1)  # (B,1)
        cache = (self.provider.enc_cache.setdefault(self._cache_key, {})
                 if self.use_text_cache else {})
        if isinstance(self.text_enc, LLMTextEncoder):
            missing = [r for r in dict.fromkeys(rows) if r not in cache]
            if missing:
                pooled = self.text_enc.pool(
                    [self.provider.text_for_row(r) for r in missing])
                for i, r in enumerate(missing):
                    cache[r] = pooled[i].detach()
            stacked = torch.stack([cache[r] for r in rows]).to(self.device)
            return self.text_enc.project(stacked), conf
        # simple: cache token ids
        for r in dict.fromkeys(rows):
            if r not in cache:
                cache[r] = self.provider.tokenize(
                    self.provider.text_for_row(r))
        ids = torch.tensor([cache[r] for r in rows], dtype=torch.long,
                           device=self.device)
        return self.text_enc(ids), conf

    def _update_his(self, his, g):
        """[O2] his_grad update: ema (default) / latest / sum (old behaviour, unbounded)."""
        if g is None:
            return his
        if self.his_mode == "sum":
            return his + g
        if self.his_mode == "latest":
            return g
        return self.his_momentum * his + (1.0 - self.his_momentum) * g

    def _bb_call(self, x, z, z_loc, text_emb=None, text_conf=None):
        """Unified backbone forward call: backbones with USES_TEXT=True (GPT4MTS_z/
        VoT_z) additionally receive (text_emb, text_conf) and consume them directly; other backbones keep
        the original (x, z=, z_loc=) call unchanged. Passing None for text_emb/text_conf (e.g. in
        _z_grad's pure z-gradient probe, which involves no text) is equivalent to turning off the prompt branch and does not affect
        the original semantics."""
        if self.backbone_uses_text:
            return self.backbone(x, z=z, z_loc=z_loc,
                                 text_emb=text_emb, text_conf=text_conf)
        return self.backbone(x, z=z, z_loc=z_loc)

    def _z_grad(self, x_b, y_b):
        """[O2] ∂L/∂z via autograd.grad: no accumulation, no pollution of parameters, no retain_graph.
        Returns (grad(N,D)|None, feature.detach())."""
        out = self._bb_call(x_b, self.z, self.args.z_loc)
        loss = self.loss_func(out["pred"], y_b)
        gz = torch.autograd.grad(loss, self.z, allow_unused=True)[0]
        gz = gz.mean(dim=0).detach() if gz is not None else None
        return gz, out["feature"].detach()

    def _compute_text_grad(self, x_b, y_b, feature_b, his_grad, text_emb,
                           his_text_grad=None, offset_b=None):
        """[O2] ∂L/∂text_emb via autograd.grad (modulation context identical to inference).
        [C1] the text_emb argument has already been multiplied by the EMA confidence by the caller — the gradient magnitude
        scales naturally with conf, so feedback is weaker in low-confidence periods, with no extra gating needed."""
        txt = text_emb.detach().clone().requires_grad_(True)
        z = self.adapter(feature_b, his_grad, txt, his_text_grad, offset_b)
        out = self._bb_call(x_b, z, self.args.z_loc, txt)
        loss = self.loss_func(out["pred"], y_b)
        gt = torch.autograd.grad(loss, txt, allow_unused=True)[0]
        if gt is not None:
            return gt.mean(dim=0).detach()
        return torch.zeros(txt.shape[-1], device=self.device)

    def _offset(self, x):
        return (compute_offset_features(x).to(self.device)
                if self.fusion in ("moe", "moe_pos") else None)

    def _keep_batches(self):
        """[O4] number of batches the buffer needs: slices only use the most recent pred_len+batch rows."""
        need = self.args.pred_len + self.args.batch_size
        return int(math.ceil(need / self.args.batch_size)) + 1

    @staticmethod
    def _buffer_slice(arrs, sl):
        return np.concatenate(arrs, axis=0)[sl]

    def _buf_text_windows(self, global_idx, Bb):
        """[O5] exact window indices of the buffer samples: last row = global_idx-1, step back pred, then take Bb of them."""
        b0 = max(0, global_idx - self.args.pred_len - Bb)
        return range(b0, b0 + Bb)

    # ── val / warm-up phase ──────────────────────────────────

    def val(self, val_loader, mode="val"):
        print(f"[MM] val (fusion={self.fusion}, text={self.use_text}, "
              f"text_grad={self.use_text_grad}, his={self.his_mode})")
        self._cur_offset = self._offsets["val"]
        lr_key = "finetune_lr" if mode == "train" else "adapter_lr"
        adp_lr = settings[self.args.backbone][self._settings_data][lr_key]
        wd = self._weight_decay
        all_params = self._adapter_params + self._text_params
        self.optimizer = optim.Adam(all_params, lr=adp_lr, weight_decay=wd)
        self.optimizer2 = (optim.Adam(                      # [O6] only needed for train
            self.backbone.parameters(),
            lr=settings[self.args.backbone][self._settings_data]["online_lr"],
            weight_decay=wd) if mode == "train" else None)

        his_grad = torch.zeros(self.args.z_shape[1:], device=self.device)
        his_text_grad = (torch.zeros(self.text_dim, device=self.device)
                         if self.use_text else None)

        preds, truths, x_list = [], [], []
        keep = self._keep_batches()
        global_idx = 0
        for _, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            B = x.size(0)
            x_list.append(x.detach().cpu().numpy())
            truths.append(y.detach().cpu().numpy())

            text_emb, text_conf = self._encode_text(
                range(global_idx, global_idx + B))
            text_emb_in = (text_emb * text_conf
                           if text_emb is not None else None)   # [C1]
            offset = self._offset(x)

            with torch.no_grad():                           # [O2] feature extraction builds no graph
                feat = self._bb_call(x, self.z, self.args.z_loc,
                                     text_emb, text_conf)["feature"]
            z = self.adapter(feat, his_grad, text_emb_in, his_text_grad, offset)
            out = self._bb_call(x, z, self.args.z_loc, text_emb, text_conf)
            pred = out["pred"]
            preds.append(pred.detach().cpu().numpy())
            _replay_event("val_prediction", locals())

            clip_loss = feat.new_zeros(())
            eff_weight = self.clip_weight
            if self.clip_align is not None and text_emb is not None:
                rows = self._rows_of_windows(range(global_idx, global_idx + B))
                clip_loss = self.clip_align(
                    feat, text_emb, text_conf, self._dup_text_mask(rows))
                eff_weight = self._adaptive_clip_weight(
                    text_emb, self.clip_weight)

            loss = (self.loss_func(pred, y)
                    + getattr(self.adapter, "aux_loss", 0.0)
                    + out.get("aux_loss", 0.0)
                    + eff_weight * clip_loss)
            self.optimizer.zero_grad()
            if self.optimizer2:
                self.optimizer2.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(all_params, self.grad_clip)
            self.optimizer.step()
            if self.optimizer2:
                self.optimizer2.step()
            else:
                self.backbone.zero_grad(set_to_none=True)      # clear orphaned gradients

            if len(x_list) > keep:                          # [O4]
                x_list = x_list[-keep:]

            total_seen = sum(a.shape[0] for a in x_list)
            if total_seen > self.args.pred_len + self.args.batch_size:
                sl = slice(-self.args.pred_len - self.args.batch_size,
                           -self.args.pred_len)
                x_b = torch.from_numpy(
                    self._buffer_slice(x_list, sl)).to(self.device)
                y_b = torch.from_numpy(
                    self._buffer_slice(truths[-keep:], sl)).to(self.device)

                gz, feat_b = self._z_grad(x_b, y_b)         # [O2]
                his_grad = self._update_his(his_grad, gz)

                if self.use_text and self.use_text_grad:
                    emb_buf, conf_buf = self._encode_text(
                        self._buf_text_windows(global_idx, x_b.size(0)))
                    if emb_buf is not None:
                        his_text_grad = self._compute_text_grad(
                            x_b, y_b, feat_b, his_grad,
                            emb_buf * conf_buf,              # [C1]
                            his_text_grad, self._offset(x_b))

                # VoT_z Historical In-Context memory write: must use the window delayed by pred_len steps
                # (consistent with the strict online protocol of his_grad above); otherwise, with a stride-1
                # sliding window, the current y would leak directly into the next window and be retrieved immediately.
                if hasattr(self.backbone, "update_memory"):
                    mem_emb, _ = self._encode_text(
                        self._buf_text_windows(global_idx, x_b.size(0)))
                    if mem_emb is not None:
                        self.backbone.update_memory(mem_emb, y_b)
            _replay_event("step_end", locals())
            global_idx += B

        mae, mse = calculate_metrics(np.concatenate(preds, axis=0),
                                     np.concatenate(truths, axis=0))
        print(f"[MM] val  mae={mae:.4f}  mse={mse:.4f}")
        return mse

    # ── online prediction phase ──────────────────────────────

    def online(self, test_loader):
        print(f"[MM] online (fusion={self.fusion}, text={self.use_text}, "
              f"text_grad={self.use_text_grad}, "
              f"update_backbone={self.update_backbone}, his={self.his_mode})")
        self._cur_offset = self._offsets["test"]

        if self.args.backbone == "iTransformer":
            proj_params = list(self.backbone.projector.parameters())
        elif self.args.backbone in ("GPT4MTS_z", "VoT_z"):
            proj_params = list(self.backbone.out_layer.parameters())
        else:
            proj_params = list(self.backbone.projection.parameters())

        wd = self._weight_decay
        all_adp = self._adapter_params + self._text_params
        self.optimizer = optim.Adam(
            all_adp,
            lr=settings[self.args.backbone][self._settings_data]["adp_online_lr"],
            weight_decay=wd)
        self.optimizer2 = optim.Adam(
            proj_params,
            lr=settings[self.args.backbone][self._settings_data]["online_lr"],
            weight_decay=wd)

        his_grad = torch.zeros(self.args.z_shape[1:], device=self.device)
        his_text_grad = (torch.zeros(self.text_dim, device=self.device)
                         if self.use_text else None)
        z0 = torch.zeros(self.args.z_shape, device=self.device)   # [O4] reused

        preds, truths = [], []
        x_list, feat_list = [], []                 # rolling buffer (np)
        snap_list, cnt_list = [], []               # [O4] his snapshots (N,D) + row counts
        self.gate_history, self.text_grad_history = [], []
        keep = self._keep_batches()
        global_idx = 0

        for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            B = x.size(0)

            text_emb, text_conf = self._encode_text(
                range(global_idx, global_idx + B))
            text_emb_in = (text_emb * text_conf
                           if text_emb is not None else None)   # [C1]
            offset = self._offset(x)

            with torch.no_grad():                           # [O2]
                feat = self._bb_call(x, z0, self.args.z_loc,
                                     text_emb, text_conf)["feature"]
            z = self.adapter(feat, his_grad, text_emb_in, his_text_grad, offset)
            pred = self._bb_call(x, z, self.args.z_loc, text_emb, text_conf)["pred"]

            preds.append(pred.detach().cpu().numpy())
            truths.append(y.detach().cpu().numpy())
            x_list.append(x.detach().cpu().numpy())
            feat_list.append(feat.cpu().detach().numpy())
            snap_list.append(his_grad.detach().cpu().numpy())      # (N,D)
            cnt_list.append(B)
            _replay_event("test_prediction", locals())
            if getattr(self.adapter, "last_gate", None) is not None:
                lg = self.adapter.last_gate
                self.gate_history.append(
                    lg.mean(dim=tuple(range(lg.dim() - 1))).cpu().numpy())

            # [C3] resonance logging: must come after gate_history — the logger's internal
            # counterfactual forward calls the adapter again and overwrites last_gate / aux_loss.
            if self.reslog is not None:
                self.reslog.on_step(
                    mm=self, x=x, y=y, pred=pred,
                    text_emb_in=text_emb_in, text_conf=text_conf,
                    his_grad=his_grad, his_text_grad=his_text_grad,
                    z0=z0, offset=offset,
                    rows=self._rows_of_windows(
                        range(global_idx, global_idx + B)))

            if len(x_list) > keep:                          # [O4]
                x_list, feat_list = x_list[-keep:], feat_list[-keep:]
                snap_list, cnt_list = snap_list[-keep:], cnt_list[-keep:]

            total_seen = sum(cnt_list)
            if total_seen > self.args.pred_len + self.args.batch_size:
                sl = slice(-self.args.pred_len - self.args.batch_size,
                           -self.args.pred_len)
                x_b = torch.from_numpy(
                    self._buffer_slice(x_list, sl)).to(self.device)
                y_b = torch.from_numpy(
                    self._buffer_slice(truths[-keep:], sl)).to(self.device)
                ft_b = torch.from_numpy(
                    self._buffer_slice(feat_list, sl)).to(self.device)
                # [O4] expanding the snapshots per row and averaging the slice == the original hg_b.mean(dim=0)
                hg_rows = np.repeat(np.stack(snap_list, 0),
                                    cnt_list, axis=0)[sl]
                hg_mean = torch.from_numpy(
                    hg_rows.mean(axis=0)).to(self.device)
                offset_b = self._offset(x_b)

                # ── Gradient feedback 1: ∂L/∂z (autograd.grad, no accumulation/pollution) [O2] ──
                gz, _ = self._z_grad(x_b, y_b)
                his_grad = self._update_his(his_grad, gz)

                # ── Gradient feedback 2: ∂L/∂text_emb (exact buffer indices [O5]) ──
                emb_buf_in = None
                emb_buf = conf_buf = None
                clip_loss = ft_b.new_zeros(())
                eff_weight = self.clip_weight
                if self.use_text or self.backbone_uses_text:
                    buf_window_idxs = self._buf_text_windows(
                        global_idx, x_b.size(0))
                    emb_buf, conf_buf = self._encode_text(buf_window_idxs)
                    if emb_buf is not None:
                        # VoT_z Historical In-Context memory write: window delayed by pred_len
                        # steps (consistent with the strict online protocol of his_grad/his_text_grad below);
                        # otherwise, with a stride-1 sliding window, the current y would leak directly
                        # into the next window and be retrieved immediately.
                        if hasattr(self.backbone, "update_memory"):
                            self.backbone.update_memory(emb_buf, y_b)
                        emb_buf_in = emb_buf * conf_buf      # [C1]
                        if self.use_text_grad:
                            his_text_grad = self._compute_text_grad(
                                x_b, y_b, ft_b, hg_mean, emb_buf_in,
                                his_text_grad, offset_b)
                            self.text_grad_history.append(
                                his_text_grad.cpu().numpy())
                        buf_rows = self._rows_of_windows(buf_window_idxs)
                        if self.clip_align is not None:
                            clip_loss = self.clip_align(
                                ft_b, emb_buf, conf_buf,
                                self._dup_text_mask(buf_rows))
                            eff_weight = self._adaptive_clip_weight(
                                emb_buf, self.clip_weight)
                        if self.embed_log is not None:
                            self.embed_log.on_step(
                                self, ft_b, emb_buf, x_b, buf_rows)

                # ── Adapter (expert fusion) update — the backbone is untouched by default ──
                z_up = self.adapter(ft_b, hg_mean, emb_buf_in,
                                    his_text_grad, offset_b)
                out_up = self._bb_call(x_b, z_up, self.args.z_loc,
                                       emb_buf, conf_buf)
                loss_up = (self.loss_func(out_up["pred"], y_b)
                           + getattr(self.adapter, "aux_loss", 0.0)
                           + out_up.get("aux_loss", 0.0)
                           + eff_weight * clip_loss)
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()
                loss_up.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_adp + proj_params,
                                             self.grad_clip)
                self.optimizer.step()
                if self.update_backbone:
                    self.optimizer2.step()
                self.backbone.zero_grad(set_to_none=True)      # clear orphaned gradients
            _replay_event("step_end", locals())
            global_idx += B

        pred_arr = np.concatenate(preds, 0)
        truth_arr = np.concatenate(truths, 0)
        mae, mse = calculate_metrics(pred_arr, truth_arr)
        print(f"[MM] online  mae={mae:.4f}  mse={mse:.4f}")
        return mae, mse, pred_arr, truth_arr


# ══════════════════════════════════════════════════════════════
# SECTION 8 — BACKBONE (ITransformerZ, checkpoint-compatible) + automatic pretraining [O8]
# ══════════════════════════════════════════════════════════════

class _DataEmbed_inv(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(x.permute(0, 2, 1)))


class _EncLayer(nn.Module):
    def __init__(self, d_model, nhead=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model),
                                nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        x = self.norm1(x + y)
        return self.norm2(x + self.ff(x))


class ITransformerZ(nn.Module):
    """Minimal iTransformer with additive Z injection (checkpoint-compatible).
    [C2] the output side has only the standard instance denorm — no revin_delta / revin_scale
    or other text-conditioned distribution-shift branch."""
    SUPPORTS_POS_GATE = True  # supports iAdapter+PosGate multi-position dict-z injection

    def __init__(self, seq_len, pred_len, enc_in,
                 d_model=64, e_layers=2, nhead=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.seq_len, self.pred_len, self.d_model = seq_len, pred_len, d_model
        self.embedding = _DataEmbed_inv(seq_len, d_model, dropout)
        self.layers = nn.ModuleList([_EncLayer(d_model, nhead, d_ff, dropout)
                                     for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x, z=None, z_loc=None):
        B, L, N = x.shape
        mu = x.mean(1, keepdim=True).detach()
        sigma = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5
                           ).detach()
        x = (x - mu) / sigma
        h, feature = self.embedding(x), None
        is_multi = isinstance(z, dict)
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if is_multi:
                zi = z.get(idx + 1)
                if zi is not None:
                    if feature is None:
                        feature = h.clone()
                    h = h + zi[:B]
            elif z is not None and z_loc is not None and z_loc == idx + 1:
                feature = h.clone()
                h = h + z[:B]
        if feature is None:
            feature = h.clone()
        pred = self.projector(self.norm(h)).permute(0, 2, 1)
        pred = pred * sigma[:, 0, :].unsqueeze(1) + mu[:, 0, :].unsqueeze(1)
        return {"pred": pred, "feature": feature}


class TransformerZ(nn.Module):
    """Minimal encoder-only vanilla Transformer forecaster with additive Z
    injection (checkpoint-compatible). Follows the "prediction-length expansion" technique of models/TimesNet_z.py
    (predict_linear: Linear(seq_len, seq_len+pred_len)); per-timestep tokens
    pass through e_layers standard multi-head self-attention layers (reusing _EncLayer above); z is added after the layer hit by z_loc,
    written the same way as in ITransformerZ/TimesNet_z."""
    SUPPORTS_POS_GATE = True  # supports iAdapter+PosGate multi-position dict-z injection

    def __init__(self, seq_len, pred_len, enc_in,
                 d_model=64, e_layers=2, nhead=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.seq_len, self.pred_len, self.d_model = seq_len, pred_len, d_model
        self.value_embedding = nn.Linear(enc_in, d_model)
        self.dropout = nn.Dropout(dropout)
        self.predict_linear = nn.Linear(seq_len, seq_len + pred_len)
        self.layers = nn.ModuleList([_EncLayer(d_model, nhead, d_ff, dropout)
                                     for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, enc_in)

    def _pos_embedding(self, length, device):
        pe = torch.zeros(length, self.d_model, device=device)
        position = torch.arange(0, length, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float()
                             * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, x, z=None, z_loc=None):
        B, L, N = x.shape
        mu = x.mean(1, keepdim=True).detach()
        sigma = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5
                           ).detach()
        x = (x - mu) / sigma
        h = self.dropout(self.value_embedding(x))
        h = self.predict_linear(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = h + self._pos_embedding(h.shape[1], h.device)
        feature = None
        is_multi = isinstance(z, dict)
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if is_multi:
                zi = z.get(idx + 1)
                if zi is not None:
                    if feature is None:
                        feature = h.clone()
                    h = h + zi[:B]
            elif z is not None and z_loc is not None and z_loc == idx + 1:
                feature = h.clone()
                h = h + z[:B]
        if feature is None:
            feature = h.clone()
        out = self.projection(self.norm(h))
        pred = out[:, -self.pred_len:, :]
        pred = pred * sigma + mu
        return {"pred": pred, "feature": feature}


class _PatchTSTZAdapter(nn.Module):
    """Adapter: wraps the Model of models/PatchTST_z.py (RevIN + trend/res decomposition +
    TSTEncoder) into the unified forward(x, z=, z_loc=) -> {'pred','feature'} contract.

    PatchTST_z's native per-layer z injection happens on a (batch*channel, patch_num, d_model) tensor,
    which has no reliable correspondence to the (batch, channel, d_model) convention shared by the other backbones, so this adapter
    always uses PatchTST_z's z_loc='input2' injection point instead (before RevIN normalization, injecting into the decomposed raw
    res_init/trend_init tensors, whose shape is naturally [B, enc_in, seq_len]) and ignores the externally passed
    integer z_loc. Correspondingly, _backbone_dims() returns eff_d_model=seq_len for PatchTST.
    The trend/res branches share the same injected value, and only the res branch's feature is returned as the unified external gradient
    observation point (the trend branch's featuret is not exposed separately — a known simplification)."""
    def __init__(self, seq_len, pred_len, enc_in,
                 d_model=64, e_layers=2, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        from models.PatchTST_z import Model as _PatchTSTModel
        patch_len = min(16, seq_len)
        stride = max(1, min(8, patch_len))
        cfg = argparse.Namespace(
            enc_in=enc_in, pred_len=pred_len, e_layers=e_layers,
            n_heads=n_heads, d_model=d_model, d_ff=d_ff,
            dropout=dropout, fc_dropout=dropout, head_dropout=0.0,
            norm="batchnorm", individual=False,
            patch_len=patch_len, stride=stride, padding_patch="end",
            revin=1, affine=0, subtract_last=0, kernel_size=25,
            res_attention=True,
        )
        self.inner = _PatchTSTModel(cfg, seq_len)

    @property
    def projection(self):
        return nn.ModuleList([self.inner.model_res.head, self.inner.model_trend.head])

    def forward(self, x, z=None, z_loc=None):
        out = self.inner(x, zs=z, zt=z, z_loc="input2")
        return {"pred": out["pred"], "feature": out.get("features")}


def build_backbone(backbone, sl, pl, enc_in, d_model, e_layers, text_dim=64):
    """Build the backbone module dynamically by name (iTransformer/Transformer use inline implementations,
    SOFTS/TimesNet are loaded dynamically from models/*_z.py, PatchTST is wrapped by an adapter around
    models/PatchTST_z.py, GPT4MTS_z/VoT_z are loaded dynamically from models/*_z.py (both
    USES_TEXT=True; text_dim must match the output dimension of OnlineMM's text encoder),
    following finetune() in adapt_z.py)."""
    if backbone == "iTransformer":
        return ITransformerZ(sl, pl, enc_in, d_model=d_model, e_layers=e_layers)
    if backbone == "SOFTS":
        from models.SOFTS_z import Model as _SOFTSModel
        return _SOFTSModel(argparse.Namespace(pred_len=pl), sl)
    if backbone == "TimesNet":
        from models.TimesNet_z import Model as _TimesNetModel
        cfg = argparse.Namespace(seq_len=sl, label_len=0, pred_len=pl,
                                 enc_in=enc_in, c_out=enc_in,
                                 d_model=d_model, e_layers=e_layers, freq="h")
        return _TimesNetModel(cfg, pl)
    if backbone == "PatchTST":
        return _PatchTSTZAdapter(sl, pl, enc_in, d_model=d_model, e_layers=e_layers)
    if backbone == "Transformer":
        return TransformerZ(sl, pl, enc_in, d_model=d_model, e_layers=e_layers)
    if backbone == "GPT4MTS_z":
        from models.GPT4MTS_z import Model as _GPT4MTSModel
        return _GPT4MTSModel(sl, pl, enc_in, text_dim=text_dim, e_layers=e_layers)
    if backbone == "VoT_z":
        from models.VoT_z import Model as _VoTModel
        return _VoTModel(sl, pl, enc_in, text_dim=text_dim, e_layers=e_layers)
    raise ValueError(f"unknown backbone: {backbone!r} (options: iTransformer/SOFTS/"
                     f"TimesNet/PatchTST/Transformer/GPT4MTS_z/VoT_z)")


def _backbone_dims(backbone, sl, pl, enc_in, d_model, e_layers, z_loc):
    """Return (eff_d_model, eff_z_loc, z_n_dim) — backbones differ in z insertion dimension / layer count."""
    if backbone == "SOFTS":
        return 256, min(max(z_loc, 1), 3), enc_in
    if backbone in ("TimesNet", "Transformer"):
        return d_model, min(max(z_loc, 1), e_layers), sl + pl
    if backbone == "PatchTST":
        # PatchTST_z injects at z_loc='input2' into the raw decomposed [B, enc_in, seq_len] tensor
        # (rather than into an encoder-layer tensor of shape (batch, channel, d_model) as the other backbones do),
        # so here the last dimension of z_shape is seq_len rather than d_model.
        return sl, 1, enc_in
    if backbone == "GPT4MTS_z":
        # [O8] per-layer injection is now supported (see models/GPT4MTS_z.py::_gpt2_layerwise):
        # z_loc=0 = the legacy single fixed pre-GPT2 injection (control group); z_loc=1..gpt_layers
        # = injection after the output of the k-th block inside the frozen GPT2; larger is deeper. eff_d_model
        # is fixed at the GPT2 hidden size 768 (independent of e_layers; e_layers only sets
        # the upper bound of gpt_layers).
        from models.GPT4MTS_z import infer_patch_num
        gpt_layers = min(max(int(e_layers), 1), 12)
        return 768, min(max(z_loc, 0), gpt_layers), infer_patch_num(sl, pl)
    if backbone == "VoT_z":
        # Supports 4 candidate injection points (module docstring of models/VoT_z.py:
        # POS_TREND=1/POS_SEASONAL=2/POS_PRE_GPT2=3/POS_POST_GPT2=4);
        # eff_z_loc defaults to 3 (pre_gpt2) to match the legacy single fixed injection; in ablations
        # each mode can override it with another candidate point via kw["z_loc"]. eff_d_model is likewise fixed
        # at the GPT2 hidden size 768.
        from models.GPT4MTS_z import infer_patch_num
        return 768, 3, infer_patch_num(sl, pl)
    return d_model, min(max(z_loc, 1), e_layers), enc_in


def pretrain_backbone(bb, train_loader, val_loader, device,
                      epochs=50, lr=1e-3, patience=3, ckpt_path=None,
                      z_shape=None, z_loc=None):
    """Backbone pretraining when the checkpoint is missing (early stopping + saving); the backbone stays frozen in the online phase. [O8]

    If z_shape/z_loc are provided, an all-zero z is always passed in the training/validation forward (required for SOFTS_z
    — its forecast() crashes on an unchecked len(z.shape) when z=None;
    for ITransformerZ/TimesNet_z it is an equivalent no-op).
    """
    opt = optim.Adam(bb.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best, bad, best_state = float("inf"), 0, None

    def _fwd(xb):
        if z_shape is not None:
            z0 = torch.zeros((xb.shape[0],) + tuple(z_shape[1:]), device=xb.device)
            return bb(xb, z=z0, z_loc=z_loc)["pred"]
        return bb(xb)["pred"]

    for ep in range(epochs):
        bb.train()
        tr = 0.0
        for xb, yb, *_ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(_fwd(xb), yb)
            loss.backward()
            opt.step()
            tr += loss.item()
        bb.eval()
        vl, m = 0.0, 0
        with torch.no_grad():
            for xb, yb, *_ in val_loader:
                vl += lossf(_fwd(xb.to(device)), yb.to(device)).item()
                m += 1
        vl /= max(m, 1)
        print(f"[pretrain] epoch {ep + 1}/{epochs}  "
              f"train={tr / max(len(train_loader), 1):.4f}  val={vl:.4f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in bb.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("[pretrain] early stop")
                break
    if best_state is not None:
        bb.load_state_dict(best_state)
    if ckpt_path:
        # Atomic write: write a temporary file first and then rename, so that a killed process or full disk does not leave
        # a checkpoint that "exists but is corrupted", which the next run would mistake for already trained
        # and try to load directly (torch.load raises "failed finding central
        # directory").
        tmp_path = f"{ckpt_path}.tmp{os.getpid()}"
        torch.save(bb.state_dict(), tmp_path)
        os.replace(tmp_path, ckpt_path)
        print(f"[pretrain] saved -> {ckpt_path}")
    return bb


# ══════════════════════════════════════════════════════════════
# SECTION 9 — VISUALISATION (same as the original; [O7] fixes a typo in the legend dict)
# ══════════════════════════════════════════════════════════════

_PALETTE = ["#9a9a9a", "#e0a840", "#e07040", "#4080c0", "#2e9e55"]

_MODE_COLOR = {
    "Unimodal (ADAPT-Z)":      "#9a9a9a",
    "+Text":                   "#e0a840",
    "+Text+Grad":              "#e07040",
    "MoE Fusion":              "#4080c0",
    "iAdapter":                "#2e9e55",
    "iAdapter+PosGate ": "#1f7a3f",

    "VoT_z (pos=pre_gpt2, default)": "#9a9a9a",
    "VoT_z (pos=trend)":          "#e0a840",
    "VoT_z (pos=seas.)":       "#e07040",
    "VoT_z (pos=post_gpt2)":      "#4080c0",
    "VoT_z+iAdapter ":   "#2e9e55",

    "GPT4MTS_z (pos=pre_gpt2, legacy)":       "#9a9a9a",
    "GPT4MTS_z (pos=layer1, shallow)":          "#e0a840",
    "GPT4MTS_z (pos=layer_mid, middle)":       "#e07040",
    "GPT4MTS_z (pos=layer_last, last-layer output)": "#4080c0",
    "GPT4MTS_z+iAdapter": "#2e9e55",
}

_MODE_NICE = {
    "ADAPT-Z":      "ADAPT-Z",
    "+Text":                   "+Text  (semantic prior)",
    "+Text+Grad":              "+Text+Grad  (dual-gradient feedback)",
    "MoE Fusion":              "MoE Fusion (adaptive fusion)",
    "iAdapter":                "iAdapter  (ours)",
    "iAdapter+PosGate": "iAdapter+PosGate",

    "VoT_z (pos=pre_gpt2, default)": "Fixed @ pre-GPT2",
    "VoT_z (pos=trend)":          "Fixed @ Trend",
    "VoT_z (pos=seas.)":       "Fixed @ Seas.",
    "VoT_z (pos=post_gpt2)":      "Fixed @ post-GPT2",
    "VoT_z+iAdapter":   "iAdapter ",

    "GPT4MTS_z (pos=pre_gpt2, legacy)":       "Fixed @ pre-GPT2 (legacy)",
    "GPT4MTS_z (pos=layer1, shallow)":          "Fixed @ GPT2 layer 1 (shallow)",
    "GPT4MTS_z (pos=layer_mid, middle)":       "Fixed @ GPT2 mid-layer",
    "GPT4MTS_z (pos=layer_last, last-layer output)": "Fixed @ deepest GPT2 layer",
    "GPT4MTS_z+iAdapter ": "+ iAdapter @ deepest layer ",
}


def _mode_color(mode, i=0):
    return _MODE_COLOR.get(mode.strip(), _PALETTE[i % len(_PALETTE)])


def _mode_nice(mode):
    return _MODE_NICE.get(mode.strip(), mode)


def _is_ours(mode):
    # "iAdapter" covers iAdapter/iAdapter+PosGate in ABLATION_MODES and
    # GPT4MTS_z+iAdapter; "PosGate" additionally covers the adaptive-position mode in VOT_Z_POSITION_MODES
    # ("VoT_z+PosGate (adaptive position)") — both must be shown bold/highlighted
    # in every figure. [O9] strip() before the lookup, so that labels with trailing spaces in ABLATION_MODES,
    # such as "iAdapter ", are not missed.
    mode = mode.strip()
    return "iAdapter" in mode or "PosGate" in mode


def _rolling(a, w):
    out = np.full(len(a), np.nan)
    for i in range(w - 1, len(a)):
        out[i] = a[i - w + 1:i + 1].mean()
    return out


def _rolling_dist(G, w):
    """Per-expert rolling mean distribution, (T, E) -> (T, E); _rolling is called independently on each column."""
    out = np.full_like(G, np.nan, dtype=float)
    for e in range(G.shape[1]):
        out[:, e] = _rolling(G[:, e], w)
    return out


def _pick_drift_snapshots(G, n_snapshots=4, W=30, min_gap_frac=0.08):
    """Pick the n_snapshots time points where the expert-gate distribution drifts most.

    The per-step distribution is smoothed with a rolling window (W steps), and the drift magnitude is measured by the
    L1 distance between two smoothed distributions W steps apart; the points with the highest drift scores are taken greedily (forced to be at least
    min_gap_frac * T apart, so adjacent frames of the same drift are not selected twice), and finally sorted in time
    order for display. When the sequence is too short for the rolling window, it falls back to equally spaced sampling.
    """
    T = len(G)
    W = max(3, min(W, T // 4)) if T >= 12 else max(1, T // 2)
    smooth = _rolling_dist(G, W)
    valid_start = 2 * W - 1

    if valid_start >= T:
        picked = sorted(set(np.linspace(0, T - 1, min(n_snapshots, T)).astype(int)))
        return picked, np.nan_to_num(smooth, nan=0.0)

    drift = np.zeros(T)
    drift[valid_start:] = np.abs(
        smooth[valid_start:] - smooth[valid_start - W:T - W]
    ).sum(axis=1)

    min_gap = max(1, int(T * min_gap_frac))
    order = np.argsort(-drift[valid_start:]) + valid_start
    picked = []
    for idx in order:
        if all(abs(idx - p) >= min_gap for p in picked):
            picked.append(int(idx))
        if len(picked) == n_snapshots:
            break
    picked.sort()
    return picked, smooth


def _style_legend(ax, loc="upper center", ncol=3, bbox_to_anchor=None,
                  fontsize=12, handles=None, labels=None):
    """Unify the legend look (matching xihe.png: framed, semi-transparent background, thin border), so that "which line/which
    bar is which method" is clear at a glance instead of guessed from colour. Only the look is unified; the label content and colour semantics
    already designed for each figure are not changed. `ax` can be an Axes or a Figure (for a figure-wide
    legend across several subplots, e.g. plot_ablation_summary)."""
    kw = dict(loc=loc, ncol=ncol, bbox_to_anchor=bbox_to_anchor,
              fontsize=fontsize, frameon=True, framealpha=0.92,
              edgecolor="#cfcfcf", fancybox=False)
    if handles is not None:
        leg = ax.legend(handles, labels, **kw)
    else:
        leg = ax.legend(**kw)
    leg.get_frame().set_linewidth(0.8)
    return leg


def _adaptive_ncol(labels, fig_width_in, fontsize=14, char_width_in=0.011):
    """Lay the legend out on one row by default; fall back to two rows only when the estimated text width really does not fit, instead of
    the earlier empirically fixed ncol, which split it into several rows even when one row would fit."""
    if len(labels) <= 1:
        return 1
    avg_char_in = char_width_in * fontsize
    est_width = sum((len(l) + 4) * avg_char_in for l in labels)
    if est_width <= fig_width_in * 0.92:
        return len(labels)
    return math.ceil(len(labels) / 2)


def plot_ablation_summary(results, save_path="iAdapter_ablation_summary.png"):
    """MAE / MSE ablation radar chart (no title), annotated with the percentage gain relative to Unimodal."""
    names = [r["mode"] for r in results]
    maes = [r["mae"] for r in results]
    mses = [r["mse"] for r in results]
    uni_mae = next((r["mae"] for r in results if "Unimodal" in r["mode"]),
                   maes[0])
    uni_mse = next((r["mse"] for r in results if "Unimodal" in r["mode"]),
                   mses[0])

    n = len(names)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles_closed = angles + angles[:1]
    colors = [_mode_color(m, i) for i, m in enumerate(names)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6), dpi=130,
                             subplot_kw={"projection": "polar"})
    legend_handles = None
    for ax, vals, base, metric in ((axes[0], maes, uni_mae, "MAE"),
                                   (axes[1], mses, uni_mse, "MSE")):
        vals_closed = vals + vals[:1]

        ax.plot(angles_closed, vals_closed, color="#555555", lw=1.3,
               alpha=0.85, zorder=2)
        ax.fill(angles_closed, vals_closed, color="#8a8a8a", alpha=0.10,
               zorder=1)

        markers = []
        for angle, v, m, c in zip(angles, vals, names, colors):
            ours = _is_ours(m)
            h = ax.scatter([angle], [v],
                           s=95 if ours else 55,
                           color=c,
                           edgecolor="#1a1a1a" if ours else "white",
                           linewidth=1.6 if ours else 0.8,
                           zorder=6 if ours else 4)
            markers.append(h)

            gain = (base - v) / base * 100
            tag = f"{v:.4f}" + ("" if abs(gain) < 1e-9 else f"\n({gain:+.1f}%)")
            ax.annotate(tag, xy=(angle, v), xytext=(0, 8),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=13,
                       fontweight="bold" if ours else "normal", zorder=10)

        legend_handles = legend_handles or markers

        ax.set_xticks(angles)
        ax.set_xticklabels([_mode_nice(m).replace("  ", "\n") for m in names],
                           fontsize=14)
        # Leave enough radial margin so value annotations/dots do not touch the mode-name labels on the outer ring; the margin cannot
        # simply keep growing — the radius in pixels is fixed, so the larger ylim is, the fewer pixels the data range 0~max(vals)
        # gets, and the equally spaced 0.25/0.5/0.75 grid-line labels get squeezed closer together
        # and overlap. A moderate margin (1.22) + fewer ticks/smaller font solves this.
        # Leave enough radial margin: 1.50× gives two-line annotation text enough room (1.22 was too small; the tips ran
        # into the legend/outer values); nbins=3 keeps the radial ticks from crowding under the larger ylim.
        ax.set_ylim(0, max(vals) * 1.50)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="y", labelsize=7, pad=2)
        # Radial tick labels sit on the theta=0 spoke by default and easily collide with that category's data points/
        # annotations; move them to the gap angle between two adjacent spokes.
        ax.set_rlabel_position(180.0 / n if n else 0.0)
        ax.tick_params(axis="x", pad=14)
        ax.set_title(f"{metric}   (%: gain vs Unimodal ADAPT-Z)",
                    fontweight="bold", pad=22, fontsize=14)
        ax.grid(True, lw=0.4, alpha=0.35)
        ax.spines["polar"].set_linewidth(0.8)
        ax.spines["polar"].set_color("#555555")

    # Figure-wide legend across both subplots (originally there was no legend at all, and the x-axis text was the only way to tell which point was
    # which mode) — the scatter points already drawn are reused directly as legend handles, so colour/label match
    # the data points exactly, with no risk of a second hand-written mapping drifting.
    labels = [_mode_nice(m) for m in names]
    ncol = _adaptive_ncol(labels, fig.get_size_inches()[0], fontsize=14)
    _style_legend(fig, loc="upper center", ncol=ncol,
                 bbox_to_anchor=(0.5, 1.06), fontsize=14,
                 handles=list(legend_handles), labels=labels)
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"[viz] {save_path}")


def plot_rolling_mae(results, save_path="iAdapter_rolling_mae.png", W=30):
    """Top: rolling MAE; bottom: Δ|Error| = iAdapter − Unimodal net-gain band. No title."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=130, sharex=True,
                                  gridspec_kw={"height_ratios": [2.0, 1.0]})
    step_mae = {}
    for ci, r in enumerate(results):
        sm = np.mean(np.abs(r["preds"] - r["truths"]), axis=(1, 2))
        step_mae[r["mode"]] = sm
        ours = _is_ours(r["mode"])
        ax.plot(_rolling(sm, W), lw=2.4 if ours else 1.2,
                alpha=1.0 if ours else 0.75, zorder=6 if ours else 3,
                color=_mode_color(r["mode"], ci), label=_mode_nice(r["mode"]))
    ax.set_ylabel(f"Rolling MAE (window={W})")
    _style_legend(ax, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.18),
                 fontsize=8.5)
    ax.grid(True, lw=0.4, alpha=0.4)

    ours_key = next((m for m in step_mae if _is_ours(m)), None)
    uni_key = next((m for m in step_mae if "Unimodal" in m), None)
    if ours_key and uni_key:
        raw = step_mae[ours_key] - step_mae[uni_key]
        diff = _rolling(raw, W)
        steps = np.arange(len(diff))
        ax2.fill_between(steps, np.where(diff < 0, diff, 0), color="#2e9e55",
                         alpha=0.55, label="iAdapter better")
        ax2.fill_between(steps, np.where(diff >= 0, diff, 0), color="#e05050",
                         alpha=0.45, label="iAdapter worse")
        ax2.axhline(0, color="black", lw=0.8, ls="--")
        ax2.plot(diff, color="#666666", lw=0.7, alpha=0.6)
        ax2.text(0.99, 0.05, f"mean Δ|err| = {np.nanmean(raw):+.4f}",
                 transform=ax2.transAxes, ha="right", va="bottom", fontsize=9)
        ax2.set_ylabel("Δ|Error| (ours − Unimodal)")
        _style_legend(ax2, loc="upper center", ncol=2,
                     bbox_to_anchor=(0.5, -0.18), fontsize=8)
    ax2.set_xlabel("Online step")
    ax2.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {save_path}")


def plot_pred_curves(results, sensor_idx=0, n_show=300,
                     save_path="iAdapter_pred_curves.png"):
    """Ground truth vs each mode's forecast (first forecast step); top = beginning segment / bottom = end segment. No title."""
    truth = results[0]["truths"]
    n = min(n_show, len(truth))
    fig, axes = plt.subplots(2, 1, figsize=(16, 8.5), dpi=130)
    for ax, sl in ((axes[0], slice(0, n)),
                   (axes[1], slice(max(0, len(truth) - n), None))):
        t = np.arange(len(truth))[sl]
        ax.plot(t, truth[sl, 0, sensor_idx], color="#222222", lw=1.8,
                label="Ground Truth", zorder=7)
        for ci, r in enumerate(results):
            ours = _is_ours(r["mode"])
            ax.plot(t, r["preds"][sl, 0, sensor_idx],
                    color=_mode_color(r["mode"], ci),
                    lw=1.8 if ours else 0.9, alpha=0.95 if ours else 0.65,
                    zorder=6 if ours else 3, label=_mode_nice(r["mode"]))
        ax.set_ylabel(f"Sensor #{sensor_idx} (normalised)")
        ax.grid(True, lw=0.35, alpha=0.35)
    _style_legend(axes[0], loc="upper right", ncol=3, fontsize=7.5)
    axes[1].set_xlabel("Online step")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {save_path}")


def plot_gate_usage(gate_history, expert_names=None,
                    save_path="iAdapter_gate_usage.png",
                    n_snapshots=4, snapshot_window=30):
    """Evolution of the MoE expert gate (no title). Below it, distribution snapshots (D1..Dn) at the n_snapshots moments of
    largest drift are attached, so "how the expert-weight distribution drifts over time" can be compared at a glance, without
    having to read it off the stacked plot."""
    if not gate_history:
        return
    G = np.array(gate_history)                          # (T, E)
    expert_names = expert_names or ["Identity", "Trend", "Seasonality",
                                    "Fluctuation", "Text-FiLM"]
    colors = ["#bfbfbf", "#e0a840", "#40b060", "#4080c0", "#d9985f"]

    picked, smooth = _pick_drift_snapshots(G, n_snapshots=n_snapshots,
                                           W=snapshot_window)
    n_cols = max(len(picked), 1)

    fig = plt.figure(figsize=(14, 6.6), dpi=130)
    gs = fig.add_gridspec(2, n_cols, height_ratios=[2.4, 1.0],
                          hspace=0.6, wspace=0.35)

    ax = fig.add_subplot(gs[0, :])
    ax.stackplot(np.arange(len(G)), G.T, labels=expert_names,
                 colors=colors, alpha=0.85)
    ax.set_xlabel("Online step")
    ax.set_ylabel("Mean gate weight")
    ax.set_ylim(0, 1)
    # Originally loc="upper center" sat at the top inside the plot and covered the top layer of the stacked area — moved
    # below the plot, no longer covering anything.
    _style_legend(ax, loc="upper center", ncol=len(expert_names),
                 bbox_to_anchor=(0.5, -0.16), fontsize=8.5)
    ax.grid(True, lw=0.35, alpha=0.3)

    for rank, t in enumerate(picked, start=1):
        ax.axvline(t, color="#222222", lw=1.0, ls="--", alpha=0.7, zorder=9)
        ax.text(t, 1.02, f"D{rank}", ha="center", va="bottom", fontsize=14,
               fontweight="bold", transform=ax.get_xaxis_transform())

    # Expert-weight distribution at each selected moment (rolling mean, to avoid single-frame noise), compared side by side
    for col, t in enumerate(picked):
        sax = fig.add_subplot(gs[1, col])
        dist = smooth[t]
        bottom = 0.0
        for w, c in zip(dist, colors):
            sax.bar(0, w, bottom=bottom, color=c, width=0.6,
                   edgecolor="white", linewidth=0.6)
            if w > 0.06:
                sax.text(0, bottom + w / 2, f"{w:.2f}", ha="center",
                        va="center", fontsize=7)
            bottom += w
        sax.set_xlim(-0.5, 0.5)
        sax.set_ylim(0, 1)
        sax.set_xticks([])
        sax.set_title(f"D{col + 1}  (step={t})", fontsize=14)
        if col == 0:
            sax.set_ylabel("Gate weight", fontsize=14)
        else:
            sax.set_yticks([])
        for spine in sax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#888888")

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"[viz] {save_path}")


def plot_multi_pred_len_summary(
        all_results: dict,
        save_path="iAdapter_multi_predlen.png",
        baseline_mode=None):
    """
    Multi-horizon summary figure:
    x-axis = pred_len
    y-axis = MAE / MSE

    dark red: iAdapter
    dark blue: main baseline
    low-saturation colours: other ablation methods
    """

    pred_lens = sorted(all_results.keys())
    modes = [r["mode"] for r in next(iter(all_results.values()))]

    # By default the first non-ours mode is the main baseline;
    # it can also be set explicitly via baseline_mode.
    if baseline_mode is None:
        baseline_mode = next(
            (mode for mode in modes if not _is_ours(mode)),
            modes[0],
        )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.8),
        dpi=160,
    )

    for index, mode in enumerate(modes):
        maes = [
            next(
                r["mae"]
                for r in all_results[pred_len]
                if r["mode"] == mode
            )
            for pred_len in pred_lens
        ]

        mses = [
            next(
                r["mse"]
                for r in all_results[pred_len]
                if r["mode"] == mode
            )
            for pred_len in pred_lens
        ]

        style = _get_line_style(
            mode=mode,
            index=index,
            baseline_mode=baseline_mode,
        )

        label = _mode_nice(mode)

        ax1.plot(
            pred_lens,
            maes,
            label=label,
            **style,
        )

        ax2.plot(
            pred_lens,
            mses,
            label=label,
            **style,
        )

    for ax, metric in ((ax1, "MAE"), (ax2, "MSE")):
        ax.set_xlabel("Prediction horizon (pred_len)")
        ax.set_ylabel(metric, fontweight="bold")
        ax.set_xticks(pred_lens)
        _style_axis(ax)

    # Take the legend from the first subplot only, to avoid duplicates
    handles, labels = ax1.get_legend_handles_labels()

    # Lay the legend out on one row by default; fall back to two rows only when the estimated width really does not fit.
    ncol = _adaptive_ncol(labels, fig.get_size_inches()[0],
                          fontsize=LEGEND_FONT.get_size())

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=ncol,
        frameon=False,
        prop=LEGEND_FONT,
        handlelength=2.6,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    # Leave room for the shared legend at the bottom
    fig.tight_layout(rect=(0, 0.17, 1, 1))

    _save_figure(fig, save_path)
    plt.close(fig)

    print(f"[viz] {save_path}")


_GPT4MTS_Z_ADV_BASE = "GPT4MTS_z "
_GPT4MTS_Z_ADV_OURS = "+iAdapter "


def plot_gpt4mts_z_advantage(
        all_results: dict,
        save_path="iAdapter_gpt4mts_advantage.png"):

    pred_lens = sorted(all_results.keys())

    have = [
        pred_len
        for pred_len in pred_lens
        if any(
            r["mode"].strip() == _GPT4MTS_Z_ADV_BASE
            for r in all_results[pred_len]
        )
        and any(
            r["mode"].strip() == _GPT4MTS_Z_ADV_OURS
            for r in all_results[pred_len]
        )
    ]

    if not have:
        print(
            "[viz] plot_gpt4mts_z_advantage: "
            "Results for the two modes baseline/iAdapter not found, skipping"
        )
        return

    base_vals = {"mae": [], "mse": []}
    ours_vals = {"mae": [], "mse": []}

    for pred_len in have:
        base = next(
            r for r in all_results[pred_len]
            if r["mode"].strip() == _GPT4MTS_Z_ADV_BASE
        )

        ours = next(
            r for r in all_results[pred_len]
            if r["mode"].strip() == _GPT4MTS_Z_ADV_OURS
        )

        for metric in ("mae", "mse"):
            base_vals[metric].append(base[metric])
            ours_vals[metric].append(ours[metric])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        dpi=160,
    )

    x = np.arange(len(have))
    width = 0.34

    legend_handles = None

    for ax, metric in zip(axes, ("mae", "mse")):
        baseline_values = base_vals[metric]
        ours_values = ours_vals[metric]

        baseline_bars = ax.bar(
            x - width / 2,
            baseline_values,
            width,
            color=DARK_BLUE,
            edgecolor="#17386E",
            linewidth=1.0,
            zorder=3,
            label=_mode_nice(_GPT4MTS_Z_ADV_BASE),
        )

        ours_bars = ax.bar(
            x + width / 2,
            ours_values,
            width,
            color=DARK_RED,
            edgecolor="#541022",
            linewidth=1.2,
            zorder=4,
            label=_mode_nice(_GPT4MTS_Z_ADV_OURS),
        )

        if legend_handles is None:
            legend_handles = [
                baseline_bars[0],
                ours_bars[0],
            ]

        for index, (baseline_value, ours_value) in enumerate(
                zip(baseline_values, ours_values)):

            gain = (
                (baseline_value - ours_value)
                / baseline_value
                * 100
            )

            ax.annotate(
                f"{gain:+.1f}%",
                xy=(x[index] + width / 2, ours_value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color=DARK_RED,
                fontfamily=CN_FONT,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(pred_len) for pred_len in have])

        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel(metric.upper(), fontweight="bold")

        # Keep the gain text from touching the figure edge
        highest_value = max(
            max(baseline_values),
            max(ours_values),
        )
        lowest_value = min(
            min(baseline_values),
            min(ours_values),
        )
        value_range = max(highest_value - lowest_value, highest_value * 0.02)

        ax.set_ylim(
            lowest_value - value_range * 0.12,
            highest_value + value_range * 0.28,
        )

        _style_axis(ax)

    fig.legend(
        legend_handles,
        [
            _mode_nice(_GPT4MTS_Z_ADV_BASE),
            _mode_nice(_GPT4MTS_Z_ADV_OURS),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        prop=LEGEND_FONT,
        handlelength=2.0,
        columnspacing=1.8,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.91))

    _save_figure(fig, save_path)
    plt.close(fig)

    print(f"[viz] {save_path}")

def _reso_ours(results):
    return next((r for r in results if _is_ours(r["mode"])), None)


def _reso_base(results):
    return next((r for r in results if "Unimodal" in r["mode"]), None)


def _mm01(a):
    """min-max normalization to [0,1] (nan-safe)."""
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo + 1e-12)


def _period_1d(g, min_p=4, frac=0.5):
    """1-D dominant period (detrend linearly + rFFT peak); returns None if there is no significant period."""
    g = np.asarray(g, dtype=float)
    n = len(g)
    if n < 2 * min_p:
        return None
    t = np.arange(n)
    s = np.polyfit(t, g, 1)
    gd = g - (s[0] * t + s[1])
    pw = np.abs(np.fft.rfft(gd)) ** 2
    fr = np.fft.rfftfreq(n)
    pw[0] = 0.0
    best, bp = -1.0, None
    for k in range(1, len(fr)):
        if fr[k] <= 0:
            continue
        p = 1.0 / fr[k]
        if min_p <= p <= n * frac and pw[k] > best:
            best, bp = pw[k], p
    return None if bp is None else int(round(bp))


# def _lag_corr(e, s, lag):
#     """corr(e_t, s_{t+lag}); lag>0 means the energy leads the signal by lag steps."""
#     e = np.asarray(e, dtype=float)
#     s = np.asarray(s, dtype=float)
#     if lag >= 0:
#         x, y = e[: len(e) - lag] if lag else e, s[lag:]
#     else:
#         x, y = e[-lag:], s[: len(s) + lag]
#     m = np.isfinite(x) & np.isfinite(y)
#     if m.sum() < 8:
#         return np.nan
#     xs, ys = x[m], y[m]
#     if xs.std() < 1e-12 or ys.std() < 1e-12:
#         return np.nan
#     return float(np.corrcoef(xs, ys)[0, 1])
def _lag_corr(x, y, lag):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = min(x.size, y.size)
    if n - abs(lag) < 3:
        return np.nan
    x, y = x[:n], y[:n]              # force equal lengths first
    if lag > 0:
        x, y = x[:n - lag], y[lag:]
    elif lag < 0:
        x, y = x[-lag:], y[:n + lag]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])

class ResonanceLogger:
    """
    Attached to OnlineMM (mm.reslog = ResonanceLogger()); the online() main loop
    calls on_step once per batch (the hook is built into OnlineMM.online [C3]).

    Records per sample: rows / conf / emb_norm / gate_text / eff_experts /
    tg_norm / real_change / pred_change; with counterfactual=True it additionally runs a
    "text zeroed" counterfactual forward and records the per-sample mean of |Δpred| and its per-horizon profile.

    Extra cost: 2 no_grad forwards per batch (feature + counterfactual pred);
    negligible effect on total time when attached only to M5.
    """

    def __init__(self, counterfactual=True):
        self.cf = counterfactual
        self.log = dict(rows=[], conf=[], emb_norm=[], gate_text=[],
                        eff=[], tg_norm=[], dpred_abs=[], dpred_h=[],
                        pred_change=[], real_change=[])

    @torch.no_grad()
    def on_step(self, mm, x, y, pred, text_emb_in, text_conf,
                his_grad, his_text_grad, z0, offset, rows):
        B, H, N = pred.shape
        L, ad = self.log, mm.adapter

        L["rows"] += [int(r) for r in rows]
        L["conf"] += (text_conf.squeeze(-1).float().cpu().numpy().tolist()
                      if text_conf is not None else [1.0] * B)
        L["emb_norm"] += (text_emb_in.norm(dim=-1).float().cpu().numpy()
                          .tolist() if text_emb_in is not None else [0.0] * B)
        lg = getattr(ad, "last_gate", None)
        L["gate_text"] += (lg[..., -1].mean(1).float().cpu().numpy().tolist()
                           if (lg is not None and lg.dim() == 3)
                           else [float("nan")] * B)
        L["eff"] += [float(getattr(ad, "eff_experts", float("nan")))] * B
        L["tg_norm"] += [float(his_text_grad.norm())
                         if his_text_grad is not None else 0.0] * B

        last = x[:, -1, :]                          # globally standardized space
        L["real_change"] += (y.mean(1) - last).mean(-1).cpu().numpy().tolist()
        L["pred_change"] += (pred.mean(1) - last).mean(-1).cpu().numpy().tolist()

        # ── Counterfactual: text zeroed (adapter text path fully off) → net text control ──
        if self.cf and mm.use_text:
            feat = mm.backbone(x, z=z0, z_loc=mm.args.z_loc)["feature"]
            zeros = torch.zeros(B, mm.text_dim, device=x.device)
            z_cf = mm.adapter(feat, his_grad, zeros, his_text_grad, offset)
            pred_cf = mm.backbone(x, z=z_cf, z_loc=mm.args.z_loc)["pred"]
            d = pred - pred_cf
            L["dpred_abs"] += d.abs().mean(dim=(1, 2)).cpu().numpy().tolist()
            L["dpred_h"] += d.abs().mean(dim=2).cpu().numpy().tolist()  # (B,H)
        else:
            L["dpred_abs"] += [float("nan")] * B
            L["dpred_h"] += [[float("nan")] * H] * B

    def arrays(self):
        out = {}
        for k, v in self.log.items():
            try:
                out[k] = np.asarray(v, dtype=float)
            except Exception:
                out[k] = np.asarray(v, dtype=object)
        return out


class EmbedAlignLogger:
    """Attached to OnlineMM (mm.embed_log = EmbedAlignLogger()); online() calls on_step once at each
    buffer update, recording the batch samples' TS vectors / text vectors in the VICRegAlign
    shared space, together with the dominant period and jump strength of the input window —
    for the semantic-space visualization in plot_clip_alignment_space; zero overhead when not attached."""

    def __init__(self):
        self.log = dict(ts_e=[], text_e=[], period=[], jump=[], rows=[],
                        text_var_history=[], ts_var_history=[])

    @torch.no_grad()
    def on_step(self, mm, feature, text_emb, x_b, rows):
        if mm.clip_align is None or text_emb is None:
            return
        ts_e, text_e = mm.clip_align.embed(feature, text_emb)
        self.log["ts_e"].append(ts_e.cpu().numpy())
        self.log["text_e"].append(text_e.cpu().numpy())
        self.log["period"].append(_dominant_period(x_b).cpu().numpy())
        self.log["jump"].append(_jump_score(x_b).cpu().numpy())
        self.log["rows"].extend(int(r) for r in rows)

        # [NEW] record in-batch text/TS variance for collapse monitoring (the variance subplot of plot_clip_alignment_space)
        # — the lower var is, the more that modality's pooled vectors are collapsing to a point.
        text_var = float(text_emb.var(dim=0).mean().item())
        ts_var = float(feature.mean(dim=1).var(dim=0).mean().item())
        self.log["text_var_history"].append(text_var)
        self.log["ts_var_history"].append(ts_var)

    def arrays(self):
        if not self.log["rows"]:
            return None
        return {
            "ts_e": np.concatenate(self.log["ts_e"], axis=0),
            "text_e": np.concatenate(self.log["text_e"], axis=0),
            "period": np.concatenate(self.log["period"], axis=0),
            "jump": np.concatenate(self.log["jump"], axis=0),
            "rows": np.asarray(self.log["rows"]),
            "text_var_history": np.asarray(self.log["text_var_history"]),
            "ts_var_history": np.asarray(self.log["ts_var_history"]),
        }


def plot_text_resonance_timeline(res, results, sensor_idx=0, W=15,
                                 save_path="iAdapter_resonance_timeline.png"):
    """Three-band resonance timeline:
    P1 ground truth + M5/M1 forecasts, red vertical lines = top-1% squared-error steps;
    P2 four text-energy curves (min-max normalized): conf / ‖e_text‖ / gate_text / ‖∇e‖;
    P3 counterfactual |Δpred| + conf band, co-located with the red lines; the top-right corner gives the co-location ratio
       (mean |Δpred| on top-1% error steps ÷ on the other steps) — a ratio clearly >1 means
       text control is concentrated on tail-error steps, direct evidence of the MSE degradation."""
    ours, base = _reso_ours(results), _reso_base(results)
    if ours is None:
        print("[reso] no iAdapter mode found in results, skipping")
        return
    gt = ours["truths"][:, 0, sensor_idx]
    n = min(len(gt), len(res["conf"]))
    t = np.arange(n)

    e2 = ((ours["preds"] - ours["truths"]) ** 2).mean(axis=(1, 2))[:n]
    k1 = max(1, int(0.01 * n))
    top1 = np.argsort(e2)[-k1:]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 9.5), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.3, 1.0]})

    # ── P1: forecasts + tail markers ──
    ax1.plot(t, gt[:n], color="#111111", lw=1.6, label="Ground Truth",
             zorder=6)
    ax1.plot(t, ours["preds"][:n, 0, sensor_idx], color="#2e9e55", lw=1.7,
             label="iAdapter (ours)", zorder=5)
    if base is not None:
        ax1.plot(t, base["preds"][:n, 0, sensor_idx], color="#9a9a9a",
                 lw=1.0, alpha=0.8, label="Unimodal (ADAPT-Z)", zorder=4)
    for x0 in top1:
        ax1.axvline(x0, color="#d02020", lw=0.9, alpha=0.35, zorder=2)
    ax1.set_ylabel(f"Sensor #{sensor_idx} (scaled)")
    _style_legend(ax1, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16),
                 fontsize=8)
    ax1.grid(True, lw=0.35, alpha=0.35)
    ax1.text(0.995, 0.03, "red vlines = top-1% squared-error steps",
             transform=ax1.transAxes, ha="right", fontsize=7.5,
             color="#d02020")

    # ── P2: text energy (unified min-max for comparison on the same axis) ──
    curves = [("conf (EMA)", res["conf"][:n], "#4080c0", "-"),
              (r"$\|e_{text}\|$", res["emb_norm"][:n], "#2e9e55", "-"),
              ("Text-FiLM gate", res["gate_text"][:n], "#d9985f", "-"),
              (r"$\|\nabla e\|$", res["tg_norm"][:n], "#e0a840", ":")]
    for name, v, c, ls in curves:
        v = np.asarray(v, dtype=float)
        if not np.isfinite(v).any():
            continue
        ax2.plot(t, _rolling(_mm01(v), W), color=c, lw=1.6, ls=ls, label=name)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_ylabel("text energy (min-max)")
    _style_legend(ax2, loc="upper left", ncol=4, fontsize=7.5)
    ax2.grid(True, lw=0.35, alpha=0.35)

    # ── P3: counterfactual net control + conf band + tail co-location ──
    conf = np.asarray(res["conf"][:n], dtype=float)
    da = np.asarray(res["dpred_abs"][:n], dtype=float)
    dp = _rolling(np.nan_to_num(da), W)
    scale = np.nanmax(dp) if np.isfinite(dp).any() and np.nanmax(dp) > 0 else 1.0
    ax3.fill_between(t, 0, conf * 0.3 * scale, step="mid", color="#4080c0",
                     alpha=0.28, label="conf (EMA, scaled)")
    if np.isfinite(da).any():
        ax3.plot(t, dp, color="#2e9e55", lw=1.7,
                 label=r"$|\Delta pred|$ (counterfactual: text zeroed)")
    for x0 in top1:
        ax3.axvline(x0, color="#d02020", lw=0.9, alpha=0.35)
    ax3.set_ylabel("text influence")
    ax3.set_xlabel("Online step")
    _style_legend(ax3, loc="upper left", ncol=2, fontsize=7.5)
    ax3.grid(True, lw=0.35, alpha=0.35)

    if np.isfinite(da).any():
        m_top = np.nanmean(da[top1])
        rest = np.setdiff1d(np.arange(n), top1)
        m_rest = np.nanmean(da[rest]) if len(rest) else np.nan
        ax3.text(0.995, 0.90,
                 f"|Dpred| @top-1% err = {m_top:.4f}   @rest = {m_rest:.4f}"
                 f"   ratio = {m_top / (m_rest + 1e-12):.2f}",
                 transform=ax3.transAxes, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[reso] {save_path}")


def plot_tsimage_resonance(res, results, sensor_idx=0,
                           save_path="iAdapter_tsimage_resonance.png"):
    """TS-image-style periodic-folding two-band figure (inspired by TS2I in TimeOmni-VL):
    top band = test-segment ground truth folded by the dominant period f into a (phase × period) heat map, red vertical lines mark
    numerically anomalous periods (column mean |z|>1.5);
    bottom band = per-period text energy (conf / ‖e_text‖ / gate / |Δpred|, each row
    min-max). Vertical alignment of bright columns in the two bands is the "resonance"."""
    ours = _reso_ours(results)
    if ours is None:
        return
    g = ours["truths"][:, 0, sensor_idx]
    n = min(len(g), len(res["conf"]))
    g = g[:n]

    f = _period_1d(g) or max(4, n // 10)
    C = n // f
    if C < 3:
        f = max(4, n // 8)
        C = n // f
    if C < 2:
        print("[reso] test segment too short for periodic folding")
        return
    band = g[: f * C].reshape(C, f).T               # (f, C): row = phase within the period

    def _cyc(v):
        v = np.asarray(v[: f * C], dtype=float)
        return np.nanmean(v.reshape(C, f), axis=1)

    rows_txt = ["conf", r"$\|e_{text}\|$", "gate", r"$|\Delta pred|$"]
    txt = [_cyc(res["conf"][:n]),
           _cyc(res["emb_norm"][:n]),
           _cyc(np.nan_to_num(np.asarray(res["gate_text"][:n], dtype=float))),
           _cyc(np.nan_to_num(np.asarray(res["dpred_abs"][:n], dtype=float)))]
    T = np.stack([_mm01(v) for v in txt])

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13.5, 6.8), dpi=130)
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.6], hspace=0.16)

    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(band, aspect="auto", cmap="viridis",
                     interpolation="nearest")
    ax1.set_ylabel(f"phase within period (f={f})")
    ax1.set_xticks([])
    fig.colorbar(im1, ax=ax1, fraction=0.015, pad=0.01)
    cm = band.mean(axis=0)
    zsc = (cm - cm.mean()) / (cm.std() + 1e-12)
    for c in np.where(np.abs(zsc) > 1.5)[0]:
        ax1.axvline(c, color="#ff5050", lw=1.0, alpha=0.55)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    im2 = ax2.imshow(T, aspect="auto", cmap="magma", interpolation="nearest")
    ax2.set_yticks(range(len(rows_txt)))
    ax2.set_yticklabels(rows_txt, fontsize=8)
    ax2.set_xlabel(f"cycle index (test segment folded by period f={f}; "
                   f"red vlines above = numeric anomaly cycles)")
    fig.colorbar(im2, ax=ax2, fraction=0.015, pad=0.01)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[reso] {save_path}")


def plot_energy_crosscorr(res, results, sensor_idx=0, max_lag=20, W=5,
                          save_path="iAdapter_energy_crosscorr.png"):
    """Quantitative description of resonance — lagged cross-correlation corr(energy_t, signal_{t+lag}):
    left: text energy ↔ local volatility of the series |Δgt| — a peak near lag=0 means the text energy
        resonates with the series volatility; a peak at positive lag means the text **leads** the volatility (early-warning value);
    right: text energy ↔ error improvement (|err_M1| − |err_M5|) — a positive peak means moments of high text
        energy really bring accuracy gains; a negative one means the text actually hurts."""
    ours, base = _reso_ours(results), _reso_base(results)
    if ours is None:
        return
    n = min(ours["preds"].shape[0], len(res["conf"]))

    # Energy: prefer counterfactual |Δpred|; if missing, fall back to the mean of conf and ‖e_text‖
    da = np.asarray(res["dpred_abs"][:n], dtype=float)
    if np.isfinite(da).any():
        energy, ename = da, r"$|\Delta pred|$"
    else:
        energy = 0.5 * (_mm01(res["conf"][:n]) + _mm01(res["emb_norm"][:n]))
        ename = "conf & ||e_text|| composite"
    energy = _rolling(np.nan_to_num(energy), W)

    gt = ours["truths"][:n, 0, sensor_idx]
    vol = _rolling(np.abs(np.diff(gt, prepend=gt[0])), W)   # local volatility of the series
    energy = np.asarray(energy, float).ravel()
    vol = np.asarray(vol, float).ravel()
    if min(energy.size, vol.size) < 3:
        print(f"[reso] skip energy-crosscorr: len(energy)={energy.size}, len(vol)={vol.size}")
        return

    lags = np.arange(-max_lag, max_lag + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=130)

    cv = np.array([_lag_corr(energy, vol, int(l)) for l in lags])
    ax1.plot(lags, cv, color="#4080c0", lw=1.8, marker="o", ms=3)
    ax1.axvline(0, color="#444444", lw=0.8, ls="--")
    ax1.axhline(0, color="#444444", lw=0.8)

    if not np.any(np.isfinite(cv)):
        plt.close(fig)
        return
    if np.isfinite(cv).any():
        pk = int(lags[np.nanargmax(np.abs(cv))])
        ax1.text(0.03, 0.95, f"peak |corr| at lag={pk:+d} "
                             f"(r={cv[np.nanargmax(np.abs(cv))]:+.3f})",
                 transform=ax1.transAxes, fontsize=9, va="top")
    ax1.set_xlabel("lag (steps; +lag = energy leads signal)")
    ax1.set_ylabel(f"corr({ename}, |ΔGT| volatility)")
    ax1.grid(True, lw=0.35, alpha=0.35)

    if base is not None:
        imp = (np.abs(base["preds"] - base["truths"]).mean(axis=(1, 2))[:n]
               - np.abs(ours["preds"] - ours["truths"]).mean(axis=(1, 2))[:n])
        imp = _rolling(imp, W)
        ci = np.array([_lag_corr(energy, imp, int(l)) for l in lags])
        ax2.plot(lags, ci, color="#2e9e55", lw=1.8, marker="s", ms=3)
        ax2.axvline(0, color="#444444", lw=0.8, ls="--")
        ax2.axhline(0, color="#444444", lw=0.8)
        if np.isfinite(ci).any():
            pk = int(lags[np.nanargmax(np.abs(ci))])
            ax2.text(0.03, 0.95, f"peak |corr| at lag={pk:+d} "
                                 f"(r={ci[np.nanargmax(np.abs(ci))]:+.3f})",
                     transform=ax2.transAxes, fontsize=9, va="top")
        ax2.set_xlabel("lag (steps; +lag = energy leads improvement)")
        ax2.set_ylabel("corr(energy, |err M1| − |err M5|)")
        ax2.grid(True, lw=0.35, alpha=0.35)
    else:
        ax2.text(0.5, 0.5, "Unimodal baseline not in results", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[reso] {save_path}")


def plot_clip_alignment_space(embed_arrays, max_lines=300,
                              save_path="iAdapter_clip_align_space.png"):
    """Visualization of the CLIP shared semantic space: TS vectors and text vectors (collected by EmbedAlignLogger)
    are projected together to 2D with a single PCA (the same coordinate system; otherwise "alignment" cannot be seen), in two subplots:
      left: coloured by the dominant period of the input window (the period length of the largest frequency in the FFT amplitude spectrum);
      right: coloured by jump strength (extreme first difference / std within the window); high-jump samples get a red edge.
    Dots = TS vectors, triangles = text vectors; thin grey lines connect the (TS, text) pair of the same sample;
    the shorter the line, the tighter that sample is aligned."""
    ts_e, text_e = embed_arrays["ts_e"], embed_arrays["text_e"]
    period, jump = embed_arrays["period"], embed_arrays["jump"]
    M = ts_e.shape[0]
    if M < 2:
        print("[clip-align] not enough samples, skipping semantic-space visualization")
        return

    joint = np.concatenate([ts_e, text_e], axis=0)
    coords = PCA(n_components=2).fit_transform(joint)
    ts_xy, text_xy = coords[:M], coords[M:]

    rng = np.random.default_rng(0)
    line_idx = (rng.choice(M, size=max_lines, replace=False)
               if M > max_lines else np.arange(M))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=130)

    for ax, color_val, cmap, title in (
        (ax1, period, "viridis", "colored by dominant period (FFT)"),
        (ax2, jump, "magma", "colored by jump score (max |Δx| / std)"),
    ):
        for i in line_idx:
            ax.plot([ts_xy[i, 0], text_xy[i, 0]],
                   [ts_xy[i, 1], text_xy[i, 1]],
                   color="#999999", lw=0.4, alpha=0.35, zorder=1)
        sc1 = ax.scatter(ts_xy[:, 0], ts_xy[:, 1], c=color_val, cmap=cmap,
                        marker="o", s=22, label="TS embedding", zorder=5,
                        edgecolors="none")
        ax.scatter(text_xy[:, 0], text_xy[:, 1], c=color_val, cmap=cmap,
                  marker="^", s=26, label="text embedding", zorder=5,
                  edgecolors="black", linewidths=0.3)
        if color_val is jump:
            hi = jump >= np.quantile(jump, 0.9)
            ax.scatter(ts_xy[hi, 0], ts_xy[hi, 1], facecolors="none",
                      edgecolors="red", s=60, linewidths=1.1, zorder=5,
                      label="high-jump TS")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        _style_legend(ax, loc="best", ncol=1, fontsize=14)
        ax.grid(True, lw=0.3, alpha=0.3)
        plt.colorbar(sc1, ax=ax, shrink=0.85)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[clip-align] {save_path}  (M={M} samples)")

    # [NEW] third subplot: in-batch variance curves of text vs TS, quantitatively confirming whether the collapse is gone
    # (evolving over buffer-update steps; below the collapse threshold the modality is considered collapsed).
    if "text_var_history" in embed_arrays and len(
            embed_arrays["text_var_history"]) > 0:
        fig2, ax = plt.subplots(figsize=(10, 3), dpi=130)
        steps = np.arange(len(embed_arrays["text_var_history"]))
        ax.plot(steps, embed_arrays["text_var_history"],
               color="#E24B4A", lw=1.4, label="text emb variance")
        ax.plot(steps, embed_arrays["ts_var_history"],
               color="#3B8BD4", lw=1.4, label="TS emb variance")
        ax.axhline(0.05, color="#888", lw=0.8, ls="--",
                  label="collapse threshold (0.05)")
        ax.set_xlabel("Buffer-update step")
        ax.set_ylabel("Intra-batch variance (mean over dims)")
        _style_legend(ax, loc="upper center", ncol=3,
                     bbox_to_anchor=(0.5, -0.2), fontsize=14)
        ax.grid(True, lw=0.35, alpha=0.4)
        plt.tight_layout()
        vpath = save_path.replace(".png", "_variance.png")
        plt.savefig(vpath, bbox_inches="tight")
        plt.close(fig2)
        print(f"[clip-align] variance monitor -> {vpath}")


def tail_report(results, fracs=(0.01, 0.05)):
    """Decomposition of the contribution of the top-k% online steps to MSE (recompute after excluding them; see whether the ranking flips)."""
    print(f"\n{'=' * 74}\n[TAIL] contribution of the top-k% online steps to MSE\n{'=' * 74}")
    for frac in fracs:
        print(f"\n-- exclude top {frac * 100:.0f}% squared-error steps --")
        print(f"{'mode':<26}{'MSE(all)':>10}{'MSE(excl)':>11}"
              f"{'tail share':>12}{'MAE(all)':>10}{'MAE(excl)':>11}")
        for r in results:
            e2 = ((r["preds"] - r["truths"]) ** 2).mean(axis=(1, 2))
            e1 = np.abs(r["preds"] - r["truths"]).mean(axis=(1, 2))
            k = max(1, int(frac * len(e2)))
            order = np.argsort(e2)
            keep = order[:-k]
            share = e2[order[-k:]].sum() / (e2.sum() + 1e-12)
            print(f"{r['mode']:<26}{e2.mean():>10.4f}{e2[keep].mean():>11.4f}"
                  f"{share * 100:>11.1f}%{e1.mean():>10.4f}"
                  f"{e1[keep].mean():>11.4f}")
    print("(if iAdapter's MSE(excl) overtakes Unimodal while its MSE(all) lags behind, "
          "the MSE degradation is caused entirely by a few tail steps)")


def make_all_resonance_figures(mm, provider, results, tag="",
                               sensor_idx=0):
    """Produce all resonance figures + the tail report in one go (call on the M5 adapter that has a reslog attached)."""
    if getattr(mm, "reslog", None) is None:
        print("[reso] mm.reslog not attached, skipping resonance figures")
        return
    res = mm.reslog.arrays()
    tail_report(results)
    plot_text_resonance_timeline(
        res, results, sensor_idx=sensor_idx,
        save_path=f"iAdapter_resonance_timeline{tag}.png")
    plot_tsimage_resonance(
        res, results, sensor_idx=sensor_idx,
        save_path=f"iAdapter_tsimage_resonance{tag}.png")
    plot_energy_crosscorr(
        res, results, sensor_idx=sensor_idx,
        save_path=f"iAdapter_energy_crosscorr{tag}.png")


# ══════════════════════════════════════════════════════════════
# SECTION 10 — ENTRY POINT (ablation main loop + CLI)
# ══════════════════════════════════════════════════════════════

ABLATION_MODES = [
    ("Unimodal (ADAPT-Z)",      dict(fusion="add", use_text=False, use_text_grad=False)),
    ("+Text",                   dict(fusion="add", use_text=True,  use_text_grad=False)),
    ("+Text+Grad",              dict(fusion="add", use_text=True,  use_text_grad=True)),
    ("MoE Fusion",              dict(fusion="moe", use_text=True,  use_text_grad=False)),
    ("iAdapter", dict(fusion="moe", use_text=True,  use_text_grad=True)),
    ("iAdapter+PosGate ",  dict(fusion="moe_pos", use_text=True, use_text_grad=True)),
]

# VoT_z only: ablation of the z-injection position inside the backbone (the 4 candidate points of models/VoT_z.py + adaptive
# PosGate), used to argue whether "adaptively allocating the injection position" beats every single fixed position. All 5 modes
# keep use_text=False/use_text_grad=False constant — only the single variable "injection position"
# changes, without mixing in VoT_z's known "external use_text and internal USES_TEXT consuming the text twice"
# problem (which only exists in the explicit --modes m1..m6 escape-hatch scenario). The first row
# (pos=pre_gpt2) is the default behaviour of the legacy single fixed injection; the other 4 rows change the injection position.
VOT_Z_POSITION_MODES = [
    ("VoT_z (pos=pre_gpt2)", dict(fusion="add", use_text=False, use_text_grad=False, z_loc=3)),
    ("VoT_z (pos=trend)",          dict(fusion="add", use_text=False, use_text_grad=False, z_loc=1)),
    ("VoT_z (pos=seasonal)",       dict(fusion="add", use_text=False, use_text_grad=False, z_loc=2)),
    ("VoT_z (pos=post_gpt2)",      dict(fusion="add", use_text=False, use_text_grad=False, z_loc=4)),
    ("iAdapter ",   dict(fusion="moe_pos", use_text=True, use_text_grad=True)),
]

_MODE_KEYS = {"m1": 0, "m2": 1, "m3": 2, "m4": 3, "m5": 4, "m6": 5}


def parse_modes(spec: str):
    if spec.strip().lower() in ("all", ""):
        return None
    idx = sorted({_MODE_KEYS[t.strip().lower()]
                  for t in spec.split(",") if t.strip()})
    return [ABLATION_MODES[i] for i in idx]


def _gpt4mts_z_position_modes(e_layers):
    """GPT4MTS_z only: ablation of the per-layer z-injection position inside the backbone (frozen GPT2) [O8].

    The candidate positions reuse the existing {first layer, middle layer, last layer} heuristic of _position_candidates()
    (z_loc=k semantics of models/GPT4MTS_z.py::_gpt2_layerwise: larger k is deeper),
    plus z_loc=0 (the legacy single fixed pre-GPT2 injection, as a control group). use_text/
    use_text_grad are False throughout, for the same reason as GPT4MTS_z's original single-mode design: the backbone
    itself has USES_TEXT=True and already consumes the text directly, so an external adapter with use_text=True would process
    the same text twice, which is unfair; only the single variable "injection position" changes.

    Finally one more mode is appended that stacks the iAdapter itself at the deepest position (z_loc=gpt_layers, closest to the output)
    (fusion="moe": IGSoftMoEFusionAdapter multi-expert fusion, whose forward already
    consumes his_grad, i.e. the z-gradient feedback, unconditionally, regardless of use_text_grad), used to
    compare the separate gains of "only changing the injection position" and "injection position + iAdapter fusion", i.e.
    to show iAdapter's advantage when GPT4MTS_z is the backbone (plot_gpt4mts_z_advantage
    consumes this mode + the two deep baselines). use_text_grad is still False — that switch
    only gates the "text gradient" feedback (`self.use_text and self.use_text_grad`, see
    OnlineMM.val/train); with use_text=False it is a dead switch, so setting it to True would have no effect
    and would only mislead; hence it is not used."""
    gpt_layers = min(max(int(e_layers), 1), 12)
    mid = max(1, gpt_layers // 2)
    locs = sorted(set([0, 1, mid, gpt_layers]))
    labels = {
        0: "GPT4MTS_z (pos=pre_gpt2, legacy)",
        1: "GPT4MTS_z (pos=layer1, shallow)",
        mid: "GPT4MTS_z (pos=layer_mid, middle)",
        gpt_layers: "GPT4MTS_z (pos=layer_last, last-layer output)",
    }
    modes = [(labels[k], dict(fusion="add", use_text=False,
                              use_text_grad=False, z_loc=k)) for k in locs]
    modes.append((
        "iAdapter ",
        dict(fusion="moe", use_text=True, use_text_grad=True, z_loc=gpt_layers)
    ))
    return modes


def default_modes_for_backbone(backbone, e_layers=2):
    """GPT4MTS_z/VoT_z consume the text inside the backbone (USES_TEXT=True, directly taking the
    text_emb encoded by RealTextProvider); if the external IGSoftMoEFusionAdapter/
    AddFusionAdapter also used use_text=True, the same text would be processed twice, which is
    unfair. GPT4MTS_z [O8] now runs _gpt4mts_z_position_modes() (ablation of the per-layer
    z-injection position inside the backbone + iAdapter stacked at the deepest position, use_text False throughout); VoT_z
    supports multi-position dict-z injection (SUPPORTS_POS_GATE=True, see models/VoT_z.py) and
    runs the dedicated VOT_Z_POSITION_MODES (4 fixed injection positions + 1 adaptive PosGate,
    use_text False throughout to isolate the single variable "position"). Other backbones use the full
    iAdapter ablation list ABLATION_MODES.

    PatchTST's z-injection point is fixed at 'input2' (raw series before decomposition), so it does not support
    the multi-layer dict-z injection needed by 'iAdapter+PosGate' (explicitly excluded by design decision 7);
    therefore the PosGate row of ABLATION_MODES is removed for it alone, to avoid triggering OnlineMM's
    SUPPORTS_POS_GATE check error directly in a default full batch run."""
    if backbone == "GPT4MTS_z":
        return _gpt4mts_z_position_modes(e_layers)
    if backbone == "VoT_z":
        return VOT_Z_POSITION_MODES
    if backbone == "PatchTST":
        return [(label, kw) for label, kw in ABLATION_MODES
                if kw.get("fusion") != "moe_pos"]
    return ABLATION_MODES


def finetune_mm(data="Traffic", data_dir="./data", seq_len=None, pred_lens=None,
                seed=2023, batch_size=16, backbone= 'TimesNet',
                d_model=64, e_layers=2, z_loc=2,
                text_encoder="BERT", text_dim=64, llm_layers=6,
                dropout=0.1, weight_decay=1e-4, grad_clip=1.0, patience=3,
                warmup_rounds=3, his_mode="ema", his_momentum=0.9,
                text_source="both", text_lookback=2, max_clauses=6,
                max_chars=1200, keep_history_stats=False, ot_only=False,
                raw_text_topk=6, raw_lookback_days=60, use_text_cache=True,
                modes=None, make_figures=True, ckpt_dir=".",
                pretrain_epochs=15, pretrain_lr=3e-4,
                lr_overrides=None, csv_path="finally_backbone.csv",
                conf_momentum=0.7, log_resonance=True,
                use_clip_align=True, clip_dim=64, clip_weight=0.1):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    domain = canonical_domain(data)
    modes = modes or default_modes_for_backbone(backbone, e_layers)

    # [C1] the confidence EMA momentum is injected before the provider is constructed (class attribute, effective at construction)
    RealTextProvider.CONF_MOMENTUM = float(conf_momentum)

    # 1) Real-text provider: numeric + text in one place; the encoding cache is shared across modes
    provider = RealTextProvider(
        domain, data_dir=data_dir, text_source=text_source,
        text_lookback=text_lookback, max_clauses=max_clauses,
        max_chars=max_chars, raw_text_topk=raw_text_topk,
        raw_lookback_days=raw_lookback_days,
        keep_history_stats=keep_history_stats, ot_only=ot_only
    )

    # 2) Normalize pred_lens: None -> domain default; int -> single-element list
    ds_, dp_ = _DOMAIN_WINDOW.get(domain, (36, 6))
    base_seq_len = seq_len or ds_

    if pred_lens is None:
        pred_lens = list(dp_) if isinstance(dp_, (list, tuple)) else [dp_]
    elif isinstance(pred_lens, int):
        pred_lens = [pred_lens]
    else:
        pred_lens = list(pred_lens)

    # Adaptive settings for small-sample scenarios
    if provider.n_rows < 500:
        d_model = min(d_model, 32)
        pretrain_epochs = max(pretrain_epochs, 50)
        dropout = max(dropout, 0.2)
        batch_size = min(batch_size, 8)

    os.makedirs(ckpt_dir, exist_ok=True)

    all_results = {}  # {pred_len: [result_dict, ...]}

    # 3) Multi-horizon main loop
    for raw_pred_len in pred_lens:
        print(f"\n{'#' * 66}\n"
              f"[MULTI-H] domain={domain}  pred_len={raw_pred_len}\n"
              f"{'#' * 66}")

        # Shrink the window automatically for each pred_len independently
        sl, pl = auto_window(provider.n_rows, base_seq_len, raw_pred_len)

        # Backbones differ in z insertion dimension / layer count / effective d_model (SOFTS fixed at 256/3 layers,
        # TimesNet by time step rather than by number of variables)
        eff_d_model, eff_z_loc, z_n_dim = _backbone_dims(
            backbone, sl, pl, provider.numeric.shape[1], d_model, e_layers, z_loc
        )
        pos_candidates = _position_candidates(backbone, e_layers)

        # Load the data independently for each pred_len
        train_set, val_set, test_set, _sc, offsets = load_real_data_mm(
            provider, sl, pl
        )

        _replay_event("setup", locals())
        enc_in = provider.numeric.shape[1]
        print(f"[MM] enc_in={enc_in}, seq_len={sl}, pred_len={pl}")

        train_loader = DataLoader(
            train_set, batch_size=batch_size,
            shuffle=False, drop_last=True
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size,
            shuffle=False
        )

        args = argparse.Namespace(
            backbone=backbone,
            data=domain,
            seq_len=sl,
            pred_len=pl,
            seed=seed,
            device=device,
            batch_size=batch_size,
            batch_size2=batch_size,
            z_loc=eff_z_loc,
            z_shape=(batch_size, z_n_dim, eff_d_model),
            pos_candidates=pos_candidates,
            checkpoints="./checkpoints2",
            enc_in=enc_in,
            label_len=0,
            c_out=enc_in,
            freq="h"
        )

        ensure_settings(args.backbone, domain.lower(), lr_overrides)

        # 4) Independent checkpoint for each pred_len
        stem = (
            f"iAdapter_backbone_{backbone}_{domain}_L{sl}_h{pl}"
            f"_dm{eff_d_model}_el{e_layers}_c{enc_in}_seed{seed}.pth"
        )
        ckpt = os.path.join(ckpt_dir, stem)

        # Compatible with old names: reuse only when the shapes match
        legacy_names = [
            os.path.join(
                ckpt_dir,
                f"iAdapter_backbone_{domain}_h{pl}_seed{seed}.pth"
            ),
            f"iAdapter_backbone_{domain}_h{pl}_seed{seed}.pth",
        ]

        def _ckpt_matches(path):
            """
            Check whether the checkpoint's embedding.proj.weight matches the current
            (d_model, seq_len), to avoid a size mismatch.
            """
            try:
                sd = torch.load(path, map_location="cpu")
                w = sd.get("embedding.proj.weight")
                return w is not None and tuple(w.shape) == (d_model, sl)
            except Exception:
                return False

        if not os.path.exists(ckpt):
            reused = next(
                (p for p in legacy_names
                 if os.path.exists(p) and _ckpt_matches(p)),
                None
            ) if backbone == "iTransformer" else None

            if reused:
                ckpt = reused
                print(f"[backbone] reusing compatible legacy checkpoint: {ckpt}")
            else:
                stale = [p for p in legacy_names if os.path.exists(p)]
                if stale:
                    print(
                        f"[backbone] found legacy checkpoint(s) with different "
                        f"shape: {stale} -> ignored."
                    )

                print(
                    f"[backbone] checkpoint missing -> pretraining "
                    f"({pretrain_epochs} epochs) target={ckpt}"
                )

                set_seed(seed)
                bb = build_backbone(
                    backbone, sl, pl, enc_in, d_model, e_layers,
                    text_dim=text_dim
                ).to(device)

                pretrain_backbone(
                    bb, train_loader, val_loader, device,
                    epochs=pretrain_epochs,
                    lr=pretrain_lr,
                    patience=patience,
                    ckpt_path=ckpt,
                    z_shape=args.z_shape,
                    z_loc=eff_z_loc
                )

        def _load_backbone_state(bb):
            try:
                state = torch.load(ckpt, map_location=device)
            except Exception as e:
                # The file exists but is not a valid torch archive (typically when the last pretraining run was
                # interrupted by kill/OOM/full disk, leaving a truncated .pth) — a different kind of error from "shape mismatch",
                # so the corrupted file is deleted here and pretraining is rerun,
                # instead of reporting it to the user as a configuration error.
                print(
                    f"[backbone] checkpoint '{ckpt}' is corrupted/unreadable "
                    f"({e}) -> deleting and re-pretraining."
                )
                os.remove(ckpt)
                set_seed(seed)
                fresh_bb = build_backbone(
                    backbone, sl, pl, enc_in, d_model, e_layers,
                    text_dim=text_dim
                ).to(device)
                pretrain_backbone(
                    fresh_bb, train_loader, val_loader, device,
                    epochs=pretrain_epochs,
                    lr=pretrain_lr,
                    patience=patience,
                    ckpt_path=ckpt,
                    z_shape=args.z_shape,
                    z_loc=eff_z_loc
                )
                state = torch.load(ckpt, map_location=device)

            try:
                bb.load_state_dict(state)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to load backbone checkpoint '{ckpt}' for config "
                    f"backbone={backbone}, seq_len={sl}, pred_len={pl}, "
                    f"d_model={d_model}, e_layers={e_layers}, enc_in={enc_in}.\n"
                    f"This usually means the checkpoint was trained with a "
                    f"different backbone / seq_len / pred_len / d_model / enc_in. "
                    f"Delete the checkpoint file to re-pretrain, or pass "
                    f"matching arguments.\nOriginal error: {e}"
                ) from e

        def fresh_backbone():
            bb = build_backbone(backbone, sl, pl, enc_in, d_model, e_layers,
                               text_dim=text_dim)
            _load_backbone_state(bb)
            return bb.to(device).eval()

        # 5) Ablation main loop for the current pred_len
        results = []
        gate_hist_full = None
        m5_adapter = None                       # [C3] for the resonance figures

        for label, kw in modes:
            print(f"\n{'=' * 62}\n"
                  f"[RUN:{domain}] h={pl} {label}  {kw}  "
                  f"(REAL text: {text_source})\n"
                  f"{'=' * 62}")

            set_seed(seed)

            args.seq_len = sl
            args.pred_len = pl
            args.enc_in = enc_in
            # Each mode of VOT_Z_POSITION_MODES can carry a z_loc in kw,
            # overriding the default injection position (fixed-position ablation); other backbones/ABLATION_MODES
            # have no such key and fall back to eff_z_loc unchanged.
            args.z_loc = kw.get("z_loc", eff_z_loc)
            args.z_shape = (batch_size, z_n_dim, eff_d_model)

            # z_loc is only a position choice for the backbone forward; OnlineMM does not accept
            # it as a constructor argument — it is filtered out before being passed to OnlineMM, but the CSV below still uses
            # the full kw (including z_loc), for per-position review.
            online_kw = {k: v for k, v in kw.items() if k != "z_loc"}
            adapter = OnlineMM(
                backbone=fresh_backbone(),
                d_model=eff_d_model,
                args=args,
                provider=provider,
                split_offsets=offsets,
                text_encoder_type=text_encoder,
                text_dim=text_dim,
                llm_layers=llm_layers,
                dropout=dropout,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                patience=patience,
                update_backbone=False,
                his_mode=his_mode,
                his_momentum=his_momentum,
                use_text_cache=use_text_cache,
                enc_in=enc_in,
                use_clip_align=use_clip_align,
                clip_dim=clip_dim,
                clip_weight=clip_weight,
                **online_kw
            )

            # [C3] marking m5_adapter is decoupled from attaching the resonance logger: even when --no_resonance
            # turns off resonance logging (the counterfactual forward cost), the CLIP-alignment figure still needs to know which one is
            # M5, otherwise the vectors collected by embed_log below would never make it into a figure.
            if _is_ours(label):
                m5_adapter = adapter
                if log_resonance:
                    adapter.reslog = ResonanceLogger(counterfactual=True)

            # CLIP-alignment semantic-space visualization: collect vectors only for the M5 that has clip_align attached
            if make_figures and adapter.clip_align is not None:
                adapter.embed_log = EmbedAlignLogger()

            # val warm-up + early stopping
            best_mse, bad = float("inf"), 0

            for _ in range(warmup_rounds):
                mse = adapter.val(val_loader)

                if mse < best_mse:
                    best_mse, bad = mse, 0
                else:
                    bad += 1
                    print(f"[MM] val not improved ({bad}/{patience})")

                    if bad >= patience:
                        break

            mae, mse, preds, truths = adapter.online(test_loader)

            results.append(dict(
                mode=label,
                mae=mae,
                mse=mse,
                preds=preds,
                truths=truths,
                pred_len=pl,
                backbone=backbone,
                gate=list(adapter.gate_history)
            ))

            if _is_ours(label) and adapter.gate_history:
                gate_hist_full = list(adapter.gate_history)

            # CSV append: add a pred_len column
            pd.DataFrame([{
                "data": domain,
                "backbone": backbone,
                "seq_len": sl,
                "pred_len": pl,
                "seed": seed,
                "mode": label,
                **{k: kw.get(k, args.z_loc if k == "z_loc" else None)
                   for k in ("fusion", "use_text", "use_text_grad", "z_loc")},
                "text_source": text_source,
                "text_encoder": text_encoder,
                "his_mode": his_mode,
                "conf_momentum": conf_momentum,
                "mae": mae,
                "mse": mse,
            }]).to_csv(
                csv_path,
                mode="a",
                header=not os.path.exists(csv_path),
                index=False
            )

        all_results[pl] = results

        # 6) Gain summary for the current pred_len
        print(f"\n{'-' * 66}\n"
              f"[SUMMARY] {domain} h={pl} seed={seed} "
              f"text={text_source} his={his_mode} conf_m={conf_momentum}\n"
              f"{'-' * 66}")

        uni_mae = next(
            (r["mae"] for r in results if "Unimodal" in r["mode"]),
            results[0]["mae"]
        )
        uni_mse = next(
            (r["mse"] for r in results if "Unimodal" in r["mode"]),
            results[0]["mse"]
        )

        for r in results:
            g_mae = (uni_mae - r["mae"]) / uni_mae * 100
            g_mse = (uni_mse - r["mse"]) / uni_mse * 100

            print(
                f"  {_mode_nice(r['mode']):<40} "
                f"MAE={r['mae']:.4f} ({g_mae:+.1f}%)  "
                f"MSE={r['mse']:.4f} ({g_mse:+.1f}%)"
            )

        # 7) Figures for the current pred_len [PNG output had been turned off on request earlier; the core
        # ablation/gain figures are now re-enabled, to show the improvement of adaptive fusion over Unimodal directly;
        # the text-resonance figures (make_all_resonance_figures) stay off (the counterfactual forward
        # is expensive and has no direct bearing on "does adaptivity improve forecasting"); the CLIP-alignment semantic-space figure
        # has been re-enabled (see plot_clip_alignment_space below), independently of log_resonance]
        if make_figures:
            tag = f"_{domain}_h{pl}_s{seed}"

            plot_ablation_summary(
                results,
                f"iAdapter_ablation_summary{tag}.png"
            )

            # plot_rolling_mae(
            #     results,
            #     f"iAdapter_rolling_mae{tag}.png"
            # )
            #
            # plot_pred_curves(
            #     results,
            #     sensor_idx=0,
            #     save_path=f"iAdapter_pred_curves{tag}.png"
            # )
            #
            if gate_hist_full:
                plot_gate_usage(
                    gate_hist_full,
                    save_path=f"iAdapter_gate_usage{tag}.png"
                )

            # [C3] text energy × time-series signal resonance figures (timeline / periodic folding / cross-correlation)
            # if m5_adapter is not None:
            #     make_all_resonance_figures(
            #         m5_adapter, provider, results, tag=tag
            #     )
            #
            # CLIP-alignment semantic-space visualization: TS vectors / text vectors projected together into the shared space,
            # coloured by dominant period and jump strength (independent of log_resonance; a figure is produced whenever M5 has
            # clip_align attached and samples were collected)
                # [C3] text energy × time-series signal resonance timeline
            if (m5_adapter is not None
                    and getattr(m5_adapter, "reslog", None) is not None):
                _res = m5_adapter.reslog.arrays()
                plot_text_resonance_timeline(
                    _res, results,
                    save_path=f"iAdapter_resonance_timeline{tag}.png")


            # if m5_adapter is not None and getattr(
            #         m5_adapter, "embed_log", None) is not None:
            #     embed_arrays = m5_adapter.embed_log.arrays()
            #     if embed_arrays is not None:
            #         plot_clip_alignment_space(
            #             embed_arrays,
            #             save_path=f"iAdapter_clip_align_space{tag}.png")
            #     else:
            #         print("[clip-align] no samples collected (the buffer update may never have been triggered), "
            #               "skipping semantic-space visualization")

    # 8) Multi-pred_len summary figure
    if make_figures and len(pred_lens) > 1:
        plot_multi_pred_len_summary(
            all_results,
            save_path=f"iAdapter_multi_predlen_{domain}_s{seed}.png"
        )

    # 9) GPT4MTS_z only: zoomed figure of the forecast advantage of the best fixed position inside the backbone vs +iAdapter
    # (modelled on the Original vs +Energy grouped bar chart of xihe.png), reusing the same
    # all_results, with no extra training.
    if make_figures and backbone == "GPT4MTS_z":
        plot_gpt4mts_z_advantage(
            all_results,
            save_path=f"iAdapter_gpt4mts_advantage_{domain}_s{seed}.png"
        )

    return all_results


# ══════════════════════════════════════════════════════════════
# report_averages — report the average MSE/MAE of each configuration
# ══════════════════════════════════════════════════════════════

def report_averages(csv_path="finally_backbone.csv",
                    out_csv="iAdapter_avg_summary.csv"):
    """Read the run CSV and report the average MSE/MAE of each configuration (mean±std across seeds) and the overall average per mode."""
    if not os.path.exists(csv_path):
        print(f"[report] CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    need = {"data", "pred_len", "mode", "seed", "mae", "mse"}
    if not need.issubset(df.columns):
        print(f"[report] CSV missing columns: {need - set(df.columns)}")
        return None
    df = df.drop_duplicates(
        subset=["data", "pred_len", "mode", "seed"], keep="last")

    def _grouped_stats(keys):
        return df.groupby(keys, sort=False).agg(
            mae_mean=("mae", "mean"), mae_std=("mae", lambda s: s.std(ddof=0)),
            mse_mean=("mse", "mean"), mse_std=("mse", lambda s: s.std(ddof=0)),
            n_seed=("seed", "nunique"),
        ).reset_index()

    per = _grouped_stats(["data", "pred_len", "mode"])
    per_dm = _grouped_stats(["data", "mode"])
    per_m = _grouped_stats(["mode"])

    def _print_block(title, tbl, keys):
        print(f"\n{'=' * 78}\n[REPORT] {title}\n{'=' * 78}")
        print("  ".join(f"{k:<12}" for k in keys)
              + f"  {'MAE(mean±std)':<22}  {'MSE(mean±std)':<22}  seeds")
        for _, r in tbl.iterrows():
            kv = "  ".join(f"{str(r[k]):<12}" for k in keys)
            print(f"{kv}  {r['mae_mean']:.4f} ± {r['mae_std']:.4f}      "
                  f"{r['mse_mean']:.4f} ± {r['mse_std']:.4f}      "
                  f"{int(r['n_seed'])}")

    _print_block("Average MSE/MAE — per (domain, pred_len, mode), across seeds",
                 per, ["data", "pred_len", "mode"])
    _print_block("Average MSE/MAE — per (domain, mode), across pred_len×seed",
                 per_dm, ["data", "mode"])
    _print_block("Overall average MSE/MAE — per mode (across domain×pred_len×seed)",
                 per_m, ["mode"])

    uni = per_m[per_m["mode"].str.contains("Unimodal", case=False)]
    if len(uni):
        um, us = uni["mae_mean"].iloc[0], uni["mse_mean"].iloc[0]
        print(f"\n[REPORT] overall average gain relative to Unimodal(ADAPT-Z):")
        for _, r in per_m.iterrows():
            print(f"  {r['mode']:<28} "
                  f"MAE {(um - r['mae_mean']) / um * 100:+6.2f}%   "
                  f"MSE {(us - r['mse_mean']) / us * 100:+6.2f}%")

    per.to_csv(out_csv, index=False)
    per_dm.to_csv(out_csv.replace(".csv", "_by_domain.csv"), index=False)
    per_m.to_csv(out_csv.replace(".csv", "_by_mode.csv"), index=False)
    print(f"\n[report] saved -> {out_csv} (+ _by_domain / _by_mode)")
    return per, per_dm, per_m


def report_by_backbone(csv_path="finally_backbone.csv",
                       out_csv="iAdapter_avg_summary_by_backbone.csv"):
    """Read the run CSV and report the average MSE/MAE at two granularities, per (dataset, backbone) and overall (backbone only)
    (mean±std across pred_len×mode×seed)."""
    if not os.path.exists(csv_path):
        print(f"[report] CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    need = {"data", "backbone", "pred_len", "mode", "seed", "mae", "mse"}
    if not need.issubset(df.columns):
        print(f"[report] CSV missing columns for by-backbone report: "
             f"{need - set(df.columns)}")
        return None
    df = df.drop_duplicates(
        subset=["data", "backbone", "pred_len", "mode", "seed"], keep="last")

    def _grouped_stats(keys):
        return df.groupby(keys, sort=False).agg(
            mae_mean=("mae", "mean"), mae_std=("mae", lambda s: s.std(ddof=0)),
            mse_mean=("mse", "mean"), mse_std=("mse", lambda s: s.std(ddof=0)),
            n_seed=("seed", "nunique"), n_rows=("seed", "size"),
        ).reset_index()

    per_db = _grouped_stats(["data", "backbone"])
    per_b = _grouped_stats(["backbone"])

    def _print_block(title, tbl, keys):
        print(f"\n{'=' * 78}\n[REPORT] {title}\n{'=' * 78}")
        print("  ".join(f"{k:<12}" for k in keys)
              + f"  {'MAE(mean±std)':<22}  {'MSE(mean±std)':<22}  seeds")
        for _, r in tbl.iterrows():
            kv = "  ".join(f"{str(r[k]):<12}" for k in keys)
            print(f"{kv}  {r['mae_mean']:.4f} ± {r['mae_std']:.4f}      "
                  f"{r['mse_mean']:.4f} ± {r['mse_std']:.4f}      "
                  f"{int(r['n_seed'])}")

    _print_block("Average MSE/MAE — per (dataset, backbone), across pred_len×mode×seed",
                 per_db, ["data", "backbone"])
    _print_block("Overall average MSE/MAE — per backbone (across dataset×pred_len×mode×seed)",
                 per_b, ["backbone"])

    per_db.to_csv(out_csv, index=False)
    per_b.to_csv(out_csv.replace(".csv", "_overall.csv"), index=False)
    print(f"\n[report] saved -> {out_csv} (+ _overall)")
    return per_db, per_b


def paired_seed_report(csv_path="finally_backbone.csv", domain=None,
                       base_mode="Unimodal (ADAPT-Z)"):
    """[C3] same-seed paired-difference report: each mode − baseline, mean±std + sign counts.
    A must-read when differences between modes are far smaller than the cross-seed std — n/n or 0/n with the same sign means paired-significant."""
    if not os.path.exists(csv_path):
        print(f"[paired] CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if domain is not None:
        df = df[df["data"] == domain]
    df = df.drop_duplicates(
        subset=["data", "pred_len", "mode", "seed"], keep="last")
    for dom in sorted(df["data"].unique()):
        sub_d = df[df["data"] == dom]
        print(f"\n{'=' * 74}\n[PAIRED] {dom}: each mode − {base_mode}, "
              f"same-seed paired difference (negative = better)\n{'=' * 74}")
        for pl in sorted(sub_d["pred_len"].unique()):
            sub = sub_d[sub_d["pred_len"] == pl]
            piv_mae = sub.pivot_table(index="seed", columns="mode",
                                      values="mae")
            piv_mse = sub.pivot_table(index="seed", columns="mode",
                                      values="mse")
            if base_mode not in piv_mae.columns:
                continue
            print(f"\npred_len={pl}  (n_seed={len(piv_mae)})")
            for m in piv_mae.columns:
                if m == base_mode:
                    continue
                dm = (piv_mae[m] - piv_mae[base_mode]).dropna()
                ds = (piv_mse[m] - piv_mse[base_mode]).dropna()
                print(f"  {m:<26} dMAE={dm.mean():+.4f}±{dm.std(ddof=0):.4f} "
                      f"(worse {int((dm > 0).sum())}/{len(dm)})   "
                      f"dMSE={ds.mean():+.4f}±{ds.std(ddof=0):.4f} "
                      f"(worse {int((ds > 0).sum())}/{len(ds)})")


def main():
    ap = argparse.ArgumentParser(
        description="iAdapter main (REAL Time-MMD text, optimized)")
    ap.add_argument("--data", default="all",
                    help=f"domain | comma list | 'all' "
                         f"({', '.join(DOMAINS)}; Health=Health_US)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--seq_len", type=int, default=0, help="0=per-domain default")
    ap.add_argument("--pred_lens", default=" ",
                    help="comma-separated list of forecast lengths, e.g. '6,8,10,12'; "
                         "empty = the domain's default single value")
    ap.add_argument("--seeds", default="2023,2024,2025,2026,2027")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--backbone", default="VoT_z",
                    choices=["iTransformer", "SOFTS", "GPT4MTS_z",
                             "PatchTST", "Transformer",
                             "GPT4MTS_z", "VoT_z"],
                    help="backbone network; SOFTS requires --seq_len 96, the others have no such restriction; "
                         "use GPT4MTS_z/VoT_z with --modes all (when empty / not given "
                         "as specific m1..m6, it automatically narrows to the default mode set given by default_modes_for_backbone"
                         ": GPT4MTS_z is a single fair-comparison mode, "
                         "VoT_z is the 4 fixed positions + 1 adaptive "
                         "PosGate ablation set of VOT_Z_POSITION_MODES (SUPPORTS_POS_GATE=True, natively supports "
                         "m6/moe_pos); explicitly passing m1..m6 still bypasses this protection, which for "
                         "GPT4MTS_z may make the text be consumed twice)")
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--e_layers", type=int, default=6)
    ap.add_argument("--z_loc", type=int, default=2)
    ap.add_argument("--text_encoder", default="GPT2",
                    choices=["BERT", "GPT2", "GPT2M", "MODERNBERT", "T5",
                             "BART", "QWEN3EMB", "simple"])
    ap.add_argument("--text_dim", type=int, default=64)
    ap.add_argument("--llm_layers", type=int, default=6)
    ap.add_argument("--text_source", default="both",
                    choices=["fact", "preds", "both"])
    ap.add_argument("--text_lookback", type=int, default=2)
    ap.add_argument("--max_clauses", type=int, default=6)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--raw_text_topk", type=int, default=6)
    ap.add_argument("--raw_lookback_days", type=int, default=60)
    ap.add_argument("--keep_history_stats", action="store_true")
    ap.add_argument("--ot_only", action="store_true")
    ap.add_argument("--no_text_cache", action="store_true",
                    help="disable frozen-LLM pooled cache [O3]")
    ap.add_argument("--his_mode", default="ema",
                    choices=["ema", "latest", "sum"],
                    help="his_grad update [O2]; 'sum' reproduces legacy "
                         "unbounded accumulation")
    ap.add_argument("--his_momentum", type=float, default=0.9)
    ap.add_argument("--conf_momentum", type=float, default=0.8,
                    help="[C1] text-confidence EMA momentum (larger = smoother; "
                         "0 degenerates to a hard per-row indicator)")
    ap.add_argument("--no_resonance", action="store_true",
                    help="[C3] turn off resonance logging and figures (saves the counterfactual forward)")
    ap.add_argument("--no_clip_align", action="store_true",
                    help="turn off CLIP-style TS–text contrastive alignment (InfoNCE) and its "
                         "semantic-space visualization; only takes effect in modes with use_text=True")
    ap.add_argument("--clip_dim", type=int, default=64,
                    help="dimension of the CLIP-alignment shared space")
    ap.add_argument("--clip_weight", type=float, default=0.1,
                    help="weight of the CLIP-alignment InfoNCE loss")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--warmup_rounds", type=int, default=3)
    ap.add_argument("--modes", default="", help="e.g. m1,m5 or 'all' (empty = chosen automatically by "
                    "backbone, see the --backbone help)")
    ap.add_argument("--ckpt_dir", default=".")
    ap.add_argument("--pretrain_epochs", type=int, default=15)
    ap.add_argument("--pretrain_lr", type=float, default=1e-3)
    ap.add_argument("--adapter_lr", type=float, default=None)
    ap.add_argument("--finetune_lr", type=float, default=None)
    ap.add_argument("--online_lr", type=float, default=None)
    ap.add_argument("--adp_online_lr", type=float, default=None)
    ap.add_argument("--csv", default="finally_Security.csv")
    ap.add_argument("--no_figures", action="store_true", default=True)
    ap.add_argument("--preview", type=int, default=0,
                    help="N>0: only print N cleaned text samples per domain "
                         "and exit (no torch needed) [O10]")
    a = ap.parse_args()

    doms = (DOMAINS if a.data.strip().lower() == "all"
            else [canonical_domain(d) for d in a.data.split(",") if d.strip()])

    if a.preview > 0:                                    # plain-text preview
        RealTextProvider.CONF_MOMENTUM = float(a.conf_momentum)
        for d in doms:
            p = RealTextProvider(
                d, data_dir=a.data_dir, text_source=a.text_source,
                text_lookback=a.text_lookback, max_clauses=a.max_clauses,
                max_chars=a.max_chars, raw_text_topk=a.raw_text_topk,
                raw_lookback_days=a.raw_lookback_days,
                keep_history_stats=a.keep_history_stats, ot_only=a.ot_only)
            p.preview(a.preview)
        return

    if not _RUNTIME_OK:
        raise RuntimeError(
            f"Training mode needs torch/matplotlib/sklearn/tqdm. "
            f"Import failed with: {_IMPORT_ERR!r}\n"
            f"(--preview N works without them.)")
    pred_lens = ([int(x) for x in a.pred_lens.split(",") if x.strip()] or None)
    lr_over = dict(adapter_lr=a.adapter_lr, finetune_lr=a.finetune_lr,
                   online_lr=a.online_lr, adp_online_lr=a.adp_online_lr)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    for d, seed in itertools.product(doms, seeds):
        finetune_mm(
            data=d, data_dir=a.data_dir,
            seq_len=a.seq_len or None, pred_lens=pred_lens,
            seed=seed, batch_size=a.batch_size, backbone=a.backbone,
            d_model=a.d_model,
            e_layers=a.e_layers, z_loc=a.z_loc,
            text_encoder=a.text_encoder, text_dim=a.text_dim,
            llm_layers=a.llm_layers, dropout=a.dropout,
            weight_decay=a.weight_decay, grad_clip=a.grad_clip,
            patience=a.patience, warmup_rounds=a.warmup_rounds,
            his_mode=a.his_mode, his_momentum=a.his_momentum,
            text_source=a.text_source, text_lookback=a.text_lookback,
            max_clauses=a.max_clauses, max_chars=a.max_chars,
            keep_history_stats=a.keep_history_stats, ot_only=a.ot_only,
            raw_text_topk=a.raw_text_topk,
            raw_lookback_days=a.raw_lookback_days,
            use_text_cache=not a.no_text_cache,
            modes=parse_modes(a.modes), make_figures= a.no_figures,
            ckpt_dir=a.ckpt_dir, pretrain_epochs=a.pretrain_epochs,
            pretrain_lr=a.pretrain_lr, lr_overrides=lr_over, csv_path=a.csv,
            conf_momentum=a.conf_momentum,
            log_resonance=not a.no_resonance,
            use_clip_align=not a.no_clip_align,
            clip_dim=a.clip_dim, clip_weight=a.clip_weight)

    return a.csv


if __name__ == "__main__":
    _csv = main()
    if _csv:                              # --preview mode returns None; skip the summary
        _stem = _csv[:-4] if _csv.endswith(".csv") else _csv
        report_averages(_csv, out_csv=f"{_stem}_avg.csv")
        report_by_backbone(_csv, out_csv=f"{_stem}_by_backbone.csv")
        paired_seed_report(_csv)