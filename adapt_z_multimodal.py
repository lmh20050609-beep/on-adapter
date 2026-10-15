
from __future__ import annotations

import os
import re
import math
import random
import argparse
import itertools
from collections import Counter

import numpy as np
import pandas as pd

# ── 重依赖守护: --preview 在无 torch 环境同样可用 ──────────────────
_RUNTIME_OK, _IMPORT_ERR = True, None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
except Exception as _e:                                    # pragma: no cover
    _RUNTIME_OK, _IMPORT_ERR = False, _e
    from types import SimpleNamespace

    class Dataset:                                         # 占位
        pass

    class _ShimModule:                                     # 仅供类定义通过;
        pass                                               # 实例化受 _RUNTIME_OK 守护

    class _ShimNoGrad:                                     # @torch.no_grad() 占位
        def __call__(self, f):
            return f

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    nn = SimpleNamespace(Module=_ShimModule)
    torch = SimpleNamespace(no_grad=lambda: _ShimNoGrad())

# ── adapt_z 可选复用 (缺失则本地等价实现) ──────────────────────────
try:
    from adapt_z import set_seed, calculate_metrics, settings
except Exception:
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


# ══════════════════════════════════════════════════════════════
# SECTION 1 — DOMAIN REGISTRY (Time-MMD 9 域)
# ══════════════════════════════════════════════════════════════

DOMAINS = ["Agriculture", "Climate", "Economy", "Energy", "Environment",
           "Health_US", "Security", "SocialGood", "Traffic", "new_metr"]

_DOMAIN_ALIASES = {d.lower(): d for d in DOMAINS}
_DOMAIN_ALIASES.update({"health": "Health_US", "health_us": "Health_US",
                        "socialgood": "SocialGood", "social_good": "SocialGood"})

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
    "new_metr": "tianqiyubao",
}

# 启发式默认窗口 (seq_len, pred_len), 按域采样频率; CLI 可覆盖且会自动收缩
_DOMAIN_WINDOW = {
    "Agriculture": (36, 6),   "Climate": (36, 6),   "Economy": (36, 6),
    "Energy": (48, 8),        "Environment": (96, [48, 96, 192, 336]),
    "Health_US": (96, 12),    "Security": (36, 6),
    "SocialGood": (36, 6),    "Traffic": (36, 6), "new_metr": (96, [48, 96,192,336])
}


def canonical_domain(name: str) -> str:
    key = os.path.splitext(str(name).strip())[0].lower()
    if key in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[key]
    raise ValueError(f"Unknown domain '{name}'. Choose from {DOMAINS} "
                     f"(alias: Health -> Health_US).")

POSITION_NAMES = ("pre-GPT", "shallow", "middle", "tail")

BASELINE_LABEL = {
    "GPT4MTS_z": "GPT4MTS*",
    "VoT_z": "VoT*",
}

def parse_position_zlocs(spec):
    """
    Example:
      pre-GPT:0,shallow:1,middle:3,tail:6
    """
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, loc = item.split(":")
        name = name.strip()
        if name not in POSITION_NAMES:
            raise ValueError(
                f"Unknown position {name}. "
                f"Expected {POSITION_NAMES}"
            )
        out[name] = int(loc)

    missing = [p for p in POSITION_NAMES if p not in out]
    if missing:
        raise ValueError(f"Missing positions: {missing}")

    return out

# ══════════════════════════════════════════════════════════════
# SECTION 2 — 文本清洗 (纯 Python; 逐子句去 NA / 去流水线 boilerplate)
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
    """清洗单个子句; 无信息量返回 ''。"""
    if not s:
        return ""
    s = re.sub(r"\bNA(?=[A-Z(])", " ", s)          # 修复 "NAHere is" 拼接错位
    for pat in _DROP_PATTERNS:
        if pat.search(s):
            return ""
    for pat in _STRIP_PATTERNS:
        s = pat.sub(" ", s)
    s = _WS.sub(" ", s).strip(" ;,.-")
    if not s:
        return ""
    for pat in _DROP_PATTERNS:                     # 剥离后复检
        if pat.search(s):
            return ""
    return s


def split_date_segments(text: str):
    """'d1: seg1; d2: seg2 ...' -> [(date|None, body), ...] (按出现顺序)。"""
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
    清洗一个 fact/preds 单元格 -> (clean_text, n_kept, n_total)。
    保留 "YYYY-MM-DD:" 日期锚点; 信息量优先保留 **最新** 日期段
    (对在线预测最相关), 输出按时间正序。
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
# SECTION 3 — RealTextProvider (数值 + 真实文本, 两种布局, 抗泄漏)
# ══════════════════════════════════════════════════════════════

_TEXT_LIKE_COLS = {"date", "Date", "start_date", "end_date", "fact", "pred",
                   "preds", "Final_Output", "Final_Search_8", "Final_Search_16"}
_HISTORY_STAT_COLS = {"prior_history_avg", "prior_history_std"}


class RealTextProvider:
    """
    按数值行序供给清洗后的真实文本 (替换旧 TrafficTextGenerator):
      · self.numeric  : (n_rows, C) float32, NaN 已插值/填补, 未标准化;
      · self.row_text : 每个数值行清洗后的文本;
      · text_for_row(r): 行 r 及其前 text_lookback-1 行的聚合文本
        (全在历史侧); 原始布局匹配时强制 text.end_date <= 数值行.end_date
        —— 与 MM-TSFlib 抗泄漏约束一致, 未来目标行文本永不可见。
    """

    SIMPLE_MAX_LEN = 48
    _VOCAB_CAP = 4000
    _TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")

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
        # [O3] 冻结编码器缓存: {cache_key: {global_row: pooled/ids}}
        self.enc_cache: dict = {}
        if verbose:
            self.coverage_report()

    # ── 路径探测 ──
    def _resolve_paths(self):
        d, dd = self.domain, self.data_dir
        names = [d] + (["Health"] if d == "Health_US" else [])
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

    # ── 数值列选择 (NaN 线性插值, 与 TSlib uea.interpolate_missing 同法) ──
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

    # ── 布局 A: MM-TSFlib 合并 ──
    def _load_merged(self, path):
        df = pd.read_csv(path)
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

    # ── 布局 B: Time-MMD 原始 (区间匹配 + 抗泄漏: text.end <= 数值行.end) ──
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

    # ── 逐行清洗 ──
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

    # ── 对外接口 ──
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
        if len(joined) > self.max_chars:                    # 左截, 保最新
            joined = joined[-self.max_chars:]
            sp = joined.find(" ")
            if 0 < sp < 80:
                joined = joined[sp + 1:]
        return joined

    # ── simple 编码器词表 (真实语料) ──
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

    # ── 诊断 ──
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
                  f"informative={bool(self.row_has_info[r])}")
            print(f"  RAW   : {raw[:320]}{'...' if len(raw) > 320 else ''}")
            print(f"  CLEAN : {self.row_text[r][:320]}"
                  f"{'...' if len(self.row_text[r]) > 320 else ''}")


# ══════════════════════════════════════════════════════════════
# SECTION 4 — DATA (60/20/20 切分, 返回 split 行偏移供文本对齐)
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
    """按数据长度自动收缩窗口, 保证各 split 有足够样本。"""
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
    与原 load_real_data 相同的 60/20/20 切分与 scaler 逻辑, 额外返回每个
    split 的原始行偏移 offsets —— 窗口索引 w 的输入窗末行全局行号
    = offsets[split] + w + seq_len - 1 (真实文本对齐的关键)。
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
    """为新域向 settings 注入 lr 配置 (运行期注入, 不改 adapt_z.py)。 [O8]"""
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
    分布偏移统计特征 (B, 8): 供 MoE 门控感知 concept drift。
    x 已被训练集 StandardScaler 归一化, 故训练分布 ≈ N(0,1)。
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
# SECTION 5 — TEXT ENCODERS (frozen LLM, pool/project 拆分以支持缓存 [O3])
# ══════════════════════════════════════════════════════════════

class SimpleTextEncoder(nn.Module):
    """轻量可训练文本编码器 (Embedding + Transformer + mean-pool)。"""
    def __init__(self, vocab_size, embed_dim=64, output_dim=64, max_len=48,
                 nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                         dim_feedforward=embed_dim * 2,
                                         dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, ids):
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        x = self.token_embed(ids) + self.pos_embed(pos)
        mask = (ids == 0)
        x = self.transformer(x, src_key_padding_mask=mask)
        m = (~mask).float().unsqueeze(-1)
        x = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return self.norm(self.proj(x))


class LLMTextEncoder(nn.Module):
    """
    冻结 LLM 文本编码器 (BERT / GPT2), 仅投影 MLP 可训练 —— 不微调。
    [O3] 拆分为 pool() (no_grad, 结果可按行缓存) 与 project() (可训练):
         forward(texts) == project(pool(texts))。
    """
    _SPECS = {
        "GPT2":  ("gpt2", "D:/keyan/Models/gpt2",         768),
        "GPT2M": ("gpt2", "openai-community/gpt2-medium", 1024),
        "BERT":  ("bert", "D:/keyan/Models/bert-base-uncased", 768),
    }

    def __init__(self, llm_tag="BERT", output_dim=64, use_fullmodel=False,
                 llm_layers=6, max_length=64, dropout=0.1, device="cpu",
                 corrupt_mode="mask", corrupt_ratio=0.1):
        super().__init__()
        from transformers import (BertConfig, BertModel, BertTokenizer,
                                  GPT2Config, GPT2Model, GPT2Tokenizer,
                                  AutoTokenizer)
        if llm_tag not in self._SPECS:
            raise ValueError(f"Unknown llm_tag '{llm_tag}'")
        kind, hf_name, hidden = self._SPECS[llm_tag]
        self.hidden = hidden
        self.use_fullmodel = use_fullmodel
        self.max_length = max_length
        self._device = torch.device(device)
        assert corrupt_mode in ("none", "mask", "noise")
        self.corrupt_mode = corrupt_mode
        self.corrupt_ratio = float(corrupt_ratio)

        def _load(loader, name, **extra):
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            _basename = name.split("/")[-1]
            for _local in [os.path.join(_script_dir, _basename),
                           os.path.join(os.getcwd(), _basename)]:
                _zip = _local + ".zip"
                if not os.path.isdir(_local) and os.path.isfile(_zip):
                    import zipfile
                    print(f"[LLMTextEncoder] Extracting {_zip}")
                    with zipfile.ZipFile(_zip, "r") as zf:
                        zf.extractall(os.path.dirname(_zip))
                if os.path.isdir(_local):
                    try:
                        return loader.from_pretrained(_local,
                                                      local_files_only=True,
                                                      **extra)
                    except Exception:
                        pass
            try:
                return loader.from_pretrained(name, local_files_only=True,
                                              **extra)
            except Exception:
                pass
            return loader.from_pretrained(name, **extra)

        if kind == "bert":
            cfg = _load(BertConfig, hf_name)
            cfg.num_hidden_layers = llm_layers
            self.llm = _load(BertModel, hf_name, config=cfg,
                             trust_remote_code=True)
            try:
                self.tok = _load(BertTokenizer, hf_name)
            except Exception:
                self.tok = _load(AutoTokenizer, hf_name)
        else:
            cfg = _load(GPT2Config, hf_name)
            cfg.num_hidden_layers = llm_layers
            self.llm = _load(GPT2Model, hf_name, config=cfg,
                             trust_remote_code=True)
            try:
                self.tok = _load(GPT2Tokenizer, hf_name)
            except Exception:
                self.tok = _load(AutoTokenizer, hf_name)

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "[PAD]"
        for p in self.llm.parameters():
            p.requires_grad = False               # ← frozen, never fine-tuned
        self.llm.eval().to(self._device)

        self.proj = nn.Sequential(
            nn.Linear(hidden, max(hidden // 8, output_dim)), nn.ReLU(),
            nn.Linear(max(hidden // 8, output_dim), output_dim), nn.ReLU(),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(output_dim)

    @torch.no_grad()
    def pool(self, texts):
        """texts -> (B, hidden) 冻结池化表征 (逐行缓存的对象)。
        [robust] corrupt_mode="mask"/"noise" 在 token 级别作用于送入 LLM
        前的 input_ids/embedding, 用于文本分支鲁棒性消融 (--text_corrupt)。
        """
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tok(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_length)
        input_ids = enc["input_ids"].to(self._device)
        attn = enc["attention_mask"].to(self._device)

        ratio = self.corrupt_ratio
        if self.corrupt_mode != "none" and ratio > 0:
            valid = attn.bool()  # 只腐化真实 token, 不碰 padding
            corrupt_sel = (torch.rand_like(attn, dtype=torch.float32) < ratio) & valid

            if self.corrupt_mode == "mask":
                pad_id = self.tok.pad_token_id
                if pad_id is None:
                    pad_id = self.tok.eos_token_id
                input_ids = input_ids.masked_fill(corrupt_sel, pad_id)
                attn = attn.masked_fill(corrupt_sel, 0)
                emb = self.llm.get_input_embeddings()(input_ids)
            else:  # "noise": 对被选中 token 的输入 embedding 加高斯噪声
                emb = self.llm.get_input_embeddings()(input_ids)
                noise = torch.randn_like(emb) * emb.std()
                emb = emb + noise * corrupt_sel.unsqueeze(-1).float()
        else:
            emb = self.llm.get_input_embeddings()(input_ids)

        if self.use_fullmodel:
            emb = self.llm(inputs_embeds=emb, attention_mask=attn).last_hidden_state
        mask = attn.unsqueeze(-1).expand_as(emb)
        return (emb * mask).sum(1) / mask.sum(1).clamp(min=1)

    def project(self, pooled):
        """(B, hidden) -> (B, output_dim), 可训练投影。"""
        return self.norm(self.proj(pooled))

    def forward(self, texts):
        return self.project(self.pool(texts))


# ══════════════════════════════════════════════════════════════
# SECTION 6 — ADAPTERS (方法主体, 与原实现一致; 仅 [O9] 微修)
#   (a) AddFusionAdapter : 加性融合 (M1-M3)
#   (b) MoEFusionAdapter : 异构专家自适应融合 (M4-M5, ours)
# ══════════════════════════════════════════════════════════════

class AddFusionAdapter(nn.Module):
    """加性融合适配器: h = h_feat + h_grad (+ h_text)。"""
    def __init__(self, d_model, z_shape, text_dim=64, hidden=64, dropout=0.0,
                 use_text=True, use_text_grad=True, device="cpu"):
        super().__init__()
        self.use_text, self.use_text_grad = use_text, use_text_grad
        D_z = z_shape[-1]
        self.linear = nn.Linear(D_z, hidden)
        self.linear_ = nn.Linear(d_model, hidden)
        # [O9] 直接按 (N, D) 构造, 语义同原 torch.ones(z_shape)[0]
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


# ── heterogeneous experts (per-variate embedding axis D_z) ──

class IdentityExpert(nn.Module):
    """零初始化残差专家: 允许门控选择「不做修正」。"""
    def __init__(self, hidden, d_z):
        super().__init__()
        self.out = nn.Linear(hidden, d_z)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, feature, g_h, txt_mod, f_h):
        return self.out(h)


class TrendExpert(nn.Module):
    """低秩瓶颈 → 平滑趋势型修正。"""
    def __init__(self, hidden, d_z, rank=2):
        super().__init__()
        self.pre = nn.Linear(hidden, d_z)
        self.down = nn.Linear(d_z, rank, bias=False)
        self.up = nn.Linear(rank, d_z, bias=False)

    def forward(self, h, feature, g_h, txt_mod, f_h):
        return self.up(self.down(self.pre(h)))


class SeasonalityExpert(nn.Module):
    """Fourier 低频模截断 → 周期型修正。"""
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
    """高通 + 最新 Z 梯度 → 误差驱动高频修正。"""
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
    """唯一持有文本通道的专家: 文本 → FiLM(γ, β) 调制特征。"""
    def __init__(self, hidden, d_z, text_dim):
        super().__init__()
        self.film = nn.Linear(text_dim, hidden * 2)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_z))

    def forward(self, h, feature, g_h, txt_mod, f_h):
        gamma, beta = self.film(txt_mod).unsqueeze(1).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        return self.out(f_h * (1.0 + gamma) + beta)


class MoEFusionAdapter(nn.Module):
    """
    异构专家自适应融合适配器 (本文核心):
      · per-variate 门控 (B, N, E) + top-k 稀疏路由
      · 负载均衡辅助损失 self.aux_loss (Switch-Transformer 风格)
      · 门控输入包含分布偏移统计 offset_feat, 感知 concept drift
      · his_text_grad 以「方向」形式调制文本向量, 避免幅度失配
    """
    def __init__(self, d_model, z_shape, text_dim=64, hidden=64,
                 offset_dim=8, top_k=2, num_modes=8, aux_weight=0.01,
                 dropout=0.1, temperature=1.0, device="cpu",
                 use_text=True, use_text_grad=True):
        super().__init__()
        D_z = z_shape[-1]
        self.D_z, self.text_dim = D_z, text_dim
        self.offset_dim, self.temperature = offset_dim, temperature
        self.aux_weight = aux_weight
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
        self.top_k = min(top_k, self.num_experts)

        self.offset_norm = nn.LayerNorm(offset_dim)
        self.gate_f = nn.Linear(D_z, hidden)
        self.gate_g = nn.Linear(d_model, hidden)
        self.gate_t = nn.Linear(text_dim, hidden)
        self.gate_o = nn.Linear(offset_dim, hidden)
        self.gate_net = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(),
                                      nn.Linear(hidden, self.num_experts))

        self.last_gate = None
        self.aux_loss = torch.zeros(())

    def forward(self, feature, his_grad, text_emb=None, his_text_grad=None,
                offset_feat=None):
        B, N, _ = feature.shape
        device = feature.device
        if text_emb is None or not self.use_text:
            text_emb = torch.zeros(B, self.text_dim, device=device)
        if offset_feat is None:
            offset_feat = torch.zeros(B, self.offset_dim, device=device)

        g = (his_grad * self.lr).unsqueeze(0).expand(B, -1, -1)
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
        experts = torch.stack(outs, dim=2)                       # (B,N,E,D_z)

        o_exp = self.offset_norm(offset_feat).unsqueeze(1).expand(-1, N, -1)
        gate_h = torch.cat([
            torch.relu(self.gate_f(feature)), torch.relu(self.gate_g(g)),
            torch.relu(self.gate_t(txt_exp)), torch.relu(self.gate_o(o_exp)),
        ], dim=-1)
        logits = self.gate_net(gate_h) / max(self.temperature, 1e-6)  # (B,N,E)

        if self.top_k < self.num_experts:
            topv, topi = logits.topk(self.top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, topi, topv)
            logits = mask
        weights = torch.softmax(logits, dim=-1)
        self.last_gate = weights.detach()

        importance = weights.mean(dim=(0, 1))                    # (E,)
        self.aux_loss = self.aux_weight * self.num_experts * (importance ** 2).sum()

        return (experts * weights.unsqueeze(-1)).sum(dim=2)      # (B,N,D_z)


# ══════════════════════════════════════════════════════════════
# SECTION 7 — ONLINE MULTIMODAL ADAPTATION (On-Adapter, 优化版 OnlineMM)
# ══════════════════════════════════════════════════════════════

class OnlineMM:
    """
    多模态在线预测 (真实文本原生版)。

    关键设计 (方法不变):
      · update_backbone=False (默认) —— 骨干与投影头在线阶段不更新,
        适应完全来自专家自适应融合适配器, 与「在线微调」严格区分;
      · 双梯度反馈: his_grad = ∂L/∂z, his_text_grad = ∂L/∂text_emb;
      · few-shot 抗过拟合: dropout / weight_decay / grad clip / patience。

    实现优化 (见文件头 [O1]-[O6]):
      · 两处梯度反馈均改 torch.autograd.grad —— 不污染参数梯度,
        无 retain_graph, his_grad 不再无界累加 (--his_mode 可选);
      · 特征提取前向包 no_grad (adapter 参数梯度不依赖 feature 的图);
      · 冻结 LLM 池化按全局行号缓存, 跨消融模式共享;
      · 缓冲区只存所需批数 + his 快照 (N,D), 文本索引与数值切片严格对齐。
    """

    def __init__(self, model, d_model, args,
                 provider: RealTextProvider, split_offsets: dict,
                 fusion="add",                       # "moe" | "add"
                 text_encoder_type="BERT",           # "BERT"|"GPT2"|"GPT2M"|"simple"
                 text_dim=64, llm_layers=6,
                 use_text=True, use_text_grad=True,
                 update_backbone=False,
                 dropout=0.1, weight_decay=1e-4, grad_clip=1.0, patience=3,
                 top_k=2, aux_weight=0.01,
                 his_mode="ema", his_momentum=0.9, use_text_cache=True,
                 enc_in =1,
                 text_corrupt="none", text_corrupt_ratio=0.0):
        # ── [新增] enc_in 自适应专家复杂度 ──────────────────────────
        if enc_in <= 4:
            # 少通道域 (如 Agriculture OT-only): 专家多会引入噪声
            top_k = 5
            aux_weight = 0.05  # 更强负载均衡, 防单专家垄断
            if args.device != "cpu":
                print(f"[OnlineMM] enc_in={enc_in} ≤ 4 → top_k=5, "
                      f"aux_weight=0.05 (少通道自适应)")
        elif enc_in <= 12:
            top_k = min(top_k, 5)
            aux_weight = aux_weight  # 不变
        else:
            # 多通道域 (Environment/Health 可达 10+)
            top_k = min(top_k + 1, 5)  # 允许激活更多专家
            aux_weight = max(aux_weight * 0.5, 0.005)  # 放宽均衡约束
            print(f"[OnlineMM] enc_in={enc_in} > 12 → top_k={top_k}, "
                  f"aux_weight={aux_weight:.4f} (多通道自适应)")
        # ─────────────────────────────────────────────────────────────

        self.model = model
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
            self.adapter = MoEFusionAdapter(
                d_model=d_model, z_shape=args.z_shape, text_dim=text_dim,
                hidden=64, dropout=dropout, top_k=top_k,
                aux_weight=aux_weight, use_text=use_text,
                use_text_grad=use_text_grad,
                device=self.device).to(self.device)
        else:
            self.adapter = AddFusionAdapter(
                d_model=d_model, z_shape=args.z_shape, text_dim=text_dim,
                hidden=64, dropout=dropout, use_text=use_text,
                use_text_grad=use_text_grad,
                device=self.device).to(self.device)

        # ── 文本编码器 (真实文本; simple 用 provider 真实语料词表) [O1] ──
        self.text_enc = None
        self._cache_key = None
        if use_text:
            if text_encoder_type == "simple":
                if text_corrupt != "none":
                    print(f"[OnlineMM] 警告: text_encoder_type='simple' 不支持 "
                          f"text_corrupt='{text_corrupt}', 已忽略该参数。")
                self.text_enc = SimpleTextEncoder(
                    provider.vocab_size, embed_dim=text_dim,
                    output_dim=text_dim, max_len=provider.SIMPLE_MAX_LEN,
                    dropout=dropout).to(self.device)
                self._cache_key = ("simple_ids", provider.SIMPLE_MAX_LEN)
            else:
                self.text_enc = LLMTextEncoder(
                    llm_tag=text_encoder_type, output_dim=text_dim,
                    llm_layers=llm_layers, dropout=dropout,
                    device=self.device,
                    corrupt_mode=text_corrupt,
                    corrupt_ratio=text_corrupt_ratio).to(self.device)
                self._cache_key = ("pooled", text_encoder_type, llm_layers)

        self._adapter_params = list(self.adapter.parameters())
        self._text_params = ([p for p in self.text_enc.parameters()
                              if p.requires_grad]
                             if self.text_enc is not None else [])
        self._weight_decay = weight_decay
        self._settings_data = args.data.lower()
        self.text_dim = text_dim

        # 训练痕迹, 供可视化
        self.gate_history = []
        self.text_grad_history = []

    # ── helpers ──────────────────────────────────────────────

    def _rows_of_windows(self, window_idxs):
        """窗口起点索引 -> 输入窗末行全局行号 (clamp 到有效区间)。 [O1]"""
        last = self.provider.n_rows - 1
        s = self.args.seq_len - 1
        return [min(max(self._cur_offset + int(w) + s, 0), last)
                for w in window_idxs]

    def _encode_text(self, window_idxs):
        """真实文本 -> 嵌入; 冻结部分按全局行号缓存 [O3]。"""
        if not self.use_text or self.text_enc is None:
            return None
        rows = self._rows_of_windows(window_idxs)
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
            return self.text_enc.project(stacked)
        # simple: 缓存 token ids
        for r in dict.fromkeys(rows):
            if r not in cache:
                cache[r] = self.provider.tokenize(
                    self.provider.text_for_row(r))
        ids = torch.tensor([cache[r] for r in rows], dtype=torch.long,
                           device=self.device)
        return self.text_enc(ids)

    def _update_his(self, his, g):
        """[O2] his_grad 更新: ema (默认) / latest / sum (旧行为, 无界)。"""
        if g is None:
            return his
        if self.his_mode == "sum":
            return his + g
        if self.his_mode == "latest":
            return g
        return self.his_momentum * his + (1.0 - self.his_momentum) * g

    def _z_grad(self, x_b, y_b):
        """[O2] ∂L/∂z via autograd.grad: 不累加、不污染参数、无 retain_graph。
        返回 (grad(N,D)|None, feature.detach())。"""
        out = self.model(x_b, z=self.z, z_loc=self.args.z_loc)
        loss = self.loss_func(out["pred"], y_b)
        gz = torch.autograd.grad(loss, self.z, allow_unused=True)[0]
        gz = gz.mean(dim=0).detach() if gz is not None else None
        return gz, out["feature"].detach()

    def _compute_text_grad(self, x_b, y_b, feature_b, his_grad, text_emb,
                           his_text_grad=None, offset_b=None):
        """[O2] ∂L/∂text_emb via autograd.grad (调制上下文与推理一致)。"""
        txt = text_emb.detach().clone().requires_grad_(True)
        z = self.adapter(feature_b, his_grad, txt, his_text_grad, offset_b)
        out = self.model(x_b, z=z, z_loc=self.args.z_loc)
        loss = self.loss_func(out["pred"], y_b)
        gt = torch.autograd.grad(loss, txt, allow_unused=True)[0]
        if gt is not None:
            return gt.mean(dim=0).detach()
        return torch.zeros(txt.shape[-1], device=self.device)

    def _offset(self, x):
        return (compute_offset_features(x).to(self.device)
                if self.fusion == "moe" else None)

    def _keep_batches(self):
        """[O4] 缓冲区所需批数: 切片只用最近 pred_len+batch 行。"""
        need = self.args.pred_len + self.args.batch_size
        return int(math.ceil(need / self.args.batch_size)) + 1

    @staticmethod
    def _buffer_slice(arrs, sl):
        return np.concatenate(arrs, axis=0)[sl]

    def _buf_text_windows(self, global_idx, Bb):
        """[O5] 缓冲区样本的精确窗口索引: 末行=global_idx-1, 回退 pred 后取 Bb 个。"""
        b0 = max(0, global_idx - self.args.pred_len - Bb)
        return range(b0, b0 + Bb)

    # ── val / warm-up phase ──────────────────────────────────

    def val(self, val_loader, mode="val"):
        print(f"[MM] val (fusion={self.fusion}, text={self.use_text}, "
              f"text_grad={self.use_text_grad}, his={self.his_mode})")
        self._cur_offset = self._offsets["val"]
        lr_key = "finetune_lr" if mode == "train" else "adapter_lr"
        adp_lr = settings[self.args.model][self._settings_data][lr_key]
        wd = self._weight_decay
        all_params = self._adapter_params + self._text_params
        self.optimizer = optim.Adam(all_params, lr=adp_lr, weight_decay=wd)
        self.optimizer2 = (optim.Adam(                      # [O6] 仅 train 需要
            self.model.parameters(),
            lr=settings[self.args.model][self._settings_data]["online_lr"],
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

            text_emb = self._encode_text(range(global_idx, global_idx + B))
            offset = self._offset(x)

            with torch.no_grad():                           # [O2] 特征提取免建图
                feat = self.model(x, z=self.z, z_loc=self.args.z_loc)["feature"]
            z = self.adapter(feat, his_grad, text_emb, his_text_grad, offset)
            pred = self.model(x, z=z, z_loc=self.args.z_loc)["pred"]
            preds.append(pred.detach().cpu().numpy())

            loss = (self.loss_func(pred, y)
                    + getattr(self.adapter, "aux_loss", 0.0))
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
                self.model.zero_grad(set_to_none=True)      # 清理无主梯度

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
                    emb_buf = self._encode_text(
                        self._buf_text_windows(global_idx, x_b.size(0)))
                    if emb_buf is not None:
                        his_text_grad = self._compute_text_grad(
                            x_b, y_b, feat_b, his_grad, emb_buf,
                            his_text_grad, self._offset(x_b))
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

        if self.args.model == "iTransformer":
            proj_params = list(self.model.projector.parameters())
        elif hasattr(self.model, 'projection'):
            proj_params = list(self.model.projection.parameters())
        elif hasattr(self.model, 'out_layer'):
            proj_params = list(self.model.out_layer.parameters())
        else:
            raise AttributeError(f"Model {type(self.model).__name__} has no projector/projection/out_layer attribute")

        wd = self._weight_decay
        all_adp = self._adapter_params + self._text_params
        self.optimizer = optim.Adam(
            all_adp,
            lr=settings[self.args.model][self._settings_data]["adp_online_lr"],
            weight_decay=wd)
        self.optimizer2 = optim.Adam(
            proj_params,
            lr=settings[self.args.model][self._settings_data]["online_lr"],
            weight_decay=wd)

        his_grad = torch.zeros(self.args.z_shape[1:], device=self.device)
        his_text_grad = (torch.zeros(self.text_dim, device=self.device)
                         if self.use_text else None)
        z0 = torch.zeros(self.args.z_shape, device=self.device)   # [O4] 复用

        preds, truths = [], []
        x_list, feat_list = [], []                 # 滚动缓冲 (np)
        snap_list, cnt_list = [], []               # [O4] his 快照 (N,D) + 行数
        self.gate_history, self.text_grad_history = [], []
        keep = self._keep_batches()
        global_idx = 0

        for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            B = x.size(0)

            text_emb = self._encode_text(range(global_idx, global_idx + B))
            offset = self._offset(x)

            with torch.no_grad():                           # [O2]
                feat = self.model(x, z=z0, z_loc=self.args.z_loc)["feature"]
            z = self.adapter(feat, his_grad, text_emb, his_text_grad, offset)
            pred = self.model(x, z=z, z_loc=self.args.z_loc)["pred"]

            preds.append(pred.detach().cpu().numpy())
            truths.append(y.detach().cpu().numpy())
            x_list.append(x.detach().cpu().numpy())
            feat_list.append(feat.cpu().numpy())
            snap_list.append(his_grad.detach().cpu().numpy())      # (N,D)
            cnt_list.append(B)
            if getattr(self.adapter, "last_gate", None) is not None:
                lg = self.adapter.last_gate
                self.gate_history.append(
                    lg.mean(dim=tuple(range(lg.dim() - 1))).cpu().numpy())

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
                # [O4] 快照按行展开后取切片均值 == 原 hg_b.mean(dim=0)
                hg_rows = np.repeat(np.stack(snap_list, 0),
                                    cnt_list, axis=0)[sl]
                hg_mean = torch.from_numpy(
                    hg_rows.mean(axis=0)).to(self.device)
                offset_b = self._offset(x_b)

                # ── 梯度反馈 1: ∂L/∂z (autograd.grad, 无累加/污染) [O2] ──
                gz, _ = self._z_grad(x_b, y_b)
                his_grad = self._update_his(his_grad, gz)

                # ── 梯度反馈 2: ∂L/∂text_emb (精确缓冲索引 [O5]) ──
                emb_buf = None
                if self.use_text:
                    emb_buf = self._encode_text(
                        self._buf_text_windows(global_idx, x_b.size(0)))
                    if self.use_text_grad and emb_buf is not None:
                        his_text_grad = self._compute_text_grad(
                            x_b, y_b, ft_b, hg_mean, emb_buf,
                            his_text_grad, offset_b)
                        self.text_grad_history.append(
                            his_text_grad.cpu().numpy())

                # ── 适配器 (专家融合) 更新 —— 骨干默认不动 ──
                z_up = self.adapter(ft_b, hg_mean, emb_buf,
                                    his_text_grad, offset_b)
                out_up = self.model(x_b, z=z_up, z_loc=self.args.z_loc)
                loss_up = (self.loss_func(out_up["pred"], y_b)
                           + getattr(self.adapter, "aux_loss", 0.0))
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()
                loss_up.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_adp + proj_params,
                                             self.grad_clip)
                self.optimizer.step()
                if self.update_backbone:
                    self.optimizer2.step()
                self.model.zero_grad(set_to_none=True)      # 清理无主梯度
            global_idx += B

        pred_arr = np.concatenate(preds, 0)
        truth_arr = np.concatenate(truths, 0)
        mae, mse = calculate_metrics(pred_arr, truth_arr)
        print(f"[MM] online  mae={mae:.4f}  mse={mse:.4f}")
        return mae, mse, pred_arr, truth_arr


# ══════════════════════════════════════════════════════════════
# SECTION 8 — BACKBONE (ITransformerZ, checkpoint 兼容) + 自动预训练 [O8]
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
    """Minimal iTransformer with additive Z injection (checkpoint-compatible)."""
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
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if z is not None and z_loc is not None and z_loc == idx + 1:
                feature = h.clone()
                h = h + z[:B]
        if feature is None:
            feature = h.clone()
        pred = self.projector(self.norm(h)).permute(0, 2, 1)
        pred = pred * sigma[:, 0, :].unsqueeze(1) + mu[:, 0, :].unsqueeze(1)
        return {"pred": pred, "feature": feature}


def pretrain_backbone(bb, train_loader, val_loader, device,
                      epochs=50, lr=1e-3, patience=3, ckpt_path=None):
    """checkpoint 缺失时的骨干预训练 (早停+保存); 在线阶段骨干仍冻结。 [O8]"""
    opt = optim.Adam(bb.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best, bad, best_state = float("inf"), 0, None
    for ep in range(epochs):
        bb.train()
        tr = 0.0
        for xb, yb, *_ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = bb(xb)
            loss = lossf(out["pred"], yb)
            # VoT_z 返回 aux_loss (Endogenous Text Alignment)
            if "aux_loss" in out:
                loss = loss + out["aux_loss"]
            loss.backward()
            opt.step()
            tr += loss.item()
        bb.eval()
        vl, m = 0.0, 0
        with torch.no_grad():
            for xb, yb, *_ in val_loader:
                out = bb(xb.to(device))
                val_loss = lossf(out["pred"], yb.to(device))
                if "aux_loss" in out:
                    val_loss = val_loss + out["aux_loss"]
                vl += val_loss.item()
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
        torch.save(bb.state_dict(), ckpt_path)
        print(f"[pretrain] saved -> {ckpt_path}")
    return bb


# ══════════════════════════════════════════════════════════════
# SECTION 9 — POSITION SIGNIFICANCE TABLE [O11]
# ══════════════════════════════════════════════════════════════

SIGNIFICANCE_POSITIONS = ("pre-GPT", "shallow", "middle", "tail")
SIGNIFICANCE_BASELINES = ("GPT4MTS*", "VoT*")


def _paired_significance(base, enhanced, tie_atol=1e-12):
    """Paired MSE statistics used by Table~\\ref{tab:significance_position}.

    Positive signed difference means On-Adapter improves the baseline:
        d_i = baseline_mse_i - on_adapter_mse_i.

    Returns W/T/L counts, two-sided paired Wilcoxon p, rank-biserial r, and
    mean per-pair relative reduction Delta (%). Zero differences are counted
    as ties and excluded from the signed-rank sums (standard ``zero_method=wilcox``).
    """
    from scipy.stats import rankdata, wilcoxon

    base = np.asarray(base, dtype=float).reshape(-1)
    enhanced = np.asarray(enhanced, dtype=float).reshape(-1)
    if base.shape != enhanced.shape:
        raise ValueError(f"paired arrays must have same shape: {base.shape} vs {enhanced.shape}")
    ok = np.isfinite(base) & np.isfinite(enhanced)
    base, enhanced = base[ok], enhanced[ok]
    if len(base) == 0:
        raise ValueError("no finite paired observations")

    d = base - enhanced
    tie = np.isclose(d, 0.0, atol=tie_atol, rtol=0.0)
    win = d > tie_atol
    loss = d < -tie_atol
    w, t, l = int(win.sum()), int(tie.sum()), int(loss.sum())

    nz = ~tie
    if int(nz.sum()) == 0:
        p_value, r_rb = 1.0, 0.0
    else:
        # SciPy auto selects exact when valid and asymptotic otherwise.
        # Round the difference before testing to avoid artificial rank splitting
        # from floating-point subtraction, as recommended by scipy.stats.wilcoxon.
        d_test = np.round(d[nz], 12)
        try:
            res = wilcoxon(d_test, zero_method="wilcox", correction=False,
                           alternative="two-sided", method="auto")
        except TypeError:  # scipy<1.9 compatibility
            res = wilcoxon(d_test, zero_method="wilcox", correction=False,
                           alternative="two-sided")
        p_value = float(res.pvalue)

        ranks = rankdata(np.abs(d_test), method="average")
        w_plus = float(ranks[d_test > 0].sum())
        w_minus = float(ranks[d_test < 0].sum())
        denom = w_plus + w_minus
        r_rb = (w_plus - w_minus) / denom if denom > 0 else 0.0

    safe = np.abs(base) > 1e-15
    if not np.all(safe):
        raise ValueError("baseline_mse contains zero; relative reduction is undefined")
    delta = float(np.mean((base - enhanced) / base) * 100.0)
    return dict(n=int(len(base)), wins=w, ties=t, losses=l,
                p=p_value, r=float(r_rb), delta=delta)


def _significance_key_columns(df):
    """Infer the columns that uniquely identify a paired seed/horizon item."""
    preferred = ["data", "dataset", "seed", "pred_len", "horizon",
                 "run", "repeat", "fold"]
    keys = [c for c in preferred if c in df.columns]
    if not keys:
        raise ValueError(
            "Significance CSV needs pairing keys, e.g. seed + pred_len/horizon. "
            "Expected columns such as: baseline, position, seed, pred_len, "
            "baseline_mse, on_adapter_mse."
        )
    return keys


def compute_position_significance(df, data_name="Traffic", expected_n=40,
                                  tie_atol=1e-12):
    """Compute the four position rows + Avg. row for each baseline.

    Required columns
    ----------------
    baseline : str
        e.g. GPT4MTS* / VoT*
    position : str
        pre-GPT / shallow / middle / tail
    baseline_mse : float
    on_adapter_mse : float
    pairing keys : seed and pred_len/horizon (plus data/dataset when present)

    ``Avg.`` is *not* the arithmetic mean of row-level statistics. For each
    matched seed/horizon item it first averages On-Adapter MSE over the four
    positions, then performs a fresh paired test against that item's original
    baseline MSE. This keeps n unchanged (e.g. n=40), matching the paper table.
    """
    required = {"baseline", "position", "baseline_mse", "on_adapter_mse"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"significance CSV missing columns: {missing}")

    work = df.copy()
    data_col = "data" if "data" in work.columns else ("dataset" if "dataset" in work.columns else None)
    if data_col is not None and data_name:
        m = work[data_col].astype(str).str.lower() == str(data_name).lower()
        if not m.any():
            raise ValueError(f"no rows for {data_col}={data_name!r}")
        work = work.loc[m].copy()

    work["position"] = work["position"].astype(str).str.strip()
    unknown = sorted(set(work["position"]) - set(SIGNIFICANCE_POSITIONS))
    if unknown:
        raise ValueError(f"unknown position labels {unknown}; expected {SIGNIFICANCE_POSITIONS}")

    key_cols = _significance_key_columns(work)
    # data/dataset is constant after filtering, but retaining it in keys is harmless.
    group_cols = ["baseline", "position"] + key_cols
    dup = work.duplicated(group_cols, keep=False)
    if dup.any():
        print(f"[significance] WARN: {int(dup.sum())} duplicated paired rows; "
              "averaging duplicate MSE values within identical keys.")
        work = (work.groupby(group_cols, as_index=False)
                    .agg(baseline_mse=("baseline_mse", "mean"),
                         on_adapter_mse=("on_adapter_mse", "mean")))

    baselines_present = list(dict.fromkeys(work["baseline"].astype(str).tolist()))
    ordered_baselines = [b for b in SIGNIFICANCE_BASELINES if b in baselines_present]
    ordered_baselines += [b for b in baselines_present if b not in ordered_baselines]

    rows = []
    for baseline in ordered_baselines:
        bdf = work[work["baseline"].astype(str) == baseline].copy()
        for pos in SIGNIFICANCE_POSITIONS:
            pdf = bdf[bdf["position"] == pos]
            if pdf.empty:
                raise ValueError(f"missing rows for baseline={baseline!r}, position={pos!r}")
            st = _paired_significance(pdf["baseline_mse"], pdf["on_adapter_mse"], tie_atol)
            if expected_n and st["n"] != expected_n:
                print(f"[significance] WARN: {baseline}/{pos}: n={st['n']} "
                      f"(paper table expects n={expected_n})")
            rows.append(dict(baseline=baseline, position=pos, **st))

        # Build per-item position average, requiring all four positions.
        piv = bdf.pivot_table(index=key_cols, columns="position",
                              values="on_adapter_mse", aggfunc="mean")
        missing_pos = [p for p in SIGNIFICANCE_POSITIONS if p not in piv.columns]
        if missing_pos:
            raise ValueError(f"cannot form Avg. for {baseline}: missing {missing_pos}")
        piv = piv.dropna(subset=list(SIGNIFICANCE_POSITIONS))
        if piv.empty:
            raise ValueError(f"cannot form Avg. for {baseline}: no complete paired items")
        enh_avg = piv[list(SIGNIFICANCE_POSITIONS)].mean(axis=1)

        base_by_key = bdf.groupby(key_cols)["baseline_mse"].agg(["min", "max", "mean"])
        spread = (base_by_key["max"] - base_by_key["min"]).abs()
        if (spread > max(tie_atol, 1e-12)).any():
            print(f"[significance] WARN: {baseline}: baseline_mse differs across "
                  "positions for some paired keys; using the mean baseline MSE.")
        common = piv.index.intersection(base_by_key.index)
        st = _paired_significance(base_by_key.loc[common, "mean"],
                                  enh_avg.loc[common], tie_atol)
        if expected_n and st["n"] != expected_n:
            print(f"[significance] WARN: {baseline}/Avg.: n={st['n']} "
                  f"(paper table expects n={expected_n})")
        rows.append(dict(baseline=baseline, position="Avg.", **st))

    return pd.DataFrame(rows)

@torch.no_grad()
def evaluate_backbone_only(model, test_loader, device):
    model.eval()

    preds = []
    truths = []

    for batch in test_loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        # 原始 backbone，不注入 Z / On-Adapter
        out = model(x)

        preds.append(out["pred"].detach().cpu().numpy())
        truths.append(y.detach().cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0)
    truth_arr = np.concatenate(truths, axis=0)

    mae, mse = calculate_metrics(pred_arr, truth_arr)

    print(
        f"[BASELINE] mae={mae:.6f} "
        f"mse={mse:.6f}"
    )

    return mae, mse, pred_arr, truth_arr

def _fmt_p_latex(p):
    if p < 0.001:
        txt = r"<.001"
    else:
        txt = f"{p:.3f}".lstrip("0")
    return rf"$\bm{{{txt}}}$" if p < 0.05 else txt


def significance_table_latex(summary, data_name="Environment"):
    """Render the exact LaTeX layout used by the requested paper table."""
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Significance of On-Adapter gains at each insertion position",
        r"$Z\in\{$pre-GPT, shallow, middle, tail$\}$ on the " + str(data_name) + r" dataset. Each row reports a paired",
        r"Wilcoxon signed-rank test between the original baseline and its",
        r"On-Adapter-enhanced counterpart over per-horizon errors of all datasets",
        r"(MSE, $n{=}40$). $\Delta$\,(\%) is the mean relative reduction;",
        r"$r$ is the rank-biserial effect size.}",
        r"\label{tab:significance_position}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll ccc c}",
        r"\toprule",
        r"Baseline & Position $Z$ & \emph{W/T/L} & $p$ & $r$ & $\Delta$\,(\%) " + r"\\",
        r"\midrule",
    ]
    baseline_order = list(dict.fromkeys(summary["baseline"].tolist()))
    for bi, baseline in enumerate(baseline_order):
        b = summary[summary["baseline"] == baseline].copy()
        order = list(SIGNIFICANCE_POSITIONS) + ["Avg."]
        b["_ord"] = b["position"].map({p: i for i, p in enumerate(order)})
        b = b.sort_values("_ord")
        lines.append(rf"\multirow{{5}}{{*}}{{{baseline}}}")
        for i, row in enumerate(b.itertuples(index=False)):
            pos = r"\emph{Avg.}" if row.position == "Avg." else row.position
            wtl = f"{row.wins}/{row.ties}/{row.losses}"
            ptxt = _fmt_p_latex(float(row.p))
            rv = float(row.r)
            rtxt = (f"{rv:.2f}".lstrip("0") if rv >= 0
                    else "-" + f"{abs(rv):.2f}".lstrip("0"))
            dtxt = f"{float(row.delta):.1f}"
            if i == 0:
                lines.append(f" & {pos} & {wtl} & {ptxt} & {rtxt} & {dtxt} " + r"\\")
            else:
                if row.position == "Avg.":
                    lines.append(r"\cmidrule(l){2-6}")
                lines.append(f" & {pos:<9} & {wtl} & {ptxt} & {rtxt} & {dtxt} " + r"\\")
        if bi < len(baseline_order) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def run_significance_table(csv_path, data_name="Environment", expected_n=40,
                           out_summary="significance_position_summary.csv",
                           out_tex="significance_position_table.tex"):
    """Entry point for paper-table generation from paired experiment MSEs."""
    df = pd.read_csv(csv_path)
    summary = compute_position_significance(df, data_name=data_name,
                                            expected_n=expected_n)
    tex = significance_table_latex(summary, data_name=data_name)
    if out_summary:
        summary.to_csv(out_summary, index=False)
        print(f"[significance] summary -> {out_summary}")
    if out_tex:
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write(tex + "\n")
        print(f"[significance] LaTeX -> {out_tex}")
    print("\n" + tex)
    return summary, tex


# ══════════════════════════════════════════════════════════════
# SECTION 10 — VISUALISATION (保留函数; 默认不调用 [O12])
# ══════════════════════════════════════════════════════════════

_PALETTE = ["#9a9a9a", "#e0a840", "#e07040", "#4080c0", "#2e9e55"]

_MODE_COLOR = {
    "Unimodal (ADAPT-Z)":      "#9a9a9a",
    "+Text":                   "#e0a840",
    "+Text+Grad":              "#e07040",
    "Fusion":              "#4080c0",
    "On-Adapter": "#2e9e55",
}

_MODE_NICE = {
    "Unimodal (ADAPT-Z)":      "Unimodal (ADAPT-Z)",
    "+Text":                   "+Text  (semantic prior)",
    "+Text+Grad":              "+Text+Grad  (dual-gradient feedback)",
    "Fusion":              "+Fusion (adaptive fusion)",
    "On-Adapter ": "On-Adapter",     # [O7] typo 修复
}


def _mode_color(mode, i=0):
    return _MODE_COLOR.get(mode, _PALETTE[i % len(_PALETTE)])


def _mode_nice(mode):
    return _MODE_NICE.get(mode, mode)


def _is_ours(mode):
    return "On-Adapter" in mode


def _rolling(a, w):
    out = np.full(len(a), np.nan)
    for i in range(w - 1, len(a)):
        out[i] = a[i - w + 1:i + 1].mean()
    return out


def plot_ablation_summary(results, save_path="On-Adapter_ablation_summary.png"):
    """MAE / MSE 消融柱状图 (无标题), 标注相对 Unimodal 的收益百分比。"""
    names = [r["mode"] for r in results]
    maes = [r["mae"] for r in results]
    mses = [r["mse"] for r in results]
    uni_mae = next((r["mae"] for r in results if "Unimodal" in r["mode"]),
                   maes[0])
    uni_mse = next((r["mse"] for r in results if "Unimodal" in r["mode"]),
                   mses[0])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=130)
    for ax, vals, base, metric in ((axes[0], maes, uni_mae, "MAE"),
                                   (axes[1], mses, uni_mse, "MSE")):
        colors = [_mode_color(m, i) for i, m in enumerate(names)]
        edges = ["#1a1a1a" if _is_ours(m) else "white" for m in names]
        lws = [1.8 if _is_ours(m) else 0.8 for m in names]
        bars = ax.bar(range(len(names)), vals, color=colors,
                      edgecolor=edges, linewidth=lws, width=0.62, zorder=3)
        for b, v, m in zip(bars, vals, names):
            gain = (base - v) / base * 100
            tag = f"{v:.4f}" + ("" if abs(gain) < 1e-9 else f"\n({gain:+.1f}%)")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), tag,
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold" if _is_ours(m) else "normal")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([_mode_nice(m).replace("  ", "\n") for m in names],
                           fontsize=8)
        ax.set_ylabel(f"{metric}   (%: gain vs Unimodal ADAPT-Z)")
        ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {save_path}")


def plot_rolling_mae(results, save_path="On-Adapter_rolling_mae.png", W=30):
    """上: 滚动 MAE; 下: Δ|Error| = On-Adapter − Unimodal 净收益带。无标题。"""
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
    ax.legend(fontsize=8.5)
    ax.grid(True, lw=0.4, alpha=0.4)

    ours_key = next((m for m in step_mae if _is_ours(m)), None)
    uni_key = next((m for m in step_mae if "Unimodal" in m), None)
    if ours_key and uni_key:
        raw = step_mae[ours_key] - step_mae[uni_key]
        diff = _rolling(raw, W)
        steps = np.arange(len(diff))
        ax2.fill_between(steps, np.where(diff < 0, diff, 0), color="#2e9e55",
                         alpha=0.55, label="On-Adapter better")
        ax2.fill_between(steps, np.where(diff >= 0, diff, 0), color="#e05050",
                         alpha=0.45, label="On-Adapter worse")
        ax2.axhline(0, color="black", lw=0.8, ls="--")
        ax2.plot(diff, color="#666666", lw=0.7, alpha=0.6)
        ax2.text(0.99, 0.05, f"mean Δ|err| = {np.nanmean(raw):+.4f}",
                 transform=ax2.transAxes, ha="right", va="bottom", fontsize=9)
        ax2.set_ylabel("Δ|Error| (ours − Unimodal)")
        ax2.legend(fontsize=8, ncol=2)
    ax2.set_xlabel("Online step")
    ax2.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {save_path}")


def plot_pred_curves(results, sensor_idx=0, n_show=300,
                     save_path="On-Adapter_pred_curves.png"):
    """真值 vs 各模式预测 (第一个预测步), 上=开头段 / 下=末尾段。无标题。"""
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
    axes[0].legend(fontsize=7.5, ncol=3, loc="upper right")
    axes[1].set_xlabel("Online step")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {save_path}")


def plot_gate_usage(gate_history, expert_names=None,
                    save_path="On-Adapter_gate_usage.png"):
    """MoE 专家门控演化 (无标题)。"""
    if not gate_history:
        return
    G = np.array(gate_history)                          # (T, E)
    expert_names = expert_names or ["Identity", "Trend", "Seasonality",
                                    "Fluctuation", "Text-FiLM"]
    fig, ax = plt.subplots(figsize=(14, 4.6), dpi=130)
    ax.stackplot(np.arange(len(G)), G.T, labels=expert_names,
                 colors=["#bfbfbf", "#e0a840", "#40b060", "#4080c0",
                         "#d9985f"], alpha=0.85)
    ax.set_xlabel("Online step")
    ax.set_ylabel("Mean gate weight")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, ncol=5, loc="upper center")
    ax.grid(True, lw=0.35, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {save_path}")

def plot_multi_pred_len_summary(all_results: dict,
                                 save_path="On-Adapter_multi_predlen.png"):
    """
    多预测长度汇总图: 横轴=pred_len, 纵轴=MAE/MSE。
    每条线对应一种消融模式, 实线=On-Adapter, 虚线=其余。
    """
    pred_lens = sorted(all_results.keys())
    modes = [r["mode"] for r in next(iter(all_results.values()))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=130)
    for ci, mode in enumerate(modes):
        maes = [next(r["mae"] for r in all_results[pl] if r["mode"] == mode)
                for pl in pred_lens]
        mses = [next(r["mse"] for r in all_results[pl] if r["mode"] == mode)
                for pl in pred_lens]
        ours = _is_ours(mode)
        kw = dict(color=_mode_color(mode, ci),
                  lw=2.4 if ours else 1.2,
                  ls="-" if ours else "--",
                  marker="o" if ours else "s",
                  markersize=6 if ours else 4,
                  label=_mode_nice(mode),
                  zorder=6 if ours else 3)
        ax1.plot(pred_lens, maes, **kw)
        ax2.plot(pred_lens, mses, **kw)

    for ax, metric in ((ax1, "MAE"), (ax2, "MSE")):
        ax.set_xlabel("Prediction horizon (pred_len)")
        ax.set_ylabel(metric)
        ax.set_xticks(pred_lens)
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {save_path}")

# ══════════════════════════════════════════════════════════════
# SECTION 11 — ENTRY POINT (消融主循环 + CLI)
# ══════════════════════════════════════════════════════════════
#
# _MODE_KEYS = {"m1": 0, "m2": 1, "m3": 2, "m4": 3, "m5": 4}
#
# POSITION_ONADAPTER_CONFIG = [
#     ("Unimodal (ADAPT-Z)",      dict(fusion="add", use_text=False, use_text_grad=False)),
#     ("+Text",                   dict(fusion="add", use_text=True,  use_text_grad=False)),
#     ("+Text+Grad",              dict(fusion="add", use_text=True,  use_text_grad=True)),
#     ("Fusion",              dict(fusion="moe", use_text=True,  use_text_grad=False)),
#     ("On-Adapter ", dict(fusion="moe", use_text=True,  use_text_grad=True)),
# ]

_MODE_KEYS = {"m1": 0, "m2": 1}

POSITION_ONADAPTER_CONFIG = [
    ("Unimodal (ADAPT-Z)",      dict(fusion="add", use_text=False, use_text_grad=False)),
    ("On-Adapter ", dict(fusion="moe", use_text=True,  use_text_grad=True)),
]

def parse_modes(spec: str):
    if spec.strip().lower() in ("all", ""):
        return None
    idx = sorted({_MODE_KEYS[t.strip().lower()]
                  for t in spec.split(",") if t.strip()})
    return [POSITION_ONADAPTER_CONFIG[i] for i in idx]


def finetune_mm(data="Traffic", data_dir="./data", backbone="iTransformer",
                seq_len=None, pred_lens=None,
                seed=2023, batch_size=16, d_model=64, e_layers=2, z_loc=2,
                text_encoder="BERT", text_dim=64, llm_layers=6,
                dropout=0.1, weight_decay=1e-4, grad_clip=1.0, patience=3,
                warmup_rounds=3, his_mode="ema", his_momentum=0.9,
                text_source="both", text_lookback=2, max_clauses=6,
                max_chars=1200, keep_history_stats=False, ot_only=False,
                raw_text_topk=6, raw_lookback_days=60, use_text_cache=True,
                modes=None, make_figures=False, ckpt_dir=".",
                pretrain_epochs=15, pretrain_lr=3e-4,
                lr_overrides=None, csv_path="finally_traffic.csv",
                position_test=False, position_zlocs=None,
                position_csv="position_significance_traffic.csv",
                text_corrupt="none", text_corrupt_ratio=0.0):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    domain = canonical_domain(data)
    modes = modes or POSITION_ONADAPTER_CONFIG

    # 1) 真实文本供给器：数值 + 文本一站式；编码缓存跨模式共享
    provider = RealTextProvider(
        domain, data_dir=data_dir, text_source=text_source,
        text_lookback=text_lookback, max_clauses=max_clauses,
        max_chars=max_chars, raw_text_topk=raw_text_topk,
        raw_lookback_days=raw_lookback_days,
        keep_history_stats=keep_history_stats, ot_only=ot_only
    )

    # 2) pred_lens 规范化：None -> 域默认；int -> 单元素列表
    ds_, dp_ = _DOMAIN_WINDOW.get(domain, (36, 6))
    base_seq_len = seq_len or ds_

    if pred_lens is None:
        pred_lens = [dp_]
    elif isinstance(pred_lens, int):
        pred_lens = [pred_lens]
    else:
        pred_lens = list(pred_lens)

    # 小样本场景的自适应设置
    if provider.n_rows < 500:
        d_model = min(d_model, 32)
        pretrain_epochs = max(pretrain_epochs, 50)
        dropout = max(dropout, 0.2)
        batch_size = min(batch_size, 8)

    os.makedirs(ckpt_dir, exist_ok=True)

    all_results = {}  # {pred_len: [result_dict, ...]}

    # 3) 多 horizon 主循环
    for raw_pred_len in pred_lens:
        print(f"\n{'#' * 66}\n"
              f"[MULTI-H] domain={domain}  pred_len={raw_pred_len}\n"
              f"{'#' * 66}")

        # 每个 pred_len 独立自动收缩窗口
        sl, pl = auto_window(provider.n_rows, base_seq_len, raw_pred_len)

        # 每个 pred_len 独立加载数据
        train_set, val_set, test_set, _sc, offsets = load_real_data_mm(
            provider, sl, pl
        )

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

        # Determine z_shape based on backbone architecture
        if backbone == "GPT4MTS_z":
            from models.GPT4MTS_z import infer_patch_num
            patch_num = infer_patch_num(sl, pl, patch_size=16, stride=8)
            llm_dim = 768  # GPT-2's hidden dimension
            z_shape_init = (batch_size, patch_num, llm_dim)
            effective_d_model = llm_dim  # Use llm_dim for adapter
        elif backbone == "VoT_z":
            from models.VoT_z import infer_patch_num
            patch_num = infer_patch_num(sl, pl, patch_size=16, stride=8)
            llm_dim = 768  # GPT-2's hidden dimension
            z_shape_init = (batch_size, patch_num, llm_dim)
            effective_d_model = llm_dim  # Use llm_dim for adapter
        else:  # iTransformer or other backbones
            z_shape_init = (batch_size, enc_in, d_model)
            effective_d_model = d_model

        args = argparse.Namespace(
            model=backbone,  # Use actual backbone name, not hardcoded "iTransformer"
            data=domain,
            seq_len=sl,
            pred_len=pl,
            seed=seed,
            device=device,
            batch_size=batch_size,
            batch_size2=batch_size,
            z_loc=z_loc,
            z_shape=z_shape_init,
            checkpoints="./checkpoints2",
            enc_in=enc_in
        )

        ensure_settings(args.model, domain.lower(), lr_overrides)

        # 4) 每个 pred_len 独立 checkpoint
        stem = (
            f"On-Adapter_backbone_{backbone}_{domain}_L{sl}_h{pl}"
            f"_dm{d_model}_el{e_layers}_c{enc_in}_seed{seed}.pth"
        )
        ckpt = os.path.join(ckpt_dir, stem)

        # 兼容旧命名：仅当形状匹配时复用
        legacy_names = [
            os.path.join(
                ckpt_dir,
                f"On-Adapter_backbone_{domain}_h{pl}_seed{seed}.pth"
            ),
            f"On-Adapter_backbone_{domain}_h{pl}_seed{seed}.pth",
        ]

        def _ckpt_matches(path):
            """
            检查 checkpoint 的 embedding.proj.weight 是否与当前
            (d_model, seq_len) 匹配，避免 size mismatch。
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
            )

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

                if backbone == "iTransformer":
                    bb = ITransformerZ(
                        sl, pl, enc_in,
                        d_model=d_model,
                        e_layers=e_layers
                    ).to(device)
                elif backbone == "GPT4MTS_z":
                    from models.GPT4MTS_z import Model as GPT4MTS_z
                    bb = GPT4MTS_z(
                        sl, pl, enc_in,
                        text_dim=text_dim,
                        e_layers=e_layers,
                        device=device
                    ).to(device)
                elif backbone == "VoT_z":
                    from models.VoT_z import Model as VoT_z
                    bb = VoT_z(
                        sl, pl, enc_in,
                        text_dim=text_dim,
                        e_layers=e_layers,
                        device=device
                    ).to(device)
                else:
                    raise ValueError(f"Unknown backbone: {backbone}")

                pretrain_backbone(
                    bb, train_loader, val_loader, device,
                    epochs=pretrain_epochs,
                    lr=pretrain_lr,
                    patience=patience,
                    ckpt_path=ckpt
                )

        def fresh_backbone():
            if backbone == "iTransformer":
                bb = ITransformerZ(
                    sl, pl, enc_in,
                    d_model=d_model,
                    e_layers=e_layers
                )
            elif backbone == "GPT4MTS_z":
                from models.GPT4MTS_z import Model as GPT4MTS_z
                bb = GPT4MTS_z(
                    sl, pl, enc_in,
                    text_dim=text_dim,
                    e_layers=e_layers,
                    device=device
                )
            elif backbone == "VoT_z":
                from models.VoT_z import Model as VoT_z
                bb = VoT_z(
                    sl, pl, enc_in,
                    text_dim=text_dim,
                    e_layers=e_layers,
                    device=device
                )
            else:
                raise ValueError(f"Unknown backbone: {backbone}")

            try:
                bb.load_state_dict(torch.load(ckpt, map_location=device))
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to load backbone checkpoint '{ckpt}' for config "
                    f"backbone={backbone}, seq_len={sl}, pred_len={pl}, d_model={d_model}, "
                    f"e_layers={e_layers}, enc_in={enc_in}.\n"
                    f"This usually means the checkpoint was trained with a "
                    f"different seq_len / pred_len / d_model / enc_in. "
                    f"Delete the checkpoint file to re-pretrain, or pass "
                    f"matching arguments.\nOriginal error: {e}"
                ) from e

            return bb.to(device).eval()

        # ============================================================
        # Position significance experiment
        # ============================================================
        if position_test:

            if backbone not in ("GPT4MTS_z", "VoT_z"):
                raise ValueError(
                    "--position_test currently supports "
                    "GPT4MTS_z and VoT_z only."
                )

            if not position_zlocs:
                raise ValueError(
                    "--position_test requires position_zlocs"
                )

            baseline_name = BASELINE_LABEL[backbone]

            # --------------------------------------------------------
            # A. 原始 backbone baseline
            # 同一个 checkpoint / test set / seed / horizon
            # --------------------------------------------------------
            set_seed(seed)

            base_model = fresh_backbone()

            base_mae, base_mse, _, _ = evaluate_backbone_only(
                base_model,
                test_loader,
                device
            )

            position_results = []

            # --------------------------------------------------------
            # B. 四个位置分别插入完全相同的 5-expert On-Adapter
            # --------------------------------------------------------
            for position in POSITION_NAMES:

                current_z_loc = position_zlocs[position]

                print(
                    f"\n{'=' * 70}\n"
                    f"[POSITION TEST] "
                    f"{baseline_name}  "
                    f"position={position}  "
                    f"z_loc={current_z_loc}  "
                    f"seed={seed}  h={pl}\n"
                    f"{'=' * 70}"
                )

                set_seed(seed)

                # 当前位置
                args.z_loc = current_z_loc

                # 每个位置重新加载完全相同的 backbone checkpoint
                model_pos = fresh_backbone()

                adapter = OnlineMM(
                    model=model_pos,
                    d_model=effective_d_model,
                    args=args,
                    provider=provider,
                    split_offsets=offsets,

                    fusion="moe",
                    use_text=True,
                    use_text_grad=True,

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
                )

                # -------------------------
                # Validation warm-up
                # -------------------------
                best_mse = float("inf")
                bad = 0

                for _ in range(warmup_rounds):
                    val_mse = adapter.val(val_loader)

                    if val_mse < best_mse:
                        best_mse = val_mse
                        bad = 0
                    else:
                        bad += 1

                        if bad >= patience:
                            break

                # -------------------------
                # Online evaluation
                # -------------------------
                on_mae, on_mse, _, _ = adapter.online(
                    test_loader
                )

                rec = {
                    "data": domain,
                    "backbone": backbone,
                    "baseline": baseline_name,
                    "position": position,

                    "seed": seed,
                    "seq_len": sl,
                    "pred_len": pl,

                    "z_loc": current_z_loc,

                    "baseline_mae": base_mae,
                    "baseline_mse": base_mse,

                    "on_adapter_mae": on_mae,
                    "on_adapter_mse": on_mse,

                    "delta_mse_pct":
                        (base_mse - on_mse)
                        / base_mse * 100.0,

                    "text_source": text_source,
                    "text_encoder": text_encoder,
                    "his_mode": his_mode,
                }

                position_results.append(rec)

                # append raw paired observations
                pd.DataFrame([rec]).to_csv(
                    position_csv,
                    mode="a",
                    header=not os.path.exists(position_csv),
                    index=False
                )

                print(
                    f"[POSITION RESULT] "
                    f"{baseline_name} "
                    f"{position:<8} "
                    f"baseline={base_mse:.6f} "
                    f"OnAdapter={on_mse:.6f} "
                    f"Delta="
                    f"{rec['delta_mse_pct']:+.2f}%"
                )

            all_results[pl] = position_results

            # 不进入原来的 M1-M5 ablation loop
            continue

        # 5) 当前 pred_len 的消融主循环
        results = []
        gate_hist_full = None

        for label, kw in modes:
            print(f"\n{'=' * 62}\n"
                  f"[RUN:{domain}] h={pl} {label}  {kw}  "
                  f"(REAL text: {text_source})\n"
                  f"{'=' * 62}")

            set_seed(seed)

            args.seq_len = sl
            args.pred_len = pl
            args.enc_in = enc_in

            # Determine z_shape based on backbone architecture
            if backbone == "GPT4MTS_z":
                from models.GPT4MTS_z import infer_patch_num
                patch_num = infer_patch_num(sl, pl, patch_size=16, stride=8)
                llm_dim = 768  # GPT-2's hidden dimension
                args.z_shape = (batch_size, patch_num, llm_dim)
                effective_d_model = llm_dim  # Use llm_dim for adapter
            elif backbone == "VoT_z":
                from models.VoT_z import infer_patch_num
                patch_num = infer_patch_num(sl, pl, patch_size=16, stride=8)
                llm_dim = 768  # GPT-2's hidden dimension
                args.z_shape = (batch_size, patch_num, llm_dim)
                effective_d_model = llm_dim  # Use llm_dim for adapter
            else:  # iTransformer or other backbones
                args.z_shape = (batch_size, enc_in, d_model)
                effective_d_model = d_model

            adapter = OnlineMM(
                model=fresh_backbone(),
                d_model=effective_d_model,
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
                text_corrupt=text_corrupt,
                text_corrupt_ratio=text_corrupt_ratio,
                **kw
            )

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
                gate=list(adapter.gate_history)
            ))

            if "On-Adapter" in label and adapter.gate_history:
                gate_hist_full = list(adapter.gate_history)

            # CSV 追加：增加 pred_len 列
            pd.DataFrame([{
                "data": domain,
                "backbone": backbone,
                "seq_len": sl,
                "pred_len": pl,
                "seed": seed,
                "mode": label,
                **kw,
                "text_source": text_source,
                "text_encoder": text_encoder,
                "his_mode": his_mode,
                "text_corrupt": text_corrupt,
                "text_corrupt_ratio": text_corrupt_ratio,
                "mae": mae,
                "mse": mse,
            }]).to_csv(
                csv_path,
                mode="a",
                header=not os.path.exists(csv_path),
                index=False
            )

        all_results[pl] = results

        # 6) 当前 pred_len 的收益汇总
        print(f"\n{'-' * 66}\n"
              f"[SUMMARY] {domain} h={pl} seed={seed} "
              f"text={text_source} his={his_mode}\n"
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

        # 7) 当前 pred_len 单独出图
        if make_figures:
            tag = f"_{domain}_h{pl}_s{seed}"

            plot_ablation_summary(
                results,
                f"On-Adapter_ablation_summary{tag}.png"
            )

            plot_rolling_mae(
                results,
                f"On-Adapter_rolling_mae{tag}.png"
            )

            plot_pred_curves(
                results,
                sensor_idx=0,
                save_path=f"On-Adapter_pred_curves{tag}.png"
            )

            if gate_hist_full:
                plot_gate_usage(
                    gate_hist_full,
                    save_path=f"On-Adapter_gate_usage{tag}.png"
                )

    # 8) 多 pred_len 汇总图
    if make_figures and len(pred_lens) > 1:
        plot_multi_pred_len_summary(
            all_results,
            save_path=f"On-Adapter_multi_predlen_{domain}_s{seed}.png"
        )

    return all_results


def main():
    ap = argparse.ArgumentParser(
        description="On-Adapter main (REAL Time-MMD text, optimized)")
    ap.add_argument("--data", default="new_metr",
                    help=f"domain | comma list | 'all' "
                         f"({', '.join(DOMAINS)}; Health=Health_US)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--backbone", default="GPT4MTS_z",
                    choices=["iTransformer", "GPT4MTS_z", "VoT_z"],
                    help="Backbone model type")
    ap.add_argument("--seq_len", type=int, default=96, help="0=per-domain default")
    ap.add_argument("--pred_lens", default="48,96,192,336",
                    help="逗号分隔的预测长度列表, 如 '6,8,10,12'; "
                         "留空=域默认单值")
    ap.add_argument("--seeds", default="2028")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--z_loc", type=int, default=2)
    ap.add_argument("--text_encoder", default="GPT2",
                    choices=["BERT", "GPT2", "GPT2M", "simple"])
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
    ap.add_argument("--text_corrupt", default="none",
                    choices=["none", "mask", "noise"],
                    help="text-branch robustness ablation: token-level "
                         "mask (set to pad id + zero attention) or "
                         "Gaussian noise on token embeddings, applied "
                         "before the frozen LLM")
    ap.add_argument("--text_corrupt_ratio", type=float, default=0.0,
                    help="corruption ratio for --text_corrupt, e.g. 0.1 / 0.2")
    ap.add_argument("--his_mode", default="ema",
                    choices=["ema", "latest", "sum"],
                    help="his_grad update [O2]; 'sum' reproduces legacy "
                         "unbounded accumulation")
    ap.add_argument("--his_momentum", type=float, default=0.9)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--warmup_rounds", type=int, default=3)
    ap.add_argument("--modes", default="all", help="e.g. m1,m5 or 'all'")
    ap.add_argument("--ckpt_dir", default=".")
    ap.add_argument("--pretrain_epochs", type=int, default=15)
    ap.add_argument("--pretrain_lr", type=float, default=1e-3)
    ap.add_argument("--adapter_lr", type=float, default=None)
    ap.add_argument("--finetune_lr", type=float, default=None)
    ap.add_argument("--online_lr", type=float, default=None)
    ap.add_argument("--adp_online_lr", type=float, default=None)
    ap.add_argument("--csv", default="finally_mmweather.csv")
    # [O11] Paper significance table from already-collected paired MSE records.
    ap.add_argument("--significance_csv", default="",
                    help="paired MSE CSV -> W/T/L, Wilcoxon p, rank-biserial r, Delta table; then exit")
    ap.add_argument("--significance_data", default="Environment",
                    help="Dataset name for significance table filtering (auto-set from --data if single dataset)")
    ap.add_argument("--significance_expected_n", type=int, default=44)
    ap.add_argument("--significance_summary", default="significance_position_weather.csv")
    ap.add_argument("--significance_tex", default="significance_position_weather.tex")
    # [O12] Extra visualisations are opt-in for now.
    ap.add_argument("--figures", action="store_true",
                    help="enable ablation/rolling/prediction/gate figures (default: off)")
    ap.add_argument("--no_figures", action="store_true",
                    help=argparse.SUPPRESS)  # legacy compatibility; figures are already off by default
    ap.add_argument("--preview", type=int, default=0,
                    help="N>0: only print N cleaned text samples per domain "
                         "and exit (no torch needed) [O10]")

    ap.add_argument(
        "--position_test",
        action="store_true",
        help="run pre-GPT/shallow/middle/tail On-Adapter position test"
    )

    ap.add_argument(
        "--position_zlocs",
        default="pre-GPT:0,shallow:1,middle:3,tail:6",
        help=(
            "semantic position -> backbone z_loc mapping, e.g. "
            "'pre-GPT:0,shallow:1,middle:3,tail:6'"
        )
    )

    ap.add_argument(
        "--position_csv",
        default="position_significance_traffic.csv"
    )


    a = ap.parse_args()

    doms = (DOMAINS if a.data.strip().lower() == "all"
            else [canonical_domain(d) for d in a.data.split(",") if d.strip()])

    # Auto-set significance_data from --data if using default and running single dataset
    if a.significance_data == "Environment" and len(doms) == 1:
        a.significance_data = doms[0]
        print(f"[auto] Setting --significance_data to '{a.significance_data}' from --data")

    if a.preview > 0:                                    # 纯文本预览
        for d in doms:
            p = RealTextProvider(
                d, data_dir=a.data_dir, text_source=a.text_source,
                text_lookback=a.text_lookback, max_clauses=a.max_clauses,
                max_chars=a.max_chars, raw_text_topk=a.raw_text_topk,
                raw_lookback_days=a.raw_lookback_days,
                keep_history_stats=a.keep_history_stats, ot_only=a.ot_only)
            p.preview(a.preview)
        return

    if a.significance_csv:
        run_significance_table(
            a.significance_csv, data_name=a.significance_data,
            expected_n=a.significance_expected_n,
            out_summary=a.significance_summary, out_tex=a.significance_tex)
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

    # Parse position_zlocs if position_test is enabled
    position_zlocs_dict = None
    if a.position_test:
        position_zlocs_dict = {}
        for pair in a.position_zlocs.split(","):
            pos, loc = pair.split(":")
            position_zlocs_dict[pos.strip()] = int(loc.strip())

    for d, seed in itertools.product(doms, seeds):
        finetune_mm(
            data=d, data_dir=a.data_dir, backbone=a.backbone,
            seq_len=a.seq_len or None, pred_lens=pred_lens,
            seed=seed, batch_size=a.batch_size, d_model=a.d_model,
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
            modes=parse_modes(a.modes), make_figures=(a.figures and not a.no_figures),
            ckpt_dir=a.ckpt_dir, pretrain_epochs=a.pretrain_epochs,
            pretrain_lr=a.pretrain_lr, lr_overrides=lr_over, csv_path=a.csv,
            position_test=a.position_test, position_zlocs=position_zlocs_dict,
            position_csv=a.position_csv,
            text_corrupt=a.text_corrupt, text_corrupt_ratio=a.text_corrupt_ratio)

    # After all experiments complete, automatically compute significance table if position_test was enabled
    if a.position_test and os.path.exists(a.position_csv):
        print(f"\n{'=' * 80}")
        print(f"All position experiments complete. Computing significance table...")
        print(f"{'=' * 80}\n")

        run_significance_table(
            a.position_csv,
            data_name=a.significance_data,
            expected_n=a.significance_expected_n,
            out_summary=a.significance_summary,
            out_tex=a.significance_tex
        )

        print(f"\n{'=' * 80}")
        print(f"Significance table generated:")
        print(f"  Summary: {a.significance_summary}")
        print(f"  LaTeX:   {a.significance_tex}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()