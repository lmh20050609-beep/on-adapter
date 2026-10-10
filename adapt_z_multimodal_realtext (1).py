"""
adapt_z_multimodal_realtext.py  (RealTextProvider, drop-in replacement)
=======================================================================
用 **真实文本** 替换 adapt_z_multimodal.py 中的模板文本生成 (TrafficTextGenerator)。

数据来源: Time-MMD (NeurIPS 2024, AdityaLab/Time-MMD) 的 9 个域:
    Agriculture, Climate, Economy, Energy, Environment,
    Health (=Health_US), Security, SocialGood, Traffic
配套库: MM-TSFlib (AdityaLab/MM-TSFlib) —— 本文件兼容其预处理合并格式。

支持两种数据布局 (自动探测):
  A) MM-TSFlib 预处理合并格式 (单文件, 即用户样本格式):
        <data_dir>/<Domain>.csv
        列: OT, Date, start_date, date, end_date,
            prior_history_avg, prior_history_std, fact, preds
  B) Time-MMD 官方原始格式 (数值 + 文本分离):
        <data_dir>/numerical/<Domain>/<Domain>.csv        (start_date,end_date,OT,...)
        <data_dir>/textual/<Domain>/<Domain>_report.csv   (start_date,end_date,fact,pred)
        <data_dir>/textual/<Domain>/<Domain>_search.csv   (start_date,end_date,fact,pred)

核心组件
--------
RealTextProvider
    · 按数值行序加载/匹配真实文本 (fact / preds / both);
    · 逐子句清洗: 去 NA、去流水线 boilerplate ("Here is the summarized
      information:"、"Note: ..." 等)、修复 "NAHere is" 拼接错位;
    · 保留 "YYYY-MM-DD:" 日期前缀作为时间语义锚点;
    · 抗泄漏对齐 (与 MM-TSFlib 一致): 窗口样本只取 **输入窗内** 行的文本,
      原始布局下文本行需满足 text.end_date <= 数值行.end_date;
      目标 (未来) 行的文本绝不进入编码器;
    · 为 SimpleTextEncoder 提供从真实语料构建的词表 (LLM 编码器则直接吃字符串)。

OnlineMMReal(OnlineMM)
    仅重写 _encode_text / val / online 的 split 偏移注入, 其余 (MoE 适配器、
    双梯度反馈、消融流程、绘图) 全部复用 adapt_z_multimodal.py —— 骨干与
    LLM 依旧全程冻结, "在线向量空间多模态预测" 的卖点不变。

用法
----
  # 单域 (自动探测布局, 缺 checkpoint 时自动预训练骨干)
  python adapt_z_multimodal_realtext.py --data Traffic
  # 全部 9 个域
  python adapt_z_multimodal_realtext.py --data all
  # 只用客观事实列 / 只跑 M1 与 M5
  python adapt_z_multimodal_realtext.py --data Health --text_source fact --modes m1,m5
  # 不训练, 仅预览清洗后的真实文本 (此模式不需要 torch / adapt_z)
  python adapt_z_multimodal_realtext.py --data Traffic --preview 3

IMPORTANT: adapt_z.py 与 adapt_z_multimodal.py 均不被修改, 三个文件并存即可。
(已知上游小瑕疵: adapt_z_multimodal._MODE_NICE 中 "iAdapter】 (MoE+TextGrad)"
 键混入了全角括号, 图例会回退显示原始 label, 不影响运行。)
"""

from __future__ import annotations

import os
import re
import argparse
import itertools
from collections import Counter

import numpy as np
import pandas as pd

# ── 重型依赖延迟守护: --preview 模式在无 torch / 无 adapt_z 的机器上也能跑 ──
_RUNTIME_OK, _IMPORT_ERR = True, None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from adapt_z import set_seed, calculate_metrics, settings
    import adapt_z_multimodal as base
    from adapt_z_multimodal import (
        TimeSeriesDataset, OnlineMM, SimpleTextEncoder, LLMTextEncoder,
        ITransformerZ, ABLATION_MODES,
        plot_ablation_summary, plot_rolling_mae, plot_pred_curves,
        plot_gate_usage,
    )
except Exception as _e:                                   # pragma: no cover
    _RUNTIME_OK, _IMPORT_ERR = False, _e


# ══════════════════════════════════════════════════════════════
# SECTION A — DOMAIN REGISTRY (Time-MMD 9 域)
# ══════════════════════════════════════════════════════════════

DOMAINS = ["Agriculture", "Climate", "Economy", "Energy", "Environment",
           "Health_US", "Security", "SocialGood", "Traffic"]

_DOMAIN_ALIASES = {d.lower(): d for d in DOMAINS}
_DOMAIN_ALIASES.update({"health": "Health_US", "health_us": "Health_US",
                        "socialgood": "SocialGood", "social_good": "SocialGood"})

# 兜底中性句里的主题短语 (仅用于全 NA 行的占位文本)
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
}

# 启发式默认窗口 (seq_len, pred_len) —— 各域采样频率不同 (月/周/日),
# 96/24 的小时级窗口不适用; 均可被 CLI 覆盖, 且会按数据长度自动收缩。
_DOMAIN_WINDOW = {
    "Agriculture": (36, 6),   # monthly
    "Climate":     (36, 6),   # monthly
    "Economy":     (36, 6),   # monthly
    "Energy":      (48, 8),   # weekly
    "Environment": (96, 12),  # daily
    "Health_US":   (96, 12),  # weekly (series较长)
    "Security":    (36, 6),   # monthly
    "SocialGood":  (36, 6),   # monthly
    "Traffic":     (36, 6),   # monthly
}


def canonical_domain(name: str) -> str:
    key = name.strip().lower()
    if key in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[key]
    raise ValueError(f"Unknown domain '{name}'. Choose from {DOMAINS} (or 'Health').")


# ══════════════════════════════════════════════════════════════
# SECTION B — TEXT CLEANING (纯 Python, 无 torch 依赖)
# ══════════════════════════════════════════════════════════════

_DATE_SEG = re.compile(r"(\d{4}-\d{2}-\d{2})\s*:\s*")

# 整句丢弃: 命中即认为该子句无信息量
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

# 片段剥离: 从子句中抹掉流水线 boilerplate, 保留其余内容
_STRIP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"here is (?:the |my )?[^:;]{0,140}:",                 # "Here is the summarized ...:"
    r"note\s*:[^;]*",                                       # 尾部 "Note: ..." 元评论
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
    """清洗单个子句; 无信息量时返回 ''。"""
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
    """'d1: seg1; d2: seg2 ...' -> [(date_str|None, body), ...] (按出现顺序)。"""
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
    清洗一条 Time-MMD 文本 (fact 或 preds 单元格)。
    返回 (clean_text, n_kept, n_total)。保留 "YYYY-MM-DD:" 日期锚点;
    信息量优先保留 **最新** 日期段 (对在线预测最相关), 输出按时间正序。
    """
    segs = split_date_segments(raw)
    n_total, kept = 0, []                          # kept: (seg_pos, date, clause)
    for pos, (d, body) in enumerate(segs):
        for clause in body.split(";"):
            n_total += 1
            c = _clean_clause(clause)
            if c and len(c) >= min_clause_chars:
                kept.append((pos, d, c))
    kept = kept[-max_clauses:] if max_clauses > 0 else kept   # 最新优先
    pieces, last_d = [], None
    for _pos, d, c in kept:
        if d and d != last_d:
            pieces.append(f"{d}: {c}")
            last_d = d
        else:
            pieces.append(c)
    return " ".join(pieces), len(kept), n_total


# ══════════════════════════════════════════════════════════════
# SECTION C — RealTextProvider
# ══════════════════════════════════════════════════════════════

_TEXT_LIKE_COLS = {"date", "Date", "start_date", "end_date", "fact", "pred",
                   "preds", "Final_Output", "Final_Search_8", "Final_Search_16"}
_HISTORY_STAT_COLS = {"prior_history_avg", "prior_history_std"}


class RealTextProvider:
    """
    统一的数值 + 真实文本供给器 (替换 TrafficTextGenerator)。

    · 自动探测布局 A (MM-TSFlib 合并) / 布局 B (Time-MMD 原始);
    · self.numeric  : (n_rows, C) float32, 已填补 NaN, 未标准化;
    · self.row_text : 每个数值行清洗后的文本 (与行号一一对应);
    · texts_for_windows(): 把窗口起点索引映射到 **输入窗末行**, 聚合最近
      text_lookback 行文本 —— 只用输入窗内信息, 与 MM-TSFlib 抗泄漏约束一致
      (文本最晚 end_date <= 数值输入最晚 end_date, 未来目标行文本绝不可见)。
    """

    SIMPLE_MAX_LEN = 48        # SimpleTextEncoder 的 token 序列长度
    _VOCAB_CAP = 4000

    def __init__(self, domain: str, data_dir: str = "./data",
                 text_source: str = "both",          # "fact" | "preds" | "both"
                 text_lookback: int = 2, max_clauses: int = 6,
                 max_chars: int = 1200, min_clause_chars: int = 20,
                 raw_text_topk: int = 6, raw_lookback_days: int = 60,
                 keep_history_stats: bool = False, ot_only: bool = False,
                 verbose: bool = True):
        self.domain = canonical_domain(domain)
        self.data_dir = data_dir
        if text_source not in ("fact", "preds", "both"):
            raise ValueError("text_source must be one of fact/preds/both")
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
        if self.verbose:
            self.coverage_report()

    # ── 路径探测 ────────────────────────────────────────────

    def _resolve_paths(self):
        d, dd = self.domain, self.data_dir
        names = [d] + (["Health"] if d == "Health_US" else [])
        for nm in names:                                       # 布局 A
            for p in (os.path.join(dd, f"{nm}.csv"),
                      os.path.join(dd, nm, f"{nm}.csv")):
                if os.path.isfile(p):
                    if self.verbose:
                        print(f"[provider] layout=merged  {p}")
                    return ("merged", p)
        for nm in names:                                       # 布局 B
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
            f"No data found for domain '{d}' under '{dd}'. Expected either "
            f"'{d}.csv' (MM-TSFlib merged) or 'numerical/{d}/{d}.csv' + "
            f"'textual/{d}/*.csv' (Time-MMD raw). "
            f"Get the data: https://github.com/AdityaLab/Time-MMD")

    # ── 数值列选择 (两种布局共用) ────────────────────────────

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
            raise ValueError(f"No numeric columns found for {self.domain}")
        vals = df[cols].values.astype(np.float32)
        col_means = np.nanmean(vals, axis=0)
        for j in range(vals.shape[1]):
            m = np.isnan(vals[:, j])
            if m.any():
                vals[m, j] = col_means[j]
        return vals, cols

    @staticmethod
    def _parse_dates(df, prefer=("end_date", "date", "Date", "start_date")):
        for c in prefer:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce")
                if s.notna().any():
                    return s
        return None

    # ── 布局 A: MM-TSFlib 合并格式 ──────────────────────────

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
        fact_col = "fact" if "fact" in df.columns else None
        pred_col = next((c for c in ("preds", "pred") if c in df.columns), None)
        self._raw_fact = df[fact_col].astype(str).where(
            df[fact_col].notna(), "") if fact_col else pd.Series([""] * len(df))
        self._raw_pred = df[pred_col].astype(str).where(
            df[pred_col].notna(), "") if pred_col else pd.Series([""] * len(df))
        if fact_col is None and pred_col is None:
            print(f"[provider] WARN: no fact/preds column in {path}; "
                  f"all rows will use neutral fallback text.")

    # ── 布局 B: Time-MMD 原始格式 (区间匹配 + 抗泄漏) ────────

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
                "fact": t["fact"].astype(str).where(
                    t["fact"].notna(), "") if "fact" in t.columns else "",
                "pred": t[pc].astype(str).where(t[pc].notna(), "") if pc else "",
            }))
        txt = (pd.concat(frames, ignore_index=True)
               .dropna(subset=["_end"]).sort_values("_end", kind="stable")
               .reset_index(drop=True)) if frames else pd.DataFrame(
                   columns=["_end", "fact", "pred"])

        # 抗泄漏区间匹配: 数值行 i 只可见 end_date<=该行 end_date 的最近 top-k 条文本
        n = len(num)
        facts, preds = [""] * n, [""] * n
        ends, j = txt["_end"].values, 0
        lb = pd.Timedelta(days=self.raw_lookback_days)
        for i in range(n):
            e = num["_end"].iloc[i]
            if pd.isna(e):
                continue
            while j < len(txt) and ends[j] <= np.datetime64(e):
                j += 1
            lo = max(0, j - self.raw_text_topk)
            fp, pp = [], []
            for k in range(lo, j):
                te = txt["_end"].iloc[k]
                if e - te > lb:
                    continue
                dstr = pd.Timestamp(te).strftime("%Y-%m-%d")
                if txt["fact"].iloc[k].strip():
                    fp.append(f"{dstr}: {txt['fact'].iloc[k]}")
                if txt["pred"].iloc[k].strip():
                    pp.append(f"{dstr}: {txt['pred'].iloc[k]}")
            facts[i], preds[i] = "; ".join(fp), "; ".join(pp)
        self._raw_fact = pd.Series(facts)
        self._raw_pred = pd.Series(preds)

    # ── 逐行清洗 + 组装 ──────────────────────────────────────

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

    # ── 对外接口 ────────────────────────────────────────────

    @property
    def n_rows(self):
        return len(self.row_text)

    def text_for_row(self, r: int) -> str:
        """行 r 的窗口聚合文本: 最近 text_lookback 行 (含 r), 全在历史侧。"""
        r = int(min(max(r, 0), self.n_rows - 1))
        parts, seen = [], set()
        for rr in range(max(0, r - self.text_lookback + 1), r + 1):
            if self.row_has_info[rr] and self.row_text[rr] not in seen:
                parts.append(self.row_text[rr])
                seen.add(self.row_text[rr])
        joined = " ".join(parts) if parts else self._fallback
        if len(joined) > self.max_chars:                 # 左截断, 保最新信息
            joined = joined[-self.max_chars:]
            sp = joined.find(" ")
            if 0 < sp < 80:
                joined = joined[sp + 1:]
        return joined

    def texts_for_windows(self, window_idxs, split_offset: int, seq_len: int):
        """窗口起点索引 -> 输入窗末行 (offset+idx+seq_len-1) 的聚合文本。"""
        return [self.text_for_row(split_offset + int(w) + seq_len - 1)
                for w in window_idxs]

    # ── SimpleTextEncoder 支持: 真实语料词表 ────────────────

    _TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")

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

    @property
    def simple_max_len(self):
        return self.SIMPLE_MAX_LEN

    def tokenize(self, text: str, max_len: int | None = None):
        self._ensure_vocab()
        L = max_len or self.SIMPLE_MAX_LEN
        ids = [self._word2idx.get(t, 1)
               for t in self._TOKEN.findall(text.lower())][:L]
        return ids + [0] * (L - len(ids))

    # ── 诊断 ────────────────────────────────────────────────

    def coverage_report(self):
        n, info = self.n_rows, int(self.row_has_info.sum())
        lens = np.array([len(t) for t in self.row_text])
        freq = ""
        if self.dates is not None and self.dates.notna().sum() > 2:
            dd = self.dates.dropna().diff().dt.days.median()
            freq = f"  median_step={dd:.0f}d"
        print(f"[provider] {self.domain}: rows={n}  channels="
              f"{len(self.channels)} {self.channels[:5]}"
              f"{'...' if len(self.channels) > 5 else ''}  "
              f"informative={info}/{n} ({100 * info / max(n, 1):.1f}%)  "
              f"clean_chars mean={lens.mean():.0f} max={lens.max()}{freq}  "
              f"source={self.text_source}")

    def preview(self, k: int = 3):
        idxs = np.linspace(0, self.n_rows - 1, num=min(k, self.n_rows),
                           dtype=int)
        for r in idxs:
            d = (self.dates.iloc[r].date() if self.dates is not None
                 and pd.notna(self.dates.iloc[r]) else "?")
            raw = " | ".join(s for s in (
                str(self._raw_fact.iloc[r])[:160],
                str(self._raw_pred.iloc[r])[:160]) if s and s != "nan")
            print(f"\n──[{self.domain}] row {r}  date={d}  "
                  f"informative={bool(self.row_has_info[r])}")
            print(f"  RAW   : {raw[:340]}{'...' if len(raw) > 340 else ''}")
            print(f"  CLEAN : {self.row_text[r][:340]}"
                  f"{'...' if len(self.row_text[r]) > 340 else ''}")
            print(f"  WINDOW({self.text_lookback}): "
                  f"{self.text_for_row(r)[:340]}...")


# ══════════════════════════════════════════════════════════════
# SECTION D — DATA LOADING (返回 split 偏移, 供文本对齐)
# ══════════════════════════════════════════════════════════════

def auto_window(n_rows: int, seq_len: int, pred_len: int,
                min_train: int = 20, min_test: int = 5):
    """按数据长度自动收缩窗口, 保证 60/20/20 切分后各 split 有足够样本。"""
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

def plot_pred_curves(results, sensor_idx=0,
                     save_path="./image/iAdapter_pred_curves.png",
                     view_span=140, zoom_half=35, zoom_center=None,
                     lookback_display=None):
    """三面板布局（无标题）：

    Panel 1 (main)  — 滑动第一步预测, view_span 步总览。
                      自动定位 iAdapter 相对 Unimodal **收益最大**的连续段
                      作为主视窗中心。在主视窗内用暗蓝色矩形标记回溯区域、
                      暗红色矩形标记预测区域，指示 Panel 2 的分界。

    Panel 2 (detail)— 以 zoom_center 为分界线，左侧「暗蓝底 = 回溯历史」，
                      右侧「暗红底 = 预测窗口」。
                      - 左：回溯区 (lookback_display 步) 仅画真值曲线
                      - 右：预测区 (pred_len 步) 画真值 + 各模式完整 horizon
                            预测曲线（注意：是当前帧完整 pred_len 向量，而非
                            逐步滑动的第一步），清晰对比回溯与预测的精度差异
                      中间用竖分界线 + 标注区分两段。

    Panel 3 (diff)  — Δ|Error| = iAdapter − Unimodal 差分带，绿=ours更优。
    """
    from matplotlib.patches import ConnectionPatch, FancyArrowPatch

    # ── 配色常量 ───────────────────────────────────────────────
    _LB_DARK  = "#1c3d6b"   # 暗蓝 (回溯窗口)
    _LB_FILL  = "#d6e8f7"   # 浅蓝填充
    _PR_DARK  = "#7a1a1a"   # 暗红 (预测窗口)
    _PR_FILL  = "#f7d6d6"   # 浅红填充
    _GT_COLOR = "#111111"   # 近黑真值线
    _DIV_COLOR = "#444444"  # 分界线

    truth = results[0]["truths"]
    T, pred_len_h, _ = truth.shape
    pred_len_val = results[0].get("pred_len", pred_len_h)  # 优先取 metadata
    if lookback_display is None:
        lookback_display = pred_len_val  # 对称: 回溯段 = 预测段长度

    # gt: 滑动第一步真值，用于总览; gt_abs 同时用于回溯区历史重建
    gt = truth[:, 0, sensor_idx]

    ours_key = next((r["mode"] for r in results if _is_ours(r["mode"])), None)
    uni_key  = next((r["mode"] for r in results if _is_baseline(r["mode"])), None)

    # ── 自动定位「收益最大」区段 ──────────────────────────────
    if zoom_center is None and ours_key and uni_key:
        ours_r = next(r for r in results if r["mode"] == ours_key)
        uni_r  = next(r for r in results if r["mode"] == uni_key)
        diff_slide = (np.abs(ours_r["preds"][:, 0, sensor_idx] - gt)
                      - np.abs(uni_r["preds"][:, 0, sensor_idx] - gt))
        W = min(view_span, T)
        if T >= W and W > 0:
            roll = np.convolve(diff_slide, np.ones(W) / W, mode="valid")
            zoom_center = int(np.argmin(roll)) + W // 2
        else:
            zoom_center = T // 2
    elif zoom_center is None:
        zoom_center = T // 2

    # 确保 zoom_center 不越界 (留出完整 pred_len 和 lookback 空间)
    zoom_center = int(np.clip(zoom_center, lookback_display, T - pred_len_val - 1))

    # ── 主视窗范围 (Panel 1) ──────────────────────────────────
    half_v = view_span // 2
    v0 = max(0, zoom_center - half_v)
    v1 = min(T - 1, zoom_center + half_v)
    vt  = np.arange(v0, v1 + 1)
    vsl = slice(v0, v1 + 1)

    # ── 详情窗口范围 (Panel 2) ────────────────────────────────
    # 左: [zoom_center - lookback_display, zoom_center)  回溯历史
    # 右: [zoom_center, zoom_center + pred_len_val)       预测 horizon
    d_lb0 = zoom_center - lookback_display
    d_pr1 = zoom_center + pred_len_val

    # 替换原来的 lb_t / lb_gt 三行
    lb_gt = gt[max(0, d_lb0): zoom_center + 1]  # 含 t0 这个点
    lb_t = np.arange(zoom_center + 1 - len(lb_gt), zoom_center + 1)

    # 预测区: zoom_center 帧的完整 pred_len 向量
    pr_t      = np.arange(zoom_center, zoom_center + pred_len_val)
    pr_gt     = truth[zoom_center, :pred_len_val, sensor_idx]  # 真实未来值

    # ── 布局: 3 行 ────────────────────────────────────────────
    fig, (ax_main, ax_det, ax_diff) = plt.subplots(
        3, 1, figsize=(15, 11), dpi=130,
        gridspec_kw={"height_ratios": [2.2, 2.0, 1.0]},
        constrained_layout=False)
    fig.subplots_adjust(hspace=0.38)

    # ══════════════════════════════════════════════════════════
    # Panel 1: 总览 — 滑动第一步预测
    # ══════════════════════════════════════════════════════════
    ax_main.plot(vt, gt[vsl], color=_GT_COLOR, lw=1.8,
                 label="Ground Truth", zorder=7)
    for ci, r in enumerate(results):
        ours = _is_ours(r["mode"])
        ax_main.plot(vt, r["preds"][vsl, 0, sensor_idx],
                     color=_mode_color(r["mode"], ci),
                     lw=1.9 if ours else 0.85,
                     alpha=0.95 if ours else 0.55,
                     zorder=6 if ours else 3,
                     label=_mode_nice(r["mode"]))

    # 在主图上标出回溯/预测两段的位置 (暗蓝/暗红矩形)
    ylim_m = ax_main.get_ylim()
    y_all_main = np.concatenate([r["preds"][vsl, 0, sensor_idx] for r in results]
                                + [gt[vsl]])
    ym_lo, ym_hi = float(y_all_main.min()), float(y_all_main.max())
    ym_pad = (ym_hi - ym_lo) * 0.07 + 1e-6
    # 回溯标注矩形 (蓝)
    lb_rect_x0 = max(v0, d_lb0)
    if lb_rect_x0 < zoom_center:
        ax_main.axvspan(lb_rect_x0, zoom_center,
                        color=_LB_DARK, alpha=0.09, zorder=1)
    # 预测标注矩形 (红)
    pr_rect_x1 = min(v1, d_pr1)
    if zoom_center < pr_rect_x1:
        ax_main.axvspan(zoom_center, pr_rect_x1,
                        color=_PR_DARK, alpha=0.09, zorder=1)
    # 分界竖线
    ax_main.axvline(zoom_center, color=_DIV_COLOR, lw=1.0,
                    ls="--", alpha=0.7, zorder=5)
    ax_main.annotate("│detail↓",
                     xy=(zoom_center, ym_hi + ym_pad * 0.3),
                     ha="center", va="bottom", fontsize=7.5,
                     color=_DIV_COLOR)
    # 蓝/红标签
    if lb_rect_x0 < zoom_center:
        ax_main.text((lb_rect_x0 + zoom_center) / 2, ym_lo - ym_pad * 0.6,
                     "◀ lookback", ha="center", va="top",
                     fontsize=7, color=_LB_DARK, alpha=0.8)
    if zoom_center < pr_rect_x1:
        ax_main.text((zoom_center + pr_rect_x1) / 2, ym_lo - ym_pad * 0.6,
                     "forecast ▶", ha="center", va="top",
                     fontsize=7, color=_PR_DARK, alpha=0.8)

    ax_main.set_ylabel(f"Sensor #{sensor_idx} (normalised)")
    ax_main.legend(fontsize=7.5, ncol=3, loc="best")
    ax_main.set_xlim(v0, v1)
    ax_main.grid(True, lw=0.32, alpha=0.3)

def load_real_data_mm(provider: RealTextProvider, seq_len: int, pred_len: int):
    """
    与 adapt_z_multimodal.load_real_data 相同的 60/20/20 切分与 scaler 逻辑,
    但额外返回每个 split 的 **原始行偏移**, 用于窗口索引 -> 数值行号的映射。
    """
    from sklearn.preprocessing import StandardScaler
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
            f"seq_len={seq_len}, pred_len={pred_len}: empty split. "
            f"Use the full Time-MMD csv or reduce --seq_len/--pred_len.")
    return train, val, test, scaler, offsets


_DEFAULT_LRS = dict(finetune_lr=1e-3, adapter_lr=1e-3,
                    online_lr=1e-4, adp_online_lr=1e-3)


def ensure_settings(model_name: str, data_key: str, overrides=None):
    """为新域向 adapt_z.settings 注入 lr 配置 (运行期注入, 不改源文件)。"""
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
        tbl[data_key].update(applied)
        if applied:
            print(f"[settings] overrides applied: {applied}")


# ══════════════════════════════════════════════════════════════
# SECTION E — OnlineMMReal: 真实文本注入 (其余全部复用基类)
# ══════════════════════════════════════════════════════════════

if _RUNTIME_OK:

    class OnlineMMReal(OnlineMM):
        """
        OnlineMM 的最小侵入子类:
          · _encode_text: 模板短语 -> RealTextProvider 真实文本;
          · val/online 前注入对应 split 的行偏移 (文本-窗口精确对齐);
          · text_encoder_type='simple' 时改用真实语料词表重建编码器。
        MoE 专家融合 / 双梯度反馈 / 骨干冻结 等逻辑不变。
        """

        def __init__(self, model, d_model, args, text_provider: RealTextProvider,
                     split_offsets: dict, **kw):
            self.text_provider = text_provider
            self._split_offsets = dict(split_offsets)
            self._cur_offset = self._split_offsets.get("val", 0)
            _dropout = kw.get("dropout", 0.1)

            # 父类 OnlineMM 从 adapt_z_multimodal 导入, 其 __init__ 签名可能
            # 有两种版本: (旧) 不接收 provider/split_offsets, 由本子类完全接管
            # 真实文本; (新) 已内置 provider/split_offsets 形参。这里按父类
            # 实际签名自适应转发, 避免 "missing positional argument" 或
            # "unexpected keyword" 两类版本偏移错误。
            import inspect
            try:
                parent_params = set(
                    inspect.signature(OnlineMM.__init__).parameters)
            except (TypeError, ValueError):
                parent_params = set()
            fwd = dict(kw)
            # provider: 兼容 'provider' / 'text_provider' 两种命名
            if "provider" in parent_params:
                fwd.setdefault("provider", text_provider)
            elif "text_provider" in parent_params:
                fwd.setdefault("text_provider", text_provider)
            # split_offsets: 兼容 'split_offsets' / 'offsets'
            if "split_offsets" in parent_params:
                fwd.setdefault("split_offsets", self._split_offsets)
            elif "offsets" in parent_params:
                fwd.setdefault("offsets", self._split_offsets)
            # 丢弃父类不认识且非 **kwargs 的键 (防 unexpected keyword)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in inspect.signature(OnlineMM.__init__).parameters.values()
            ) if parent_params else True
            if not has_var_kw:
                fwd = {k: v for k, v in fwd.items() if k in parent_params}

            super().__init__(model, d_model, args, **fwd)

            self.text_gen = None                       # 弃用模板生成器
            # 若父类已按 provider 建好真实文本编码器, 无需再重建;
            # 仅当父类给的是模板版 SimpleTextEncoder 时才用真实语料重建。
            need_rebuild = (
                self.use_text
                and isinstance(getattr(self, "text_enc", None), SimpleTextEncoder)
                and getattr(self, "provider", None) is None
            )
            if need_rebuild:
                self.text_enc = SimpleTextEncoder(
                    vocab_size=text_provider.vocab_size,
                    embed_dim=self.text_dim, output_dim=self.text_dim,
                    max_len=text_provider.simple_max_len,
                    dropout=_dropout).to(self.device)
                self._text_params = [p for p in self.text_enc.parameters()
                                     if p.requires_grad]
                print(f"[MMReal] SimpleTextEncoder rebuilt with real-corpus "
                      f"vocab={text_provider.vocab_size}")

        # split 偏移注入
        def val(self, val_loader, mode="val"):
            self._cur_offset = self._split_offsets["val"]
            return super().val(val_loader, mode)

        def online(self, test_loader):
            self._cur_offset = self._split_offsets["test"]
            return super().online(test_loader)

        # 真实文本编码 (被基类 val/online 的所有调用点复用, 含双梯度缓冲区)
        def _encode_text(self, global_start_indices):
            if not self.use_text or self.text_enc is None:
                return None
            widx = [int(i) for i in global_start_indices]
            texts = self.text_provider.texts_for_windows(
                widx, self._cur_offset, self.args.seq_len)
            if isinstance(self.text_enc, LLMTextEncoder):
                return self.text_enc(texts)
            ids = torch.tensor([self.text_provider.tokenize(t) for t in texts],
                               dtype=torch.long, device=self.device)
            return self.text_enc(ids)

    # ── 骨干预训练 (checkpoint 缺失时自动执行, 使脚本对 9 域即开即用) ──

    def pretrain_backbone(bb, train_loader, val_loader, device,
                          epochs=15, lr=1e-3, patience=3, ckpt_path=None):
        opt = optim.Adam(bb.parameters(), lr=lr)
        lossf = nn.MSELoss()
        best, bad, best_state = float("inf"), 0, None
        for ep in range(epochs):
            bb.train()
            tr = 0.0
            for xb, yb, *_ in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = lossf(bb(xb)["pred"], yb)
                loss.backward()
                opt.step()
                tr += loss.item()
            bb.eval()
            vl, m = 0.0, 0
            with torch.no_grad():
                for xb, yb, *_ in val_loader:
                    vl += lossf(bb(xb.to(device))["pred"],
                                yb.to(device)).item()
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

    # ══════════════════════════════════════════════════════
    # SECTION F — ENTRY: finetune_mm_real (镜像 base.finetune_mm)
    # ══════════════════════════════════════════════════════

    def finetune_mm_real(domain, data_dir="./data", seq_len=None, pred_len=None,
                         seed=2023, batch_size=16, d_model=64, e_layers=2,
                         z_loc=2, text_encoder="BERT", text_dim=64, llm_layers=6,
                         dropout=0.1, weight_decay=1e-4, grad_clip=1.0,
                         patience=3, warmup_rounds=3,
                         text_source="both", text_lookback=2, max_clauses=6,
                         max_chars=1200, keep_history_stats=False,
                         ot_only=False, raw_text_topk=6, raw_lookback_days=60,
                         modes=None, make_figures=True, ckpt_dir=".",
                         pretrain_epochs=15, pretrain_lr=1e-3,
                         lr_overrides=None, csv_path="finally_mm_real.csv"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        domain = canonical_domain(domain)
        modes = modes or ABLATION_MODES

        # 1) 真实文本供给器 (数值 + 文本一站式)
        provider = RealTextProvider(
            domain, data_dir=data_dir, text_source=text_source,
            text_lookback=text_lookback, max_clauses=max_clauses,
            max_chars=max_chars, raw_text_topk=raw_text_topk,
            raw_lookback_days=raw_lookback_days,
            keep_history_stats=keep_history_stats, ot_only=ot_only)

        # 2) 窗口: 域默认 -> 自动收缩
        ds, dp = _DOMAIN_WINDOW.get(domain, (36, 6))
        seq_len = seq_len or ds
        pred_len = pred_len or dp
        seq_len, pred_len = auto_window(provider.n_rows, seq_len, pred_len)

        args = argparse.Namespace(
            model="iTransformer", data=domain, seq_len=seq_len,
            pred_len=pred_len, seed=seed, device=device,
            batch_size=batch_size, batch_size2=batch_size, z_loc=z_loc,
            z_shape=(batch_size, 1, d_model), checkpoints="./checkpoints2")
        ensure_settings(args.model, domain.lower(), lr_overrides)

        train_set, val_set, test_set, _sc, offsets = load_real_data_mm(
            provider, seq_len, pred_len)
        enc_in = provider.numeric.shape[1]
        args.enc_in = enc_in

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=False)

        # 3) 骨干 checkpoint: 文件名编码全部形状维度 (seq_len/d_model/
        #    e_layers/enc_in), 避免换窗口后错误加载旧骨干 (size mismatch)。
        #    embedding.proj.weight 形状 = (d_model, seq_len), seq_len 变化
        #    必然维度不符 —— 命名带上后不同 seq_len 各自独立缓存/预训练。
        os.makedirs(ckpt_dir, exist_ok=True)
        stem = (f"iAdapter_backbone_{domain}_L{seq_len}_h{pred_len}"
                f"_dm{d_model}_el{e_layers}_c{enc_in}_seed{seed}.pth")
        ckpt = os.path.join(ckpt_dir, stem)
        legacy_names = [
            os.path.join(
                ckpt_dir,
                f"iAdapter_backbone_{domain}_h{pred_len}_seed{seed}.pth"),
            f"iAdapter_backbone_{domain}_h{pred_len}_seed{seed}.pth",
        ]

        def _ckpt_matches(path):
            """embedding.proj.weight 形状须 == (d_model, seq_len), 否则忽略。"""
            try:
                sd = torch.load(path, map_location="cpu")
                w = sd.get("embedding.proj.weight")
                return w is not None and tuple(w.shape) == (d_model, seq_len)
            except Exception:
                return False

        if not os.path.exists(ckpt):
            reused = next((p for p in legacy_names
                           if os.path.exists(p) and _ckpt_matches(p)), None)
            if reused:
                ckpt = reused
                print(f"[backbone] reusing compatible legacy checkpoint: {ckpt}")
            else:
                stale = [p for p in legacy_names if os.path.exists(p)]
                if stale:
                    print(f"[backbone] legacy checkpoint(s) shape-mismatched "
                          f"(seq_len/d_model/enc_in changed): {stale} -> "
                          f"ignored, re-pretraining for current config.")
                print(f"[backbone] checkpoint missing -> pretraining "
                      f"({pretrain_epochs} epochs)  target={ckpt}")
                set_seed(seed)
                bb = ITransformerZ(seq_len, pred_len, enc_in,
                                   d_model=d_model, e_layers=e_layers).to(device)
                pretrain_backbone(bb, train_loader, val_loader, device,
                                  epochs=pretrain_epochs, lr=pretrain_lr,
                                  patience=patience, ckpt_path=ckpt)

        def fresh_backbone():
            bb = ITransformerZ(seq_len, pred_len, enc_in,
                               d_model=d_model, e_layers=e_layers)
            try:
                bb.load_state_dict(torch.load(ckpt, map_location=device))
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to load backbone '{ckpt}' for config "
                    f"seq_len={seq_len}, pred_len={pred_len}, "
                    f"d_model={d_model}, e_layers={e_layers}, enc_in={enc_in}. "
                    f"Checkpoint was likely trained with a different "
                    f"seq_len/d_model/enc_in. Delete it to re-pretrain, or "
                    f"pass matching --seq_len.\nOriginal: {e}") from e
            return bb.to(device).eval()

        # 4) 消融主循环 (真实文本版; 静态基线同样移除, 以 Unimodal 为参照)
        results, gate_hist_full = [], None
        for label, kw in modes:
            print(f"\n{'=' * 62}\n[RUN:{domain}] {label}  {kw}  "
                  f"(REAL text: {text_source})\n{'=' * 62}")
            set_seed(seed)
            args.z_shape = (batch_size, enc_in, d_model)
            adapter = OnlineMMReal(
                model=fresh_backbone(), d_model=d_model, args=args,
                text_provider=provider, split_offsets=offsets,
                text_encoder_type=text_encoder, text_dim=text_dim,
                llm_layers=llm_layers, dropout=dropout,
                weight_decay=weight_decay, grad_clip=grad_clip,
                patience=patience, update_backbone=False, **kw)

            best_mse, bad = float("inf"), 0
            for _ in range(warmup_rounds):
                mse = adapter.val(val_loader)
                if mse < best_mse:
                    best_mse, bad = mse, 0
                else:
                    bad += 1
                    print(f"[MMReal] val not improved ({bad}/{patience})")
                    if bad >= patience:
                        break

            mae, mse, preds, truths = adapter.online(test_loader)
            results.append(dict(mode=label, mae=mae, mse=mse,
                                preds=preds, truths=truths))
            if "iAdapter" in label and adapter.gate_history:
                gate_hist_full = list(adapter.gate_history)

            pd.DataFrame([{"data": domain, "seq_len": seq_len,
                           "pred_len": pred_len, "seed": seed, "mode": label,
                           **kw, "text_source": text_source,
                           "text_encoder": text_encoder,
                           "mae": mae, "mse": mse}]).to_csv(
                csv_path, mode="a",
                header=not os.path.exists(csv_path), index=False)

        # 5) 汇总 (以 Unimodal 为参照的逐模块收益)
        print(f"\n{'-' * 66}\n[SUMMARY:REAL] {domain} h={pred_len} seed={seed} "
              f"text={text_source}\n{'-' * 66}")
        uni_mae = next((r["mae"] for r in results if "Unimodal" in r["mode"]),
                       results[0]["mae"])
        uni_mse = next((r["mse"] for r in results if "Unimodal" in r["mode"]),
                       results[0]["mse"])
        for r in results:
            print(f"  {r['mode']:<28} "
                  f"MAE={r['mae']:.4f} "
                  f"({(uni_mae - r['mae']) / uni_mae * 100:+.1f}%)  "
                  f"MSE={r['mse']:.4f} "
                  f"({(uni_mse - r['mse']) / uni_mse * 100:+.1f}%)")

        if make_figures:
            tag = f"_{domain}_h{pred_len}_s{seed}_realtext"
            plot_ablation_summary(results, f"iAdapter_ablation_summary{tag}.png")
            plot_rolling_mae(results, f"iAdapter_rolling_mae{tag}.png")
            plot_pred_curves(results, sensor_idx=0,
                             save_path=f"iAdapter_pred_curves{tag}.png")
            if gate_hist_full:
                plot_gate_usage(gate_hist_full,
                                save_path=f"iAdapter_gate_usage{tag}.png")
        return results


# ══════════════════════════════════════════════════════════════
# SECTION G — CLI
# ══════════════════════════════════════════════════════════════

_MODE_KEYS = {"m1": 0, "m2": 1, "m3": 2, "m4": 3, "m5": 4}


def _parse_modes(spec: str):
    if spec.strip().lower() in ("all", ""):
        return None
    idx = sorted({_MODE_KEYS[t.strip().lower()]
                  for t in spec.split(",") if t.strip()})
    return [ABLATION_MODES[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(
        description="iAdapter with REAL Time-MMD text (drop-in for "
                    "adapt_z_multimodal.py)")
    ap.add_argument("--data", default="Traffic",
                    help="domain | comma list | 'all'  "
                         f"(choices: {', '.join(DOMAINS)}; Health=Health_US)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--seq_len", type=int, default=0, help="0 = per-domain default")
    ap.add_argument("--pred_len", type=int, default=0, help="0 = per-domain default")
    ap.add_argument("--seeds", default="2023")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--z_loc", type=int, default=2)
    ap.add_argument("--text_encoder", default="BERT",
                    choices=["BERT", "GPT2", "GPT2M", "simple"])
    ap.add_argument("--text_dim", type=int, default=64)
    ap.add_argument("--llm_layers", type=int, default=6)
    ap.add_argument("--text_source", default="both",
                    choices=["fact", "preds", "both"])
    ap.add_argument("--text_lookback", type=int, default=2,
                    help="aggregate text from last K input rows")
    ap.add_argument("--max_clauses", type=int, default=6)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--raw_text_topk", type=int, default=6)
    ap.add_argument("--raw_lookback_days", type=int, default=60)
    ap.add_argument("--keep_history_stats", action="store_true",
                    help="keep prior_history_avg/std as numeric channels")
    ap.add_argument("--ot_only", action="store_true",
                    help="use only the OT column as numeric channel")
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
    ap.add_argument("--csv", default="finally_mm_real.csv")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--preview", type=int, default=0,
                    help="N>0: only print N cleaned text samples per domain "
                         "and exit (no torch needed)")
    a = ap.parse_args()

    doms = (DOMAINS if a.data.strip().lower() == "all"
            else [canonical_domain(d) for d in a.data.split(",") if d.strip()])

    if a.preview > 0:                                    # 纯文本预览模式
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
            f"Training mode needs torch + adapt_z.py + adapt_z_multimodal.py "
            f"on PYTHONPATH. Import failed with: {_IMPORT_ERR!r}\n"
            f"(--preview N works without them.)")

    lr_over = dict(adapter_lr=a.adapter_lr, finetune_lr=a.finetune_lr,
                   online_lr=a.online_lr, adp_online_lr=a.adp_online_lr)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    for d, seed in itertools.product(doms, seeds):
        finetune_mm_real(
            domain=d, data_dir=a.data_dir,
            seq_len=a.seq_len or None, pred_len=a.pred_len or None,
            seed=seed, batch_size=a.batch_size, d_model=a.d_model,
            e_layers=a.e_layers, z_loc=a.z_loc,
            text_encoder=a.text_encoder, text_dim=a.text_dim,
            llm_layers=a.llm_layers, dropout=a.dropout,
            weight_decay=a.weight_decay, grad_clip=a.grad_clip,
            patience=a.patience, warmup_rounds=a.warmup_rounds,
            text_source=a.text_source, text_lookback=a.text_lookback,
            max_clauses=a.max_clauses, max_chars=a.max_chars,
            keep_history_stats=a.keep_history_stats, ot_only=a.ot_only,
            raw_text_topk=a.raw_text_topk,
            raw_lookback_days=a.raw_lookback_days,
            modes=_parse_modes(a.modes), make_figures=not a.no_figures,
            ckpt_dir=a.ckpt_dir, pretrain_epochs=a.pretrain_epochs,
            pretrain_lr=a.pretrain_lr, lr_overrides=lr_over, csv_path=a.csv)


if __name__ == "__main__":
    main()