"""
data_provider_realtext.py  (TSlib / MM-TSFlib style, self-contained)
====================================================================
把 Time-MMD 的 **真实文本** 接进 TSlib 式训练框架 (即你上传的
data_provider/uea.py 所在的代码栈; uea.py 源自 gzerveas/mvts_transformer,
被 TSlib 收编, MM-TSFlib 基于 TSlib 扩展)。

与 adapt_z_multimodal_realtext.py 的分工:
  · adapt_z_multimodal_realtext.py —— 面向 adapt_z / iAdapter 在线适应管线;
  · 本文件 —— 面向 TSlib / MM-TSFlib 式 data_provider 体系:
        Dataset_TimeMMD.__getitem__ 返回
            seq_x, seq_y, seq_x_mark, seq_y_mark, text
        前四项与 TSlib 的 Dataset_Custom 完全同约定 (含 label_len 解码器
        重叠段), 第五项是清洗后的真实文本 (str / token ids / 关闭)。
  两个文件的清洗规则保持一致, 修改时请同步。

数据域 (Time-MMD, NeurIPS 2024, github.com/AdityaLab/Time-MMD):
    Agriculture, Climate, Economy, Energy, Environment,
    Health (=Health_US), Security, SocialGood, Traffic
两种布局自动探测:
  A) MM-TSFlib 预处理合并格式:  <root>/<Domain>.csv
       (OT, Date, start_date, date, end_date,
        prior_history_avg, prior_history_std, fact, preds)
  B) Time-MMD 官方原始格式:     <root>/numerical/<D>/<D>.csv
                               + <root>/textual/<D>/<D>_report.csv / _search.csv

抗泄漏 (与 MM-TSFlib 论文 App.J 约束一致):
  样本文本只取 **输入窗内** 行 (末行向前 text_lookback 行);
  原始布局下匹配时强制 text.end_date <= 数值行.end_date;
  预测目标区间 (未来行) 的文本永不可见。

接入 TSlib (两行):
    # data_provider/data_factory.py
    from data_provider.data_provider_realtext import Dataset_TimeMMD
    data_dict['TimeMMD'] = Dataset_TimeMMD
    # 运行: --data TimeMMD --root_path ./data --data_path Health \
    #       --features S --target OT --freq w --seq_len 96 --pred_len 12
    # exp 循环解包: for x, y, xm, ym, text in loader: ...

独立使用:
    from data_provider_realtext import Dataset_TimeMMD, collate_timemmd
    ds = Dataset_TimeMMD(root_path='./data', data_path='Traffic', flag='train',
                         size=[36, 18, 6], features='S', target='OT',
                         freq='m', return_text='str')
    dl = DataLoader(ds, batch_size=16, collate_fn=collate_timemmd)

纯文本预览 (无需 torch):
    python data_provider_realtext.py --preview 3 --data Traffic --root ./data

顺带指出你上传的 uea.py 里两处沿袭自上游的隐患 (本文件不依赖它们):
  1) padding_mask(): `lengths.max_val()` —— torch.Tensor 只有 .max();
     max_len=None 时会 AttributeError (collate_fn 总是传 max_len, 故从未触发);
  2) collate_fn() docstring 承诺返回 target_masks, 实际只返回
     (X, targets, padding_masks) —— 文档是补全任务变体的残留。
"""

from __future__ import annotations

import os
import re
import argparse
from collections import Counter

import numpy as np
import pandas as pd

# ── torch 守护: --preview 与全部清洗/对齐逻辑在无 torch 环境同样可用 ──
_TORCH_OK, _TORCH_ERR = True, None
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception as _e:                                    # pragma: no cover
    _TORCH_OK, _TORCH_ERR = False, _e

    class Dataset:                                         # 占位, 仅供定义类
        pass

# 复用你上传的 uea.py 工具 (可用则用, 不可用则本地等价实现)
try:
    from data_provider.uea import interpolate_missing      # TSlib 布局
except Exception:
    try:
        from uea import interpolate_missing                # 同目录
    except Exception:
        def interpolate_missing(y: pd.Series) -> pd.Series:
            if y.isna().any():
                y = y.interpolate(method="linear", limit_direction="both")
            return y


# ══════════════════════════════════════════════════════════════
# SECTION A — DOMAIN REGISTRY (Time-MMD 9 域)
# ══════════════════════════════════════════════════════════════

DOMAINS = ["Agriculture", "Climate", "Economy", "Energy", "Environment",
           "Health_US", "Security", "SocialGood", "Traffic"]

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
}

# 启发式 (seq_len, label_len, pred_len) —— 按域采样频率给默认, CLI/args 可覆盖
SUGGESTED_SIZE = {
    "Agriculture": (36, 18, 6),   # monthly
    "Climate":     (36, 18, 6),   # monthly
    "Economy":     (36, 18, 6),   # monthly
    "Energy":      (48, 24, 8),   # weekly
    "Environment": (96, 48, 12),  # daily
    "Health_US":   (96, 48, 12),  # weekly
    "Security":    (36, 18, 6),   # monthly
    "SocialGood":  (36, 18, 6),   # monthly
    "Traffic":     (36, 18, 6),   # monthly
}


def canonical_domain(name: str) -> str:
    key = os.path.splitext(str(name).strip())[0].lower()   # 允许传 'Health.csv'
    if key in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[key]
    raise ValueError(f"Unknown domain '{name}'. Choose from {DOMAINS} "
                     f"(alias: Health -> Health_US).")


# ══════════════════════════════════════════════════════════════
# SECTION B — TEXT CLEANING (与 adapt_z_multimodal_realtext.py 同规则)
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
    for pat in _DROP_PATTERNS:
        if pat.search(s):
            return ""
    return s


def split_date_segments(text: str):
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
    """清洗一个 fact/preds 单元格 -> (clean_text, n_kept, n_total); 最新段优先。"""
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
# SECTION C — RealTextProvider (数值 + 真实文本, 两种布局)
# ══════════════════════════════════════════════════════════════

_TEXT_LIKE_COLS = {"date", "Date", "start_date", "end_date", "fact", "pred",
                   "preds", "Final_Output", "Final_Search_8", "Final_Search_16"}
_HISTORY_STAT_COLS = {"prior_history_avg", "prior_history_std"}


class RealTextProvider:
    """按数值行序供给清洗后的真实文本; 布局自动探测; 抗泄漏对齐见 docstring。"""

    SIMPLE_MAX_LEN = 48
    _VOCAB_CAP = 4000
    _TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")

    def __init__(self, domain: str, root_path: str = "./data",
                 text_source: str = "both", text_lookback: int = 2,
                 max_clauses: int = 6, max_chars: int = 1200,
                 min_clause_chars: int = 20, raw_text_topk: int = 6,
                 raw_lookback_days: int = 60,
                 keep_history_stats: bool = False, verbose: bool = True):
        self.domain = canonical_domain(domain)
        self.root_path = root_path
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
        if verbose:
            self.coverage_report()

    # ── 路径探测 ──
    def _resolve_paths(self):
        d, dd = self.domain, self.root_path
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

    # ── 数值列 ──
    def _pick_numeric(self, df: pd.DataFrame):
        drop = set(_TEXT_LIKE_COLS)
        if not self.keep_history_stats:
            drop |= _HISTORY_STAT_COLS
        cols = [c for c in df.columns
                if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            raise ValueError(f"No numeric columns for {self.domain}")
        sub = df[cols].apply(interpolate_missing)          # 复用 uea.py 逻辑
        vals = sub.values.astype(np.float32)
        if np.isnan(vals).any():                           # 全列 NaN 兜底
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

    # ── 布局 A ──
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

    # ── 布局 B (区间匹配 + 抗泄漏: text.end <= 数值行.end) ──
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

    # ── 对外 ──
    @property
    def n_rows(self):
        return len(self.row_text)

    def text_for_row(self, r: int) -> str:
        """全局行 r 的窗口聚合文本 (r 及其前 text_lookback-1 行, 全在历史侧)。"""
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

    # ── 轻量词表 (return_text='ids' 用) ──
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
            freq = f"  median_step={self.dates.dropna().diff().dt.days.median():.0f}d"
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
# SECTION D — 纯函数: 切分边界 / 窗口切片 / 时间标记
#   (与 torch 解耦, 便于独立单元测试)
# ══════════════════════════════════════════════════════════════

def compute_borders(n_rows: int, seq_len: int,
                    ratios=(0.6, 0.2, 0.2)):
    """
    TSlib 式三段边界 (val/test 向前重叠 seq_len 以保证窗口连续):
        borders[flag] = (border1, border2), split 数据 = data[border1:border2]
    默认 60/20/20, 与 adapt_z 管线一致; 传 ratios=(0.7,0.1,0.2) 可切 TSlib 惯例。
    """
    t_end = int(n_rows * ratios[0])
    v_end = int(n_rows * (ratios[0] + ratios[1]))
    return {
        "train": (0, t_end),
        "val": (max(0, t_end - seq_len), v_end),
        "test": (max(0, v_end - seq_len), n_rows),
    }


def slice_bounds(index: int, seq_len: int, label_len: int, pred_len: int):
    """TSlib 取样约定: 返回 (s_begin, s_end, r_begin, r_end)。"""
    s_begin = index
    s_end = s_begin + seq_len
    r_begin = s_end - label_len
    r_end = s_end + pred_len
    return s_begin, s_end, r_begin, r_end


def make_time_marks(dates, timeenc: int = 0, freq: str = "m") -> np.ndarray:
    """
    从日期序列生成时间标记 (简化版, 不依赖 TSlib 的 utils.timefeatures):
      timeenc=0 -> 整型 [month, day, weekday] (+hour 若 freq 为 h/t/b);
      timeenc=1 -> 归一化浮点, 同列布局, 值域约 [-0.5, 0.5]。
    dates 为 None 或全 NaT 时返回零矩阵并保持形状约定。
    """
    hourly = str(freq).lower()[:1] in ("h", "t", "b", "s")
    cols = 4 if hourly else 3
    if dates is None:
        return np.zeros((0, cols), dtype=np.float32)
    d = pd.to_datetime(pd.Series(dates), errors="coerce")
    n = len(d)
    out = np.zeros((n, cols), dtype=np.float32)
    ok = d.notna().values
    if ok.any():
        month = d.dt.month.fillna(1).values
        day = d.dt.day.fillna(1).values
        dow = d.dt.dayofweek.fillna(0).values
        if timeenc == 0:
            out[:, 0], out[:, 1], out[:, 2] = month, day, dow
            if hourly:
                out[:, 3] = d.dt.hour.fillna(0).values
        else:
            out[:, 0] = (month - 1) / 11.0 - 0.5
            out[:, 1] = (day - 1) / 30.0 - 0.5
            out[:, 2] = dow / 6.0 - 0.5
            if hourly:
                out[:, 3] = d.dt.hour.fillna(0).values / 23.0 - 0.5
        out[~ok] = 0.0
    return out


# ══════════════════════════════════════════════════════════════
# SECTION E — Dataset_TimeMMD (TSlib / MM-TSFlib 约定)
# ══════════════════════════════════════════════════════════════

class Dataset_TimeMMD(Dataset):
    """
    __getitem__ 返回:
        seq_x      (seq_len, C)                float32
        seq_y      (label_len+pred_len, C)     float32   ← 含解码器重叠段
        seq_x_mark (seq_len, M)                float32
        seq_y_mark (label_len+pred_len, M)     float32
        text       str | LongTensor(ids) | 省略 (return_text='none')

    features: 'S' 仅 target 列; 'M'/'MS' 全部数值列 (target 排最后, TSlib 惯例)。
    scale:    StandardScaler 只在 train 边界内拟合 (TSlib 惯例)。
    文本对齐: text = provider.text_for_row(border1 + s_end - 1)
              —— 输入窗最后一行的全局行号, 目标区间行永不可见 (抗泄漏)。
    """

    def __init__(self, root_path, flag="train", size=None, features="S",
                 data_path="Traffic", target="OT", scale=True,
                 timeenc=0, freq="m",
                 text_source="both", text_lookback=2, max_clauses=6,
                 max_chars=1200, raw_text_topk=6, raw_lookback_days=60,
                 keep_history_stats=False, return_text="str",
                 split_ratios=(0.6, 0.2, 0.2), provider=None,
                 verbose=True, **_ignored):                 # 吞掉 TSlib 额外参数
        if not _TORCH_OK:
            raise RuntimeError(f"Dataset_TimeMMD needs torch: {_TORCH_ERR!r}")
        assert flag in ("train", "val", "test", "vali")
        self.flag = "val" if flag == "vali" else flag
        self.domain = canonical_domain(data_path)
        if size is None:
            size = SUGGESTED_SIZE[self.domain]
            if verbose:
                print(f"[dataset] size=None -> suggested {size} for {self.domain}")
        self.seq_len, self.label_len, self.pred_len = size
        assert 0 <= self.label_len <= self.seq_len
        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq
        assert return_text in ("str", "ids", "none")
        self.return_text = return_text

        # provider 可跨 flag 共享 (train/val/test 传同一个, 免三次加载)
        self.provider = provider or RealTextProvider(
            self.domain, root_path=root_path, text_source=text_source,
            text_lookback=text_lookback, max_clauses=max_clauses,
            max_chars=max_chars, raw_text_topk=raw_text_topk,
            raw_lookback_days=raw_lookback_days,
            keep_history_stats=keep_history_stats, verbose=verbose)

        self.__read_data__(split_ratios, verbose)

    def __read_data__(self, ratios, verbose):
        p = self.provider
        cols = list(p.channels)
        if self.features == "S":
            if self.target not in cols:
                raise ValueError(f"target '{self.target}' not in numeric "
                                 f"columns {cols}")
            data = p.numeric[:, [cols.index(self.target)]]
        else:                                              # 'M' / 'MS'
            if self.target in cols:                        # target 放最后
                order = [c for c in cols if c != self.target] + [self.target]
                data = p.numeric[:, [cols.index(c) for c in order]]
            else:
                data = p.numeric
        n = len(data)
        borders = compute_borders(n, self.seq_len, ratios)
        b1, b2 = borders[self.flag]
        self.border1 = b1                                  # 文本对齐用全局偏移

        if self.scale:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            tr1, tr2 = borders["train"]
            self.scaler.fit(data[tr1:tr2])
            data = self.scaler.transform(data)
        else:
            self.scaler = None

        self.data_x = data[b1:b2].astype(np.float32)
        self.data_y = self.data_x                          # TSlib 同源约定
        marks = make_time_marks(p.dates, self.timeenc, self.freq)
        self.data_stamp = (marks[b1:b2] if len(marks) == n
                           else np.zeros((b2 - b1, marks.shape[1] or 3),
                                         dtype=np.float32))
        if verbose:
            print(f"[dataset] {self.domain}/{self.flag}  rows[{b1}:{b2}]  "
                  f"samples={len(self)}  C={self.data_x.shape[1]}  "
                  f"marks={self.data_stamp.shape[1]}  text={self.return_text}")
        if len(self) <= 0:
            raise ValueError(
                f"'{self.domain}' too short for seq_len={self.seq_len}, "
                f"pred_len={self.pred_len} at flag={self.flag} "
                f"(rows={b2 - b1}). Reduce size or use full Time-MMD csv.")

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s0, s1, r0, r1 = slice_bounds(index, self.seq_len,
                                      self.label_len, self.pred_len)
        seq_x = torch.from_numpy(self.data_x[s0:s1])
        seq_y = torch.from_numpy(self.data_y[r0:r1])
        seq_x_mark = torch.from_numpy(self.data_stamp[s0:s1])
        seq_y_mark = torch.from_numpy(self.data_stamp[r0:r1])
        if self.return_text == "none":
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        text = self.provider.text_for_row(self.border1 + s1 - 1)  # 抗泄漏
        if self.return_text == "ids":
            text = torch.tensor(self.provider.tokenize(text), dtype=torch.long)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, text

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) if self.scaler else data


def collate_timemmd(batch):
    """
    五元组 collate: 张量走 stack, 文本保持 list[str] (或 stack ids)。
    default_collate 其实也能处理 str list, 但显式版本更稳且风格与 uea.py 一致。
    """
    if not _TORCH_OK:
        raise RuntimeError("collate_timemmd needs torch")
    first = batch[0]
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    xm = torch.stack([b[2] for b in batch])
    ym = torch.stack([b[3] for b in batch])
    if len(first) == 4:
        return xs, ys, xm, ym
    texts = [b[4] for b in batch]
    if torch.is_tensor(texts[0]):
        texts = torch.stack(texts)
    return xs, ys, xm, ym, texts


def data_provider(args, flag, provider_cache={}):
    """
    镜像 TSlib data_factory 的工厂函数; 同一 (root,domain,source) 的
    RealTextProvider 会被缓存复用, train/val/test 三次调用只加载一次文本。
    需要的 args 字段 (缺省有兜底): root_path, data_path, seq_len, label_len,
    pred_len, features, target, embed('timeF'->timeenc=1), freq, batch_size,
    num_workers; 文本项: text_source, text_lookback, return_text 等。
    """
    if not _TORCH_OK:
        raise RuntimeError(f"data_provider needs torch: {_TORCH_ERR!r}")
    g = lambda k, d=None: getattr(args, k, d)
    domain = canonical_domain(g("data_path", g("data", "Traffic")))
    key = (g("root_path", "./data"), domain, g("text_source", "both"),
           g("text_lookback", 2), bool(g("keep_history_stats", False)))
    if key not in provider_cache:
        provider_cache[key] = RealTextProvider(
            domain, root_path=key[0], text_source=key[2],
            text_lookback=key[3], max_clauses=g("max_clauses", 6),
            max_chars=g("max_chars", 1200),
            raw_text_topk=g("raw_text_topk", 6),
            raw_lookback_days=g("raw_lookback_days", 60),
            keep_history_stats=key[4])
    timeenc = 0 if g("embed", "timeF") != "timeF" else 1
    data_set = Dataset_TimeMMD(
        root_path=key[0], flag=flag,
        size=[g("seq_len"), g("label_len"), g("pred_len")]
        if g("seq_len") else None,
        features=g("features", "S"), data_path=domain,
        target=g("target", "OT"), scale=bool(g("scale", True)),
        timeenc=timeenc, freq=g("freq", "m"),
        return_text=g("return_text", "str"),
        split_ratios=tuple(g("split_ratios", (0.6, 0.2, 0.2))),
        provider=provider_cache[key], verbose=(flag == "train"))
    shuffle = flag == "train"
    loader = DataLoader(data_set, batch_size=g("batch_size", 16),
                        shuffle=shuffle, num_workers=g("num_workers", 0),
                        drop_last=(flag == "train"),
                        collate_fn=collate_timemmd)
    return data_set, loader


# ══════════════════════════════════════════════════════════════
# SECTION F — CLI (预览 / 冒烟)
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="TSlib-style Time-MMD dataset with REAL text")
    ap.add_argument("--data", default="Traffic",
                    help=f"domain | comma list | 'all' ({', '.join(DOMAINS)})")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--text_source", default="both",
                    choices=["fact", "preds", "both"])
    ap.add_argument("--preview", type=int, default=0,
                    help="N>0: 仅打印 N 条清洗文本 (无需 torch)")
    ap.add_argument("--smoke", action="store_true",
                    help="构建 train/val/test Dataset 并取一个 batch 自检")
    a = ap.parse_args()

    doms = (DOMAINS if a.data.strip().lower() == "all"
            else [canonical_domain(d) for d in a.data.split(",") if d.strip()])

    for d in doms:
        prov = RealTextProvider(d, root_path=a.root,
                                text_source=a.text_source)
        if a.preview > 0:
            prov.preview(a.preview)
        if a.smoke:
            if not _TORCH_OK:
                print(f"[smoke] skipped (torch unavailable: {_TORCH_ERR!r})")
                continue
            size = SUGGESTED_SIZE[d]
            common = dict(root_path=a.root, data_path=d, size=list(size),
                          features="S", target="OT",
                          text_source=a.text_source, provider=prov,
                          verbose=False)
            for flag in ("train", "val", "test"):
                ds = Dataset_TimeMMD(flag=flag, **common)
                dl = DataLoader(ds, batch_size=4, shuffle=False,
                                collate_fn=collate_timemmd)
                x, y, xm, ym, txt = next(iter(dl))
                print(f"[smoke] {d}/{flag}: x{tuple(x.shape)} "
                      f"y{tuple(y.shape)} xm{tuple(xm.shape)} "
                      f"ym{tuple(ym.shape)} text[0]={txt[0][:60]!r}")


if __name__ == "__main__":
    main()
