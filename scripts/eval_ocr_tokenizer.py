#!/usr/bin/env python3
"""Evaluate tokenizer suitability for OCR targets (e.g. Donut manifests)."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("eval_ocr_tokenizer")

DEFAULT_CHECK_TOKENS = ["<NL>", "<s_ocr>", "</s_ocr>", "<s_cls1>"]
DEFAULT_BASELINE_TOKENIZERS = [
    "char://unicode",
    "gpt2",  # byte-level BPE baseline
    "bert-base-multilingual-cased",  # WordPiece multilingual baseline
    "xlm-roberta-base",  # multilingual SentencePiece baseline
    "google/byt5-small",  # byte-level tokenizer baseline
]


class _CharLevelTokenizer:
    """Simple Unicode char-level tokenizer that preserves configured special tokens."""

    def __init__(self, special_tokens: Sequence[str]):
        self.is_fast = False
        self.unk_token = None
        self.unk_token_id = None
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self.all_special_tokens: List[str] = []
        # Core specials first for deterministic ids.
        for tok in [self.pad_token, self.bos_token, self.eos_token]:
            self._ensure_token(tok, is_special=True)
        for tok in special_tokens:
            t = str(tok)
            if not t:
                continue
            self._ensure_token(t, is_special=True)
        self.pad_token_id = self._token_to_id[self.pad_token]
        self.bos_token_id = self._token_to_id[self.bos_token]
        self.eos_token_id = self._token_to_id[self.eos_token]
        # Greedy longest-first special matching during tokenization.
        self._user_specials_sorted = sorted(
            [t for t in self.all_special_tokens if t not in {self.pad_token, self.bos_token, self.eos_token}],
            key=len,
            reverse=True,
        )

    def _ensure_token(self, token: str, *, is_special: bool = False) -> int:
        if token in self._token_to_id:
            tid = self._token_to_id[token]
        else:
            tid = len(self._token_to_id)
            self._token_to_id[token] = tid
            self._id_to_token[tid] = token
        if is_special and token not in self.all_special_tokens:
            self.all_special_tokens.append(token)
        return tid

    def _split_text(self, text: str) -> List[str]:
        s = str(text)
        if not s:
            return []
        out: List[str] = []
        i = 0
        n = len(s)
        while i < n:
            matched = None
            for tok in self._user_specials_sorted:
                if tok and s.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                out.append(matched)
                i += len(matched)
                continue
            out.append(s[i])
            i += 1
        return out

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
        **_: Any,
    ) -> Dict[str, List[int]]:
        toks = self._split_text(text)
        ids = [self._ensure_token(tok) for tok in toks]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        # Flags accepted for API-compat; no-op.
        _ = (truncation, return_attention_mask, return_token_type_ids)
        return {"input_ids": ids}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False) -> str:
        _ = clean_up_tokenization_spaces
        toks: List[str] = []
        special_set = set(self.all_special_tokens) if skip_special_tokens else set()
        for i in ids:
            tok = self._id_to_token.get(int(i), "")
            if skip_special_tokens and tok in special_set:
                continue
            toks.append(tok)
        return "".join(toks)

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return [self._id_to_token.get(int(i), "") for i in ids]

    def convert_tokens_to_ids(self, tok: str) -> Optional[int]:
        return self._token_to_id.get(str(tok))

    def __len__(self) -> int:
        return len(self._token_to_id)


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_tokenizer(
    model_name_or_path: str,
    *,
    use_fast: bool,
    trust_remote_code: bool,
    char_special_tokens: Optional[Sequence[str]] = None,
):
    spec = str(model_name_or_path).strip()
    if spec.lower() in {"char", "char://unicode", "char-level", "charlevel"}:
        LOGGER.info("Using built-in char-level tokenizer baseline: %s", spec)
        return _CharLevelTokenizer(special_tokens=list(char_special_tokens or DEFAULT_CHECK_TOKENS))
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for tokenizer evaluation. Install with `pip install transformers`."
        ) from exc
    LOGGER.info("Loading tokenizer: %s (use_fast=%s)", spec, bool(use_fast))
    return AutoTokenizer.from_pretrained(
        spec,
        use_fast=bool(use_fast),
        trust_remote_code=bool(trust_remote_code),
    )


def _read_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON at %s:%d: %s", path, idx, exc)
                continue
            if isinstance(row, dict):
                yield row


def _discover_manifests(manifests_dir: Path) -> List[Path]:
    candidates = [
        manifests_dir / "train_manifest.jsonl",
        manifests_dir / "val_manifest.jsonl",
        manifests_dir / "eval_manifest.jsonl",
        manifests_dir / "test_manifest.jsonl",
        manifests_dir / "tokenizer_corpus.jsonl",
    ]
    found = [p.resolve() for p in candidates if p.exists() and p.is_file()]
    if found:
        return found
    return sorted(p.resolve() for p in manifests_dir.glob("*.jsonl") if p.is_file())


def _safe_float(num: float) -> Optional[float]:
    try:
        v = float(num)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _quantiles(values: Sequence[int], qs: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {f"p{int(round(q * 100))}": None for q in qs}
    arr = sorted(int(v) for v in values)
    out: Dict[str, Optional[float]] = {}
    n = len(arr)
    for q in qs:
        qq = min(max(float(q), 0.0), 1.0)
        if n == 1:
            val = float(arr[0])
        else:
            pos = qq * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                val = float(arr[lo])
            else:
                frac = pos - lo
                val = float(arr[lo]) * (1.0 - frac) + float(arr[hi]) * frac
        out[f"p{int(round(qq * 100))}"] = _safe_float(val)
    return out


def _decode_no_special(tokenizer, ids: Sequence[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(ids), skip_special_tokens=True))


def _tokenize_ids(tokenizer, text: str, *, add_special_tokens: bool) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=bool(add_special_tokens),
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = encoded.get("input_ids", [])
    return [int(v) for v in ids]


def _token_str_list(tokenizer, ids: Sequence[int], limit: int = 64) -> List[str]:
    toks: List[str] = []
    try:
        raw = tokenizer.convert_ids_to_tokens(list(ids))
    except Exception:
        raw = [str(i) for i in ids]
    for tok in list(raw)[: max(1, int(limit))]:
        toks.append(str(tok))
    return toks


def _normalize_ws(text: str) -> str:
    return " ".join(str(text).split())


def _safe_ratio(n: float, d: float) -> Optional[float]:
    if float(d) == 0.0:
        return None
    return _safe_float(float(n) / float(d))


def _row_name_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl"):
        name = name[:-6]
    return name


def _safe_tokenizer_label(spec: str) -> str:
    txt = str(spec or "").strip()
    if not txt:
        return "tokenizer"
    out = []
    for ch in txt:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "tokenizer"


def _resolve_tokenizer_specs(args) -> List[str]:
    specs: List[str] = []
    raw_list = list(getattr(args, "tokenizers", []) or [])
    legacy_single = str(getattr(args, "tokenizer", "") or "").strip()
    if legacy_single:
        raw_list.append(legacy_single)
    raw_list = [str(v).strip() for v in raw_list if str(v).strip()]
    if not raw_list:
        raw_list = ["openpecha/BoSentencePiece"]
    if bool(getattr(args, "with_baselines", False)):
        raw_list.extend(DEFAULT_BASELINE_TOKENIZERS)

    seen = set()
    out: List[str] = []
    for spec in raw_list:
        key = spec.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def _build_comparison_summary(report_by_tokenizer: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, rep in report_by_tokenizer.items():
        global_rollup = rep.get("global_rollup", {}) or {}
        unk_stats = global_rollup.get("unk_stats", {}) or {}
        token_stats = global_rollup.get("token_stats_no_special", {}) or {}
        roundtrip = global_rollup.get("roundtrip_decode", {}) or {}
        specials = rep.get("special_token_checks", []) or []
        special_failures = sum(1 for c in specials if bool(c.get("maps_to_unk")))
        rows.append(
            {
                "label": str(label),
                "tokenizer_source": str(((rep.get("tokenizer") or {}).get("source")) or ""),
                "class_name": str(((rep.get("tokenizer") or {}).get("class_name")) or ""),
                "rows_usable": int(global_rollup.get("rows_usable", 0) or 0),
                "unk_token_ratio_no_special": _safe_float(float(unk_stats.get("unk_token_ratio_no_special") or 0.0)),
                "samples_with_unk_ratio": _safe_float(float(unk_stats.get("samples_with_unk_ratio") or 0.0)),
                "tokens_per_char": _safe_float(float(token_stats.get("tokens_per_char") or 0.0)),
                "tokens_per_char_no_whitespace": _safe_float(float(token_stats.get("tokens_per_char_no_whitespace") or 0.0)),
                "roundtrip_exact_ratio": _safe_float(float(roundtrip.get("exact_match_ratio") or 0.0)),
                "roundtrip_ws_ratio": _safe_float(float(roundtrip.get("whitespace_normalized_match_ratio") or 0.0)),
                "special_token_map_to_unk_count": int(special_failures),
            }
        )

    def _sort_key(row: Mapping[str, Any]) -> Tuple[float, float, float]:
        unk = row.get("unk_token_ratio_no_special")
        tpc = row.get("tokens_per_char")
        rt = row.get("roundtrip_exact_ratio")
        return (
            float(unk if unk is not None else 999.0),
            float(tpc if tpc is not None else 999.0),
            -float(rt if rt is not None else 0.0),
        )

    rows.sort(key=_sort_key)
    return rows


def _evaluate_rows(
    *,
    rows: Iterable[Mapping[str, Any]],
    tokenizer,
    text_field: str,
    max_samples: int,
    example_limit: int,
) -> Dict[str, Any]:
    unk_id = tokenizer.unk_token_id
    unk_token = getattr(tokenizer, "unk_token", None)

    total_rows = 0
    usable_rows = 0
    missing_or_nonstring = 0
    empty_text_rows = 0

    total_chars = 0
    total_chars_nows = 0
    total_tokens_no_special = 0
    total_tokens_with_special = 0
    total_unk_no_special = 0
    total_unk_with_special = 0
    samples_with_unk = 0

    char_lengths: List[int] = []
    token_lengths_no_special: List[int] = []
    token_lengths_with_special: List[int] = []

    roundtrip_exact = 0
    roundtrip_ws_normalized = 0
    roundtrip_checked = 0

    unique_no_special = set()
    unique_with_special = set()

    unk_examples: List[Dict[str, Any]] = []
    roundtrip_examples: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        if int(max_samples) > 0 and usable_rows >= int(max_samples):
            break
        total_rows += 1

        text_val = row.get(text_field)
        if not isinstance(text_val, str):
            if text_val is None:
                missing_or_nonstring += 1
                continue
            text_val = str(text_val)
        text = text_val
        if text == "":
            empty_text_rows += 1
            continue

        ids_no_special = _tokenize_ids(tokenizer, text, add_special_tokens=False)
        ids_with_special = _tokenize_ids(tokenizer, text, add_special_tokens=True)

        usable_rows += 1
        char_len = len(text)
        char_nows = len("".join(text.split()))
        total_chars += char_len
        total_chars_nows += char_nows
        total_tokens_no_special += len(ids_no_special)
        total_tokens_with_special += len(ids_with_special)
        char_lengths.append(char_len)
        token_lengths_no_special.append(len(ids_no_special))
        token_lengths_with_special.append(len(ids_with_special))
        unique_no_special.update(ids_no_special)
        unique_with_special.update(ids_with_special)

        if unk_id is not None:
            unk_ns = sum(1 for tid in ids_no_special if int(tid) == int(unk_id))
            unk_ws = sum(1 for tid in ids_with_special if int(tid) == int(unk_id))
        else:
            unk_ns = 0
            unk_ws = 0
        total_unk_no_special += int(unk_ns)
        total_unk_with_special += int(unk_ws)
        if unk_ns > 0:
            samples_with_unk += 1
            if len(unk_examples) < int(example_limit):
                unk_examples.append(
                    {
                        "row_index": int(row_idx),
                        "text_preview": text[:200],
                        "char_len": int(char_len),
                        "token_len": int(len(ids_no_special)),
                        "unk_count": int(unk_ns),
                        "token_ids": [int(v) for v in ids_no_special[:64]],
                        "tokens": _token_str_list(tokenizer, ids_no_special, limit=64),
                    }
                )

        decoded = _decode_no_special(tokenizer, ids_no_special)
        roundtrip_checked += 1
        exact_match = decoded == text
        ws_match = _normalize_ws(decoded) == _normalize_ws(text)
        if exact_match:
            roundtrip_exact += 1
        if ws_match:
            roundtrip_ws_normalized += 1
        if (not exact_match) and len(roundtrip_examples) < int(example_limit):
            roundtrip_examples.append(
                {
                    "row_index": int(row_idx),
                    "text_preview": text[:200],
                    "decoded_preview": decoded[:200],
                    "char_len": int(char_len),
                    "token_len": int(len(ids_no_special)),
                    "token_ids": [int(v) for v in ids_no_special[:64]],
                    "tokens": _token_str_list(tokenizer, ids_no_special, limit=64),
                }
            )

    vocab_size = None
    try:
        vocab_size = int(len(tokenizer))
    except Exception:
        vocab_size = None

    summary: Dict[str, Any] = {
        "rows_total_seen": int(total_rows),
        "rows_usable": int(usable_rows),
        "rows_missing_or_nonstring_text": int(missing_or_nonstring),
        "rows_empty_text": int(empty_text_rows),
        "text_field": text_field,
        "char_stats": {
            "total_chars": int(total_chars),
            "total_chars_no_whitespace": int(total_chars_nows),
            "avg_chars_per_row": _safe_ratio(total_chars, max(1, usable_rows)),
            "quantiles": _quantiles(char_lengths, qs=[0.5, 0.9, 0.95, 0.99]),
        },
        "token_stats_no_special": {
            "total_tokens": int(total_tokens_no_special),
            "avg_tokens_per_row": _safe_ratio(total_tokens_no_special, max(1, usable_rows)),
            "tokens_per_char": _safe_ratio(total_tokens_no_special, max(1, total_chars)),
            "tokens_per_char_no_whitespace": _safe_ratio(total_tokens_no_special, max(1, total_chars_nows)),
            "quantiles": _quantiles(token_lengths_no_special, qs=[0.5, 0.9, 0.95, 0.99]),
            "unique_token_count": int(len(unique_no_special)),
        },
        "token_stats_with_special": {
            "total_tokens": int(total_tokens_with_special),
            "avg_tokens_per_row": _safe_ratio(total_tokens_with_special, max(1, usable_rows)),
            "quantiles": _quantiles(token_lengths_with_special, qs=[0.5, 0.9, 0.95, 0.99]),
            "unique_token_count": int(len(unique_with_special)),
        },
        "unk_stats": {
            "unk_token_id": int(unk_id) if unk_id is not None else None,
            "unk_token": str(unk_token) if unk_token is not None else None,
            "unk_count_no_special": int(total_unk_no_special),
            "unk_count_with_special": int(total_unk_with_special),
            "unk_token_ratio_no_special": _safe_ratio(total_unk_no_special, max(1, total_tokens_no_special)),
            "unk_token_ratio_with_special": _safe_ratio(total_unk_with_special, max(1, total_tokens_with_special)),
            "samples_with_unk": int(samples_with_unk),
            "samples_with_unk_ratio": _safe_ratio(samples_with_unk, max(1, usable_rows)),
            "coverage_ratio_no_special": None
            if unk_id is None
            else _safe_float(1.0 - float(total_unk_no_special) / float(max(1, total_tokens_no_special))),
        },
        "roundtrip_decode": {
            "checked_rows": int(roundtrip_checked),
            "exact_match_rows": int(roundtrip_exact),
            "exact_match_ratio": _safe_ratio(roundtrip_exact, max(1, roundtrip_checked)),
            "whitespace_normalized_match_rows": int(roundtrip_ws_normalized),
            "whitespace_normalized_match_ratio": _safe_ratio(roundtrip_ws_normalized, max(1, roundtrip_checked)),
        },
        "vocab_usage": {
            "tokenizer_vocab_size": int(vocab_size) if vocab_size is not None else None,
            "unique_vocab_used_no_special_ratio": None
            if not vocab_size
            else _safe_ratio(len(unique_no_special), vocab_size),
            "unique_vocab_used_with_special_ratio": None
            if not vocab_size
            else _safe_ratio(len(unique_with_special), vocab_size),
        },
        "examples": {
            "unk_examples": unk_examples,
            "roundtrip_mismatch_examples": roundtrip_examples,
        },
    }
    return summary


def _merge_eval_summaries(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a lightweight global rollup from per-manifest summaries."""
    total_rows_seen = 0
    total_rows_usable = 0
    total_missing = 0
    total_empty = 0
    total_chars = 0
    total_chars_nows = 0
    total_toks_ns = 0
    total_toks_ws = 0
    total_unk_ns = 0
    total_unk_ws = 0
    total_samples_with_unk = 0
    total_roundtrip_checked = 0
    total_roundtrip_exact = 0
    total_roundtrip_ws = 0

    for s in summaries:
        total_rows_seen += int(s.get("rows_total_seen", 0) or 0)
        total_rows_usable += int(s.get("rows_usable", 0) or 0)
        total_missing += int(s.get("rows_missing_or_nonstring_text", 0) or 0)
        total_empty += int(s.get("rows_empty_text", 0) or 0)
        char_stats = s.get("char_stats", {}) or {}
        total_chars += int(char_stats.get("total_chars", 0) or 0)
        total_chars_nows += int(char_stats.get("total_chars_no_whitespace", 0) or 0)
        tns = (s.get("token_stats_no_special", {}) or {})
        tws = (s.get("token_stats_with_special", {}) or {})
        total_toks_ns += int(tns.get("total_tokens", 0) or 0)
        total_toks_ws += int(tws.get("total_tokens", 0) or 0)
        unk_stats = s.get("unk_stats", {}) or {}
        total_unk_ns += int(unk_stats.get("unk_count_no_special", 0) or 0)
        total_unk_ws += int(unk_stats.get("unk_count_with_special", 0) or 0)
        total_samples_with_unk += int(unk_stats.get("samples_with_unk", 0) or 0)
        rt = s.get("roundtrip_decode", {}) or {}
        total_roundtrip_checked += int(rt.get("checked_rows", 0) or 0)
        total_roundtrip_exact += int(rt.get("exact_match_rows", 0) or 0)
        total_roundtrip_ws += int(rt.get("whitespace_normalized_match_rows", 0) or 0)

    return {
        "rows_total_seen": int(total_rows_seen),
        "rows_usable": int(total_rows_usable),
        "rows_missing_or_nonstring_text": int(total_missing),
        "rows_empty_text": int(total_empty),
        "char_stats": {
            "total_chars": int(total_chars),
            "total_chars_no_whitespace": int(total_chars_nows),
            "avg_chars_per_row": _safe_ratio(total_chars, max(1, total_rows_usable)),
        },
        "token_stats_no_special": {
            "total_tokens": int(total_toks_ns),
            "avg_tokens_per_row": _safe_ratio(total_toks_ns, max(1, total_rows_usable)),
            "tokens_per_char": _safe_ratio(total_toks_ns, max(1, total_chars)),
            "tokens_per_char_no_whitespace": _safe_ratio(total_toks_ns, max(1, total_chars_nows)),
        },
        "token_stats_with_special": {
            "total_tokens": int(total_toks_ws),
            "avg_tokens_per_row": _safe_ratio(total_toks_ws, max(1, total_rows_usable)),
        },
        "unk_stats": {
            "unk_count_no_special": int(total_unk_ns),
            "unk_count_with_special": int(total_unk_ws),
            "unk_token_ratio_no_special": _safe_ratio(total_unk_ns, max(1, total_toks_ns)),
            "unk_token_ratio_with_special": _safe_ratio(total_unk_ws, max(1, total_toks_ws)),
            "samples_with_unk": int(total_samples_with_unk),
            "samples_with_unk_ratio": _safe_ratio(total_samples_with_unk, max(1, total_rows_usable)),
        },
        "roundtrip_decode": {
            "checked_rows": int(total_roundtrip_checked),
            "exact_match_rows": int(total_roundtrip_exact),
            "exact_match_ratio": _safe_ratio(total_roundtrip_exact, max(1, total_roundtrip_checked)),
            "whitespace_normalized_match_rows": int(total_roundtrip_ws),
            "whitespace_normalized_match_ratio": _safe_ratio(total_roundtrip_ws, max(1, total_roundtrip_checked)),
        },
    }


def _check_special_tokens(tokenizer, tokens: Sequence[str]) -> List[Dict[str, Any]]:
    unk_id = tokenizer.unk_token_id
    out: List[Dict[str, Any]] = []
    for tok in tokens:
        token = str(tok)
        tid: Optional[int]
        try:
            raw_id = tokenizer.convert_tokens_to_ids(token)
            tid = None if raw_id is None else int(raw_id)
        except Exception:
            tid = None
        maps_to_unk = bool(
            tid is not None
            and unk_id is not None
            and int(tid) == int(unk_id)
            and str(token) != str(getattr(tokenizer, "unk_token", ""))
        )
        try:
            enc_ids = _tokenize_ids(tokenizer, token, add_special_tokens=False)
        except Exception:
            enc_ids = []
        out.append(
            {
                "token": token,
                "token_id": tid,
                "maps_to_unk": bool(maps_to_unk),
                "in_all_special_tokens": bool(token in set(getattr(tokenizer, "all_special_tokens", []) or [])),
                "encoded_ids_no_special": [int(v) for v in enc_ids],
                "encodes_as_single_id": bool(len(enc_ids) == 1),
            }
        )
    return out


def _evaluate_single_tokenizer(
    *,
    tokenizer_spec: str,
    tokenizer,
    manifest_paths: Sequence[Path],
    text_field: str,
    max_samples_per_manifest: int,
    example_limit: int,
    check_tokens: Sequence[str],
) -> Dict[str, Any]:
    special_token_checks = _check_special_tokens(tokenizer, check_tokens)
    for chk in special_token_checks:
        LOGGER.info(
            "[%s] Token check %s: id=%s maps_to_unk=%s in_specials=%s enc=%s",
            tokenizer_spec,
            chk.get("token"),
            chk.get("token_id"),
            chk.get("maps_to_unk"),
            chk.get("in_all_special_tokens"),
            chk.get("encoded_ids_no_special"),
        )

    per_manifest: Dict[str, Dict[str, Any]] = {}
    per_manifest_summaries: List[Dict[str, Any]] = []

    for manifest_path in manifest_paths:
        label = _row_name_from_path(manifest_path)
        LOGGER.info("[%s] Evaluating manifest: %s", tokenizer_spec, manifest_path)
        rows_iter = _read_jsonl_records(manifest_path)
        result = _evaluate_rows(
            rows=rows_iter,
            tokenizer=tokenizer,
            text_field=str(text_field),
            max_samples=int(max_samples_per_manifest),
            example_limit=int(example_limit),
        )
        result["manifest_path"] = str(manifest_path)
        per_manifest[label] = result
        per_manifest_summaries.append(result)

        unk_ratio = ((result.get("unk_stats") or {}).get("unk_token_ratio_no_special"))
        toks_per_char = ((result.get("token_stats_no_special") or {}).get("tokens_per_char"))
        LOGGER.info(
            "[%s] Manifest %s: usable=%d unk_ratio=%.6f toks/char=%.4f",
            tokenizer_spec,
            label,
            int(result.get("rows_usable", 0) or 0),
            float(unk_ratio or 0.0),
            float(toks_per_char or 0.0),
        )

    report: Dict[str, Any] = {
        "tokenizer": {
            "source": str(tokenizer_spec),
            "class_name": tokenizer.__class__.__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "vocab_size": int(len(tokenizer)) if hasattr(tokenizer, "__len__") else None,
            "unk_token": getattr(tokenizer, "unk_token", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "bos_token": getattr(tokenizer, "bos_token", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "all_special_tokens": list(getattr(tokenizer, "all_special_tokens", []) or []),
        },
        "special_token_checks": special_token_checks,
        "manifests": {k: v for k, v in per_manifest.items()},
        "global_rollup": _merge_eval_summaries(per_manifest_summaries),
    }
    global_unk_ratio = (((report.get("global_rollup") or {}).get("unk_stats") or {}).get("unk_token_ratio_no_special"))
    global_tpc = (((report.get("global_rollup") or {}).get("token_stats_no_special") or {}).get("tokens_per_char"))
    global_rows = int(((report.get("global_rollup") or {}).get("rows_usable") or 0))
    LOGGER.info(
        "[%s] Global rollup: usable=%d unk_ratio=%.6f toks/char=%.4f",
        tokenizer_spec,
        global_rows,
        float(global_unk_ratio or 0.0),
        float(global_tpc or 0.0),
    )
    return report


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate tokenizer coverage/length behavior on OCR text manifests.",
        add_help=add_help,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        dest="tokenizers",
        action="append",
        default=[],
        help=(
            "Tokenizer model name or local path (repeatable). "
            "Supports pseudo-tokenizer `char://unicode`. "
            "Default (if omitted): openpecha/BoSentencePiece."
        ),
    )
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help=(
            "Compare against curated baselines in addition to provided tokenizers: "
            "char://unicode, gpt2 (BPE), bert-base-multilingual-cased, "
            "xlm-roberta-base, google/byt5-small."
        ),
    )
    parser.add_argument(
        "--manifest",
        dest="manifests",
        action="append",
        default=[],
        help="JSONL manifest path (repeatable). If omitted, use --manifests-dir auto-discovery.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=str,
        default="",
        help="Directory containing train/val/eval manifests (auto-discovers *.jsonl).",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name containing OCR target text in the manifest (default: text).",
    )
    parser.add_argument(
        "--max-samples-per-manifest",
        type=int,
        default=0,
        help="Optional cap per manifest for quick experiments (0 = all rows).",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=10,
        help="Number of UNK/roundtrip example rows to keep per manifest.",
    )
    parser.add_argument(
        "--check-token",
        dest="check_tokens",
        action="append",
        default=[],
        help="Token to verify in the tokenizer vocab (repeatable). Default checks Donut tokens + <NL>.",
    )
    parser.add_argument("--use-fast", dest="use_fast", action="store_true", help="Use fast tokenizer if available (default).")
    parser.add_argument("--no-use-fast", dest="use_fast", action="store_false", help="Force slow tokenizer.")
    parser.set_defaults(use_fast=True)
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoTokenizer.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write the full JSON evaluation report.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser


def run(args) -> Dict[str, Any]:
    _configure_logging(bool(getattr(args, "verbose", False)))

    manifest_paths: List[Path] = []
    manifest_paths.extend(Path(p).expanduser().resolve() for p in (getattr(args, "manifests", []) or []) if str(p).strip())
    manifests_dir_raw = str(getattr(args, "manifests_dir", "") or "").strip()
    if manifests_dir_raw:
        manifest_paths.extend(_discover_manifests(Path(manifests_dir_raw).expanduser().resolve()))

    deduped_manifest_paths: List[Path] = []
    seen_manifest_paths = set()
    for p in manifest_paths:
        key = str(p.resolve())
        if key in seen_manifest_paths:
            continue
        seen_manifest_paths.add(key)
        deduped_manifest_paths.append(p)
    manifest_paths = deduped_manifest_paths

    if not manifest_paths:
        raise ValueError("No manifests provided. Use --manifest or --manifests-dir.")
    check_tokens = [str(v) for v in (getattr(args, "check_tokens", []) or []) if str(v).strip()] or list(DEFAULT_CHECK_TOKENS)

    tokenizer_specs = _resolve_tokenizer_specs(args)
    LOGGER.info("Tokenizer comparison set (%d): %s", len(tokenizer_specs), tokenizer_specs)

    reports_by_label: Dict[str, Dict[str, Any]] = {}
    for spec in tokenizer_specs:
        tokenizer = _load_tokenizer(
            spec,
            use_fast=bool(args.use_fast),
            trust_remote_code=bool(args.trust_remote_code),
            char_special_tokens=check_tokens,
        )
        base_label = _safe_tokenizer_label(spec)
        label = base_label
        suffix = 2
        while label in reports_by_label:
            label = f"{base_label}_{suffix}"
            suffix += 1

        reports_by_label[label] = _evaluate_single_tokenizer(
            tokenizer_spec=spec,
            tokenizer=tokenizer,
            manifest_paths=manifest_paths,
            text_field=str(args.text_field),
            max_samples_per_manifest=int(args.max_samples_per_manifest),
            example_limit=int(args.example_limit),
            check_tokens=check_tokens,
        )

    comparison_table = _build_comparison_summary(reports_by_label)
    for row in comparison_table:
        LOGGER.info(
            "Compare %-28s unk=%.6f toks/char=%.4f roundtrip=%.4f special_unk=%d",
            str(row.get("label", ""))[:28],
            float(row.get("unk_token_ratio_no_special") or 0.0),
            float(row.get("tokens_per_char") or 0.0),
            float(row.get("roundtrip_exact_ratio") or 0.0),
            int(row.get("special_token_map_to_unk_count") or 0),
        )

    # Backwards-compatible shape for single tokenizer; richer comparison shape otherwise.
    if len(reports_by_label) == 1:
        only_label = next(iter(reports_by_label.keys()))
        report = dict(reports_by_label[only_label])
        report["comparison"] = {
            "mode": "single",
            "label": only_label,
            "tokenizer_specs": tokenizer_specs,
            "comparison_table": comparison_table,
        }
    else:
        report = {
            "comparison": {
                "mode": "multi",
                "tokenizer_specs": tokenizer_specs,
                "comparison_table": comparison_table,
                "manifests": [str(p) for p in manifest_paths],
                "check_tokens": list(check_tokens),
            },
            "results_by_tokenizer": reports_by_label,
        }

    output_json = str(getattr(args, "output_json", "") or "").strip()
    if output_json:
        out_path = Path(output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Wrote tokenizer eval report to %s", out_path)

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
