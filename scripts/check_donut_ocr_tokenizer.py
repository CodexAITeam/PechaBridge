#!/usr/bin/env python3
"""Quick CLI to inspect tokenizer behavior on Donut OCR training manifests."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TEXT_KEY_FALLBACKS: Sequence[str] = (
    "text",
    "transcription",
    "label",
    "ocr_text",
    "line_text",
    "normalized_text",
    "content",
    "src__label",
    "src__text",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect how a tokenizer encodes/decodes OCR training texts (focus: Tibetan coverage / <unk>)."
    )
    p.add_argument("--manifest", required=True, help="Path to JSONL manifest (e.g. train_manifest.jsonl)")
    p.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer path/name (local dir or HF identifier used in training)",
    )
    p.add_argument(
        "--text-key",
        default="",
        help="Optional explicit text key. If omitted, uses the same fallback list as train_donut_ocr.py.",
    )
    p.add_argument("--max-samples", type=int, default=500, help="Max rows to inspect after manifest load (0 = all)")
    p.add_argument("--sample-random", action="store_true", help="Randomly sample rows instead of taking first N")
    p.add_argument("--seed", type=int, default=42, help="Random seed used with --sample-random")
    p.add_argument("--show-examples", type=int, default=8, help="Number of example rows to print per category")
    p.add_argument(
        "--max-preview-chars",
        type=int,
        default=120,
        help="Truncate preview strings to this many chars",
    )
    p.add_argument(
        "--max-preview-tokens",
        type=int,
        default=48,
        help="Truncate token/id previews to this many entries",
    )
    p.add_argument(
        "--normalization",
        default="NFC",
        choices=["none", "NFC", "NFD", "NFKC", "NFKD"],
        help="Unicode normalization used for roundtrip comparison only",
    )
    p.add_argument(
        "--newline-token",
        default="<NL>",
        choices=["<NL>", "\\n"],
        help="How to normalize newlines before roundtrip comparison",
    )
    p.add_argument(
        "--only-tibetan-examples",
        action="store_true",
        help="Print example rows only when the original text contains Tibetan Unicode chars",
    )
    return p.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{idx}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _first_nonempty_str(row: Dict[str, object], keys: Sequence[str]) -> Tuple[str, str]:
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return key, val.strip()
    return "", ""


def _looks_tibetan_char(ch: str) -> bool:
    cp = ord(ch)
    return 0x0F00 <= cp <= 0x0FFF


def _count_tibetan_chars(text: str) -> int:
    return sum(1 for ch in text if _looks_tibetan_char(ch))


def _normalize_text(text: str, normalization: str, newline_token: str) -> str:
    out = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if newline_token == "<NL>":
        out = out.replace("<NL>", "\n")
    else:
        out = out.replace("<NL>", "\n")
    if normalization != "none":
        out = unicodedata.normalize(normalization, out)
    return out.strip()


def _safe_decode(tokenizer, ids: List[int], *, skip_special_tokens: bool) -> str:
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _truncate_list(values: Sequence[object], limit: int) -> List[object]:
    if limit <= 0 or len(values) <= limit:
        return list(values)
    out = list(values[:limit])
    out.append("...")
    return out


def _iter_selected_rows(rows: List[Dict[str, object]], max_samples: int, sample_random: bool, seed: int) -> Iterable[Tuple[int, Dict[str, object]]]:
    if max_samples <= 0 or max_samples >= len(rows):
        for i, row in enumerate(rows):
            yield i, row
        return
    indices = list(range(len(rows)))
    if sample_random:
        rng = random.Random(seed)
        rng.shuffle(indices)
    for i in indices[:max_samples]:
        yield i, rows[i]


def _ensure_spiece_alias_if_needed(tokenizer_dir: Path) -> Optional[Path]:
    sentencepiece_model = tokenizer_dir / "sentencepiece.model"
    spiece_model = tokenizer_dir / "spiece.model"
    if not sentencepiece_model.exists() or spiece_model.exists():
        return spiece_model if spiece_model.exists() else None
    try:
        try:
            spiece_model.symlink_to(sentencepiece_model.name)
        except Exception:
            shutil.copy2(sentencepiece_model, spiece_model)
        return spiece_model
    except Exception:
        return None


def _maybe_use_local_bosentencepiece(spec: str) -> str:
    clean = str(spec or "").strip()
    if clean != "openpecha/BoSentencePiece":
        return clean
    local = (REPO_ROOT / "ext" / "BoSentencePiece").resolve()
    if local.exists() and local.is_dir():
        _ensure_spiece_alias_if_needed(local)
        return str(local)
    return clean


def _is_degenerate_sp_tokenizer(tok) -> bool:
    try:
        if int(len(tok)) > 32:
            return False
    except Exception:
        return False
    return getattr(tok, "sp_model", None) is None


def _try_albert_fast_from_local_dir(local_dir: Path):
    try:
        from transformers import AlbertTokenizerFast
    except Exception:
        return None
    if not (local_dir / "tokenizer.json").exists():
        return None
    try:
        tok_fast = AlbertTokenizerFast.from_pretrained(str(local_dir))
    except Exception:
        return None
    try:
        if int(len(tok_fast)) > 32:
            return tok_fast
    except Exception:
        return None
    return None


def _tok_cfg_value_to_str(value) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _try_pretrained_fast_from_tokenizer_json(local_dir: Path):
    try:
        from transformers import PreTrainedTokenizerFast
    except Exception:
        return None
    tok_json = local_dir / "tokenizer.json"
    if not tok_json.exists():
        return None
    kwargs: Dict[str, object] = {}
    cfg_path = local_dir / "tokenizer_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                for key in ("unk_token", "bos_token", "eos_token", "sep_token", "pad_token", "cls_token", "mask_token"):
                    sval = _tok_cfg_value_to_str(cfg.get(key))
                    if sval:
                        kwargs[key] = sval
                addl = cfg.get("additional_special_tokens")
                if isinstance(addl, list):
                    addl_vals = [s for s in (_tok_cfg_value_to_str(x) for x in addl) if s]
                    if addl_vals:
                        kwargs["additional_special_tokens"] = addl_vals
                mml = cfg.get("model_max_length")
                if isinstance(mml, (int, float)) and int(mml) > 0:
                    kwargs["model_max_length"] = int(mml)
        except Exception:
            pass
    try:
        tok_fast = PreTrainedTokenizerFast(tokenizer_file=str(tok_json), **kwargs)
    except Exception:
        return None
    try:
        if int(len(tok_fast)) > 32:
            return tok_fast
    except Exception:
        return None
    return None


class _RawSentencePieceAdapter:
    """Minimal tokenizer-like wrapper for diagnostics when HF tokenizer load is broken."""

    def __init__(self, sp):
        self._sp = sp
        self.unk_token_id = self._safe_int(getattr(sp, "unk_id", lambda: -1)())
        self.pad_token_id = self._safe_int(getattr(sp, "pad_id", lambda: -1)())
        self.bos_token_id = self._safe_int(getattr(sp, "bos_id", lambda: -1)())
        self.eos_token_id = self._safe_int(getattr(sp, "eos_id", lambda: -1)())
        self.cls_token_id = None
        self.sep_token_id = None
        self.all_special_ids = [x for x in [self.pad_token_id, self.bos_token_id, self.eos_token_id] if isinstance(x, int) and x >= 0]
        self.all_special_tokens = [self._sp.id_to_piece(int(i)) for i in self.all_special_ids]

    @staticmethod
    def _safe_int(v) -> Optional[int]:
        try:
            iv = int(v)
        except Exception:
            return None
        return iv if iv >= 0 else None

    def __len__(self) -> int:
        return int(self._sp.get_piece_size())

    def __call__(self, text: str, add_special_tokens: bool = False, **_: object) -> Dict[str, List[int]]:
        ids = [int(x) for x in self._sp.encode_as_ids(str(text or ""))]
        if add_special_tokens and self.bos_token_id is not None:
            ids = [int(self.bos_token_id)] + ids
        if add_special_tokens and self.eos_token_id is not None:
            ids = ids + [int(self.eos_token_id)]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        out: List[str] = []
        for i in ids:
            try:
                out.append(str(self._sp.id_to_piece(int(i))))
            except Exception:
                out.append("")
        return out

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False) -> str:
        _ = clean_up_tokenization_spaces
        keep: List[int] = []
        specials = set(int(x) for x in self.all_special_ids)
        for i in ids:
            try:
                ii = int(i)
            except Exception:
                continue
            if skip_special_tokens and ii in specials:
                continue
            keep.append(ii)
        try:
            return str(self._sp.decode_ids(keep))
        except Exception:
            try:
                pieces = [str(self._sp.id_to_piece(i)) for i in keep]
                return str(self._sp.decode_pieces(pieces))
            except Exception:
                return ""


def _try_raw_sentencepiece_from_local_dir(local_dir: Path):
    sp_model_candidates = [local_dir / "sentencepiece.model", local_dir / "spiece.model"]
    sp_model_path = next((p for p in sp_model_candidates if p.exists()), None)
    if sp_model_path is None:
        return None
    try:
        import sentencepiece as spm
    except Exception:
        return None
    try:
        sp = spm.SentencePieceProcessor()
        ok = sp.load(str(sp_model_path))
        if not ok:
            return None
        if int(sp.get_piece_size()) <= 32:
            return None
        return _RawSentencePieceAdapter(sp)
    except Exception:
        return None


def _load_tokenizer_robust(spec: str):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"transformers import failed: {exc}") from exc

    resolved_spec = _maybe_use_local_bosentencepiece(spec)
    p = Path(str(resolved_spec)).expanduser()
    if p.exists() and p.is_dir():
        _ensure_spiece_alias_if_needed(p.resolve())

    tok = AutoTokenizer.from_pretrained(resolved_spec, use_fast=True)
    if p.exists() and p.is_dir() and _is_degenerate_sp_tokenizer(tok):
        tok_fast = _try_albert_fast_from_local_dir(p.resolve())
        if tok_fast is not None:
            return tok_fast, resolved_spec
        tok_ptf = _try_pretrained_fast_from_tokenizer_json(p.resolve())
        if tok_ptf is not None:
            return tok_ptf, resolved_spec
        tok_raw = _try_raw_sentencepiece_from_local_dir(p.resolve())
        if tok_raw is not None:
            return tok_raw, resolved_spec
    if int(len(tok)) <= 32 and getattr(tok, "sp_model", None) is None:
        raise RuntimeError(
            "Tokenizer loaded with tiny vocab and no SentencePiece model "
            f"(len={len(tok)}, class={tok.__class__.__name__}). "
            "This usually means `spiece.model` was not found. "
            "If using BoSentencePiece, run `python cli.py download-bosentencepiece-tokenizer` "
            "(or `python scripts/download_bosentencepiece_tokenizer.py`) from the repo root, "
            "then use `--tokenizer ./ext/BoSentencePiece`. "
            "Also verify `sentencepiece` is installed/importable in this venv. "
            "This checker can fall back to raw `sentencepiece` if the import works."
        )
    return tok, resolved_spec


def main() -> int:
    args = _parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    print(f"Loading manifest: {manifest_path}")
    rows = _read_jsonl(manifest_path)
    print(f"Loaded rows: {len(rows)}")
    if not rows:
        return 1

    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer, resolved_tokenizer = _load_tokenizer_robust(str(args.tokenizer))
    except Exception as exc:
        print(f"ERROR: tokenizer load failed: {exc}", file=sys.stderr)
        return 2
    if str(resolved_tokenizer) != str(args.tokenizer):
        print(f"Resolved tokenizer path: {resolved_tokenizer}")

    text_keys = [args.text_key] if str(args.text_key).strip() else list(TEXT_KEY_FALLBACKS)
    key_usage = Counter()

    all_special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    inspected = 0
    missing_text = 0
    tibetan_rows = 0
    tibetan_chars_total = 0
    total_chars = 0
    total_tokens = 0
    total_unk_tokens = 0
    rows_with_unk = 0
    rows_with_unk_tib = 0
    roundtrip_exact = 0
    roundtrip_exact_tib = 0
    rows_with_any_special_only = 0

    bad_examples: List[Dict[str, object]] = []
    mismatch_examples: List[Dict[str, object]] = []
    good_examples: List[Dict[str, object]] = []

    for row_idx, row in _iter_selected_rows(rows, int(args.max_samples), bool(args.sample_random), int(args.seed)):
        key, text = _first_nonempty_str(row, text_keys)
        if not text:
            missing_text += 1
            continue
        key_usage[key] += 1
        inspected += 1

        tib_chars = _count_tibetan_chars(text)
        has_tibetan = tib_chars > 0
        if has_tibetan:
            tibetan_rows += 1
            tibetan_chars_total += tib_chars
        total_chars += len(text)

        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(ids)
        total_tokens += len(ids)

        unk_count = 0
        if unk_id is not None:
            unk_count = sum(1 for x in ids if int(x) == int(unk_id))
        total_unk_tokens += unk_count
        if unk_count > 0:
            rows_with_unk += 1
            if has_tibetan:
                rows_with_unk_tib += 1

        special_or_unk_only = all((int(x) in all_special_ids) for x in ids) if ids else True
        if special_or_unk_only:
            rows_with_any_special_only += 1

        decoded_no_special = _safe_decode(tokenizer, list(map(int, ids)), skip_special_tokens=True)
        decoded_with_special = _safe_decode(tokenizer, list(map(int, ids)), skip_special_tokens=False)

        src_norm = _normalize_text(text, args.normalization, args.newline_token)
        dec_norm = _normalize_text(decoded_no_special, args.normalization, args.newline_token)
        exact = (src_norm == dec_norm)
        if exact:
            roundtrip_exact += 1
            if has_tibetan:
                roundtrip_exact_tib += 1

        example = {
            "row_idx": row_idx,
            "text_key": key,
            "tib_chars": tib_chars,
            "len_chars": len(text),
            "len_tokens": len(ids),
            "unk_count": unk_count,
            "source": text,
            "decoded_no_special": decoded_no_special,
            "decoded_with_special": decoded_with_special,
            "ids": [int(x) for x in ids],
            "tokens": [str(t) for t in toks],
            "exact_roundtrip": exact,
            "special_or_unk_only": bool(special_or_unk_only),
        }

        if unk_count > 0 and len(bad_examples) < max(0, int(args.show_examples)):
            if (not args.only_tibetan_examples) or has_tibetan:
                bad_examples.append(example)

        if (not exact) and unk_count == 0 and len(mismatch_examples) < max(0, int(args.show_examples)):
            if (not args.only_tibetan_examples) or has_tibetan:
                mismatch_examples.append(example)

        if exact and unk_count == 0 and len(good_examples) < max(0, int(args.show_examples)):
            if (not args.only_tibetan_examples) or has_tibetan:
                good_examples.append(example)

    def pct(num: int, den: int) -> float:
        return 0.0 if den <= 0 else (100.0 * float(num) / float(den))

    print()
    print("=== Tokenizer Summary ===")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}")
    print(
        "Special IDs: "
        + json.dumps(
            {
                "pad": pad_id,
                "unk": unk_id,
                "bos": getattr(tokenizer, "bos_token_id", None),
                "eos": getattr(tokenizer, "eos_token_id", None),
                "cls": getattr(tokenizer, "cls_token_id", None),
                "sep": getattr(tokenizer, "sep_token_id", None),
                "num_all_special_ids": len(all_special_ids),
            },
            ensure_ascii=False,
        )
    )

    print()
    print("=== Data Summary ===")
    print(f"Rows inspected: {inspected}")
    print(f"Rows missing text: {missing_text}")
    print(f"Rows with Tibetan chars: {tibetan_rows} ({pct(tibetan_rows, inspected):.1f}%)")
    print(f"Total chars: {total_chars}")
    print(f"Total Tibetan chars: {tibetan_chars_total}")
    print(f"Total tokenized ids: {total_tokens}")
    print(f"Rows with <unk>: {rows_with_unk} ({pct(rows_with_unk, inspected):.1f}%)")
    if tibetan_rows > 0:
        print(f"Tibetan rows with <unk>: {rows_with_unk_tib} ({pct(rows_with_unk_tib, tibetan_rows):.1f}%)")
    print(f"Total <unk> ids: {total_unk_tokens} ({pct(total_unk_tokens, total_tokens):.2f}% of ids)")
    print(
        f"Rows tokenized to only special/unk-like ids: {rows_with_any_special_only} "
        f"({pct(rows_with_any_special_only, inspected):.1f}%)"
    )
    print(f"Exact roundtrip (skip_special decode): {roundtrip_exact} ({pct(roundtrip_exact, inspected):.1f}%)")
    if tibetan_rows > 0:
        print(
            f"Exact roundtrip on Tibetan rows: {roundtrip_exact_tib} "
            f"({pct(roundtrip_exact_tib, tibetan_rows):.1f}%)"
        )

    if key_usage:
        print()
        print("Text key usage:")
        for k, v in key_usage.most_common():
            print(f"  {k}: {v}")

    def _print_examples(title: str, examples: List[Dict[str, object]]) -> None:
        print()
        print(f"=== {title} ({len(examples)}) ===")
        if not examples:
            return
        for ex in examples:
            print(
                f"[row={ex['row_idx']}] key={ex['text_key']} chars={ex['len_chars']} "
                f"tib={ex['tib_chars']} toks={ex['len_tokens']} unk={ex['unk_count']} "
                f"exact={ex['exact_roundtrip']} special_only={ex['special_or_unk_only']}"
            )
            print(f"  src : {json.dumps(_truncate_text(ex['source'], args.max_preview_chars), ensure_ascii=False)}")
            print(
                "  ids : "
                + json.dumps(_truncate_list(ex["ids"], int(args.max_preview_tokens)), ensure_ascii=False)
            )
            print(
                "  toks: "
                + json.dumps(_truncate_list(ex["tokens"], int(args.max_preview_tokens)), ensure_ascii=False)
            )
            print(
                "  dec(skip_special=True) : "
                + json.dumps(_truncate_text(ex["decoded_no_special"], args.max_preview_chars), ensure_ascii=False)
            )
            print(
                "  dec(skip_special=False): "
                + json.dumps(_truncate_text(ex["decoded_with_special"], args.max_preview_chars), ensure_ascii=False)
            )

    _print_examples("Examples with <unk>", bad_examples)
    _print_examples("Examples with roundtrip mismatch but no <unk>", mismatch_examples)
    _print_examples("Examples without <unk> and exact roundtrip", good_examples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
