from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def ensure_spiece_alias_if_needed(tokenizer_dir: Path) -> Optional[Path]:
    """Create `spiece.model` alias when a repo ships `sentencepiece.model` only."""
    td = Path(tokenizer_dir).expanduser().resolve()
    if not td.exists() or not td.is_dir():
        return None
    src = td / "sentencepiece.model"
    dst = td / "spiece.model"
    if not src.exists():
        return dst if dst.exists() else None
    if dst.exists():
        return dst
    try:
        dst.symlink_to(src.name)
        return dst
    except Exception:
        shutil.copy2(src, dst)
        return dst


def find_sentencepiece_model_path(tokenizer_dir: Path) -> Optional[Path]:
    td = Path(tokenizer_dir).expanduser().resolve()
    for name in ("sentencepiece.model", "spiece.model"):
        p = td / name
        if p.exists():
            return p
    return None


def _cfg_token_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


class SentencePieceTokenizerAdapter:
    """Small HF-like tokenizer wrapper backed directly by sentencepiece.

    This is intentionally minimal and implements only the methods used by the
    Donut OCR scripts (encode/decode/pad/special token handling/save_pretrained).
    """

    model_input_names = ["input_ids", "attention_mask"]
    clean_up_tokenization_spaces = False
    is_fast = False

    def __init__(self, tokenizer_dir: Path):
        td = Path(tokenizer_dir).expanduser().resolve()
        if not td.exists() or not td.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {td}")
        ensure_spiece_alias_if_needed(td)
        model_path = find_sentencepiece_model_path(td)
        if model_path is None:
            raise FileNotFoundError(f"No sentencepiece model found in {td}")

        try:
            import sentencepiece as spm
        except Exception as exc:
            raise RuntimeError("sentencepiece package is required for direct BoSentencePiece loading") from exc

        sp = spm.SentencePieceProcessor()
        ok = sp.load(str(model_path))
        if not ok:
            raise RuntimeError(f"Failed to load sentencepiece model: {model_path}")

        self._sp = sp
        self._tokenizer_dir = td
        self._sp_model_path = model_path
        self.name_or_path = str(td)
        self.base_vocab_size = int(self._sp.get_piece_size())
        self.model_max_length = 512

        self._added_token_to_id: Dict[str, int] = {}
        self._id_to_added_token: Dict[int, str] = {}
        self.additional_special_tokens: List[str] = []

        # Core special token strings (ALBERT-style defaults).
        self._unk_token = "<unk>"
        self._pad_token = "<pad>"
        self._bos_token = "<s>"
        self._eos_token = "</s>"
        self._cls_token = "[CLS]"
        self._sep_token = "[SEP]"
        self._mask_token = "[MASK]"

        self._load_tokenizer_config_defaults()
        self._refresh_special_state()

    def _load_tokenizer_config_defaults(self) -> None:
        cfg_path = self._tokenizer_dir / "tokenizer_config.json"
        if not cfg_path.exists():
            return
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(cfg, dict):
            return

        self.model_max_length = int(cfg.get("model_max_length") or self.model_max_length or 512)
        for attr, key in [
            ("_unk_token", "unk_token"),
            ("_pad_token", "pad_token"),
            ("_bos_token", "bos_token"),
            ("_eos_token", "eos_token"),
            ("_cls_token", "cls_token"),
            ("_sep_token", "sep_token"),
            ("_mask_token", "mask_token"),
        ]:
            tok = _cfg_token_str(cfg.get(key))
            if tok:
                setattr(self, attr, tok)

        addl = cfg.get("additional_special_tokens")
        if isinstance(addl, list):
            for raw in addl:
                tok = _cfg_token_str(raw)
                if tok:
                    self._ensure_token(tok)
                    if tok not in self.additional_special_tokens:
                        self.additional_special_tokens.append(tok)

    def _piece_id_exact(self, token: str) -> Optional[int]:
        t = str(token)
        try:
            idx = int(self._sp.piece_to_id(t))
        except Exception:
            return None
        if idx < 0:
            return None
        try:
            piece = str(self._sp.id_to_piece(idx))
        except Exception:
            return None
        if piece != t:
            return None
        return idx

    def _ensure_token(self, token: str) -> int:
        t = str(token)
        if not t:
            return int(self.unk_token_id or 0)
        if t in self._added_token_to_id:
            return int(self._added_token_to_id[t])
        base_id = self._piece_id_exact(t)
        if base_id is not None:
            return int(base_id)
        new_id = int(self.base_vocab_size + len(self._added_token_to_id))
        self._added_token_to_id[t] = new_id
        self._id_to_added_token[new_id] = t
        return new_id

    def _special_token_strings(self) -> List[str]:
        vals = [
            self._pad_token,
            self._unk_token,
            self._bos_token,
            self._eos_token,
            self._cls_token,
            self._sep_token,
            self._mask_token,
        ] + list(self.additional_special_tokens)
        out: List[str] = []
        seen = set()
        for v in vals:
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(str(v))
        return out

    def _refresh_special_state(self) -> None:
        # Ensure all declared special tokens are registered (base or added).
        for tok in self._special_token_strings():
            self._ensure_token(tok)

        self.unk_token_id = self._ensure_token(self._unk_token) if self._unk_token else None
        self.pad_token_id = self._ensure_token(self._pad_token) if self._pad_token else None
        self.bos_token_id = self._ensure_token(self._bos_token) if self._bos_token else None
        self.eos_token_id = self._ensure_token(self._eos_token) if self._eos_token else None
        self.cls_token_id = self._ensure_token(self._cls_token) if self._cls_token else None
        self.sep_token_id = self._ensure_token(self._sep_token) if self._sep_token else None
        self.mask_token_id = self._ensure_token(self._mask_token) if self._mask_token else None

        self.all_special_tokens = self._special_token_strings()
        self.all_special_ids = [self._ensure_token(t) for t in self.all_special_tokens]
        self._all_special_ids_set = set(int(i) for i in self.all_special_ids if i is not None)
        # Longest-first matching avoids `<s>` matching inside `<s_ocr>`.
        self._special_match_tokens = sorted(
            [t for t in self.all_special_tokens if t],
            key=len,
            reverse=True,
        )

    @property
    def unk_token(self) -> str:
        return self._unk_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        self._pad_token = str(value)
        self._refresh_special_state()

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def cls_token(self) -> str:
        return self._cls_token

    @property
    def sep_token(self) -> str:
        return self._sep_token

    @property
    def mask_token(self) -> str:
        return self._mask_token

    def __len__(self) -> int:
        return int(self.base_vocab_size + len(self._added_token_to_id))

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.convert_tokens_to_ids(t) for t in tokens]
        return int(self._ensure_token(str(tokens)))

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, (list, tuple)):
            return [self.convert_ids_to_tokens(i) for i in ids]
        idx = int(ids)
        if idx in self._id_to_added_token:
            return self._id_to_added_token[idx]
        try:
            return str(self._sp.id_to_piece(idx))
        except Exception:
            return ""

    def _encode_text_ids(self, text: str) -> List[int]:
        s = str(text or "")
        if not s:
            return []
        if not self._special_match_tokens:
            return [int(x) for x in self._sp.encode_as_ids(s)]

        out: List[int] = []
        i = 0
        n = len(s)
        while i < n:
            matched: Optional[str] = None
            for tok in self._special_match_tokens:
                if tok and s.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                out.append(int(self._ensure_token(matched)))
                i += len(matched)
                continue

            # Consume plain text until the next special-token boundary.
            j = i + 1
            while j < n:
                if any(s.startswith(tok, j) for tok in self._special_match_tokens):
                    break
                j += 1
            out.extend(int(x) for x in self._sp.encode_as_ids(s[i:j]))
            i = j
        return out

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> Dict[str, List[int]]:
        ids = self._encode_text_ids(str(text or ""))
        if add_special_tokens:
            if self.cls_token_id is not None:
                ids = [int(self.cls_token_id)] + ids
            elif self.bos_token_id is not None:
                ids = [int(self.bos_token_id)] + ids
            if self.sep_token_id is not None:
                ids = ids + [int(self.sep_token_id)]
            elif self.eos_token_id is not None:
                ids = ids + [int(self.eos_token_id)]
        if truncation and max_length is not None and int(max_length) > 0 and len(ids) > int(max_length):
            ids = ids[: int(max_length)]
        return {"input_ids": ids}

    def _decode_base_ids(self, ids: Sequence[int]) -> str:
        if not ids:
            return ""
        try:
            return str(self._sp.decode_ids([int(x) for x in ids]))
        except Exception:
            pieces = [str(self._sp.id_to_piece(int(x))) for x in ids]
            try:
                return str(self._sp.decode_pieces(pieces))
            except Exception:
                return "".join(pieces)

    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        _ = clean_up_tokenization_spaces
        out_parts: List[str] = []
        base_buf: List[int] = []

        def _flush() -> None:
            if base_buf:
                out_parts.append(self._decode_base_ids(base_buf))
                base_buf.clear()

        for raw in ids:
            try:
                idx = int(raw)
            except Exception:
                continue
            if idx in self._all_special_ids_set:
                _flush()
                if not skip_special_tokens:
                    out_parts.append(str(self.convert_ids_to_tokens(idx)))
                continue
            if idx in self._id_to_added_token:
                _flush()
                if not skip_special_tokens:
                    out_parts.append(self._id_to_added_token[idx])
                continue
            base_buf.append(idx)
        _flush()
        return "".join(out_parts)

    def batch_decode(
        self,
        sequences: Sequence[Sequence[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **_: Any,
    ) -> List[str]:
        out: List[str] = []
        for seq in sequences:
            try:
                seq_list = [int(x) for x in seq]
            except Exception:
                seq_list = [int(x) for x in list(seq)]
            out.append(
                self.decode(
                    seq_list,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
            )
        return out

    def pad(
        self,
        encoded_inputs: Dict[str, Any],
        padding: bool = True,
        return_tensors: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        seqs = encoded_inputs.get("input_ids")
        if seqs is None:
            raise ValueError("encoded_inputs must contain input_ids")
        seq_lists = [[int(x) for x in seq] for seq in seqs]
        if not seq_lists:
            padded: List[List[int]] = []
        elif padding:
            max_len = max(len(seq) for seq in seq_lists)
            pad_id = int(self.pad_token_id if self.pad_token_id is not None else 0)
            padded = [seq + [pad_id] * (max_len - len(seq)) for seq in seq_lists]
        else:
            padded = seq_lists

        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor(padded, dtype=torch.long)}
        return {"input_ids": padded}

    def add_special_tokens(self, special_tokens_dict: Dict[str, Any]) -> int:
        added_before = len(self._added_token_to_id)

        pad_token = _cfg_token_str(special_tokens_dict.get("pad_token"))
        if pad_token:
            self._pad_token = pad_token

        addl = special_tokens_dict.get("additional_special_tokens")
        if isinstance(addl, list):
            for raw in addl:
                tok = _cfg_token_str(raw)
                if not tok:
                    continue
                if tok not in self.additional_special_tokens:
                    self.additional_special_tokens.append(tok)
                self._ensure_token(tok)

        self._refresh_special_state()
        return int(len(self._added_token_to_id) - added_before)

    def train_new_from_iterator(self, *args, **kwargs):
        raise NotImplementedError(
            "SentencePieceTokenizerAdapter does not support train_new_from_iterator. "
            "Use a HF tokenizer backend if you need tokenizer retraining."
        )

    def save_pretrained(self, save_directory: str | Path) -> Tuple[str]:
        out_dir = Path(save_directory).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Persist SP model files.
        for name in ("sentencepiece.model", "sentencepiece.vocab", "tokenizer.json"):
            src = self._tokenizer_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)
        ensure_spiece_alias_if_needed(out_dir)

        cfg = {
            "tokenizer_class": "SentencePieceTokenizerAdapter",
            "source_tokenizer_dir": str(self._tokenizer_dir),
            "model_max_length": int(self.model_max_length),
            "unk_token": self._unk_token,
            "pad_token": self._pad_token,
            "bos_token": self._bos_token,
            "eos_token": self._eos_token,
            "cls_token": self._cls_token,
            "sep_token": self._sep_token,
            "mask_token": self._mask_token,
            "additional_special_tokens": list(self.additional_special_tokens),
        }
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return (str(out_dir),)


def load_sentencepiece_tokenizer(tokenizer_dir: Path) -> SentencePieceTokenizerAdapter:
    return SentencePieceTokenizerAdapter(tokenizer_dir=tokenizer_dir)
