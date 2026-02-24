#!/usr/bin/env python3
"""Download and verify OpenPecha BoSentencePiece tokenizer into ext/BoSentencePiece."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "ext" / "BoSentencePiece"


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download OpenPecha BoSentencePiece tokenizer locally and prepare an Albert-compatible spiece.model alias.",
        add_help=add_help,
    )
    p.add_argument("--repo-id", default="openpecha/BoSentencePiece", help="HF repo id to download")
    p.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="Destination directory (default: ./ext/BoSentencePiece)",
    )
    p.add_argument(
        "--allow-patterns",
        default="*.model,*.vocab,*.json,*.txt,.gitattributes",
        help="Comma-separated HF snapshot allow-patterns",
    )
    p.add_argument(
        "--force-alias-copy",
        action="store_true",
        help="Copy sentencepiece.model to spiece.model even if symlinks are supported",
    )
    return p


def _ensure_spiece_alias(tokenizer_dir: Path, *, force_copy: bool = False) -> Path | None:
    sentencepiece_model = tokenizer_dir / "sentencepiece.model"
    spiece_model = tokenizer_dir / "spiece.model"
    if not sentencepiece_model.exists():
        return None
    if spiece_model.exists():
        return spiece_model
    if force_copy:
        shutil.copy2(sentencepiece_model, spiece_model)
        return spiece_model
    try:
        spiece_model.symlink_to(sentencepiece_model.name)
        return spiece_model
    except Exception:
        shutil.copy2(sentencepiece_model, spiece_model)
        return spiece_model


def run(args: argparse.Namespace) -> int:
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(f"ERROR: huggingface_hub is required: {exc}", file=sys.stderr)
        print("Install with: python3 -m pip install huggingface_hub", file=sys.stderr)
        return 2

    allow_patterns = [p.strip() for p in str(args.allow_patterns).split(",") if p.strip()]
    print(f"Downloading {args.repo_id} -> {dest}")
    snapshot_download(
        repo_id=str(args.repo_id),
        local_dir=str(dest),
        allow_patterns=allow_patterns or None,
    )

    alias = _ensure_spiece_alias(dest, force_copy=bool(args.force_alias_copy))
    if alias is not None:
        print(f"Prepared alias: {alias.name} -> sentencepiece.model")
    else:
        print("No sentencepiece.model found; alias not created")

    print("Files:")
    for p in sorted(dest.iterdir()):
        kind = "dir" if p.is_dir() else "file"
        print(f"  - {p.name} ({kind})")

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"WARNING: transformers not available for validation: {exc}")
        return 0

    try:
        import transformers as _tf
        print(f"transformers: {_tf.__version__}")
    except Exception:
        pass
    try:
        import tokenizers as _toks
        print(f"tokenizers: {_toks.__version__}")
    except Exception as exc:
        print(f"tokenizers import error: {exc}")
    try:
        import sentencepiece as _spm
        print(f"sentencepiece: {_spm.__version__}")
        try:
            sp_test = _spm.SentencePieceProcessor()
            model_path = dest / "sentencepiece.model"
            if not model_path.exists():
                model_path = dest / "spiece.model"
            ok = sp_test.load(str(model_path))
            print(f"sentencepiece direct load ok: {bool(ok)} path={model_path}")
            print(f"sentencepiece direct piece_size: {sp_test.get_piece_size()}")
            sample_text = "བོད་སྐད་ཀྱི་ཚིག་གྲུབ་འདི་ཡིན།"
            sample_pieces = sp_test.encode_as_pieces(sample_text)
            print(f"sentencepiece sample pieces: {sample_pieces[:24]}")
        except Exception as exc:
            print(f"sentencepiece direct load error: {exc}")
    except Exception as exc:
        print(f"sentencepiece import error: {exc}")

    print("Validating local tokenizer load...")
    tok = AutoTokenizer.from_pretrained(str(dest), use_fast=True)
    print(f"class: {tok.__class__.__name__}")
    print(f"len(tokenizer): {len(tok)}")
    print(f"all_special_tokens: {getattr(tok, 'all_special_tokens', None)}")
    print(f"vocab_file: {getattr(tok, 'vocab_file', None)}")
    sp = getattr(tok, "sp_model", None)
    print(f"sp_model: {type(sp).__name__ if sp is not None else None}")
    if sp is not None:
        try:
            print(f"sp_piece_size: {sp.get_piece_size()}")
        except Exception as exc:
            print(f"sp_piece_size error: {exc}")

    if int(len(tok)) <= 32:
        try:
            from transformers import AlbertTokenizerFast
            tok_fast = AlbertTokenizerFast.from_pretrained(str(dest))
            print("AlbertTokenizerFast fallback validation:")
            print(f"class: {tok_fast.__class__.__name__}")
            print(f"len(tokenizer): {len(tok_fast)}")
            if int(len(tok_fast)) > 32:
                print("AutoTokenizer was degenerate, but AlbertTokenizerFast works. Training/checker loaders can use this fallback.")
                return 0
        except Exception as exc:
            print(f"AlbertTokenizerFast fallback failed: {exc}")

        print(
            "ERROR: tokenizer still loaded with tiny vocab. Check local files and transformers/tokenizers/sentencepiece versions. "
            "Try `python3 -m pip install -U sentencepiece tokenizers transformers` in this venv.",
            file=sys.stderr,
        )
        return 1

    print("\nUse this in training:")
    print(f"  --tokenizer_path {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = create_parser(add_help=True)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
