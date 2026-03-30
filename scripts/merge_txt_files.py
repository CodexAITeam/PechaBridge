#!/usr/bin/env python3
"""Merge all .txt files in a folder into a single file.

Usage:
    python scripts/merge_txt_files.py <input_dir> <output_file>

Example:
    python scripts/merge_txt_files.py /data/pechas/W1234__checkpoint-5000 merged.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge all .txt files in a folder into one file.",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing .txt files to merge.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the merged output file.",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="",
        help="Optional blank line(s) between entries (default: none).",
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        return 1

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{input_dir}'.", file=sys.stderr)
        return 1

    separator = args.separator if args.separator else ""

    with output_file.open("w", encoding="utf-8") as out:
        for i, txt_path in enumerate(txt_files):
            content = txt_path.read_text(encoding="utf-8")
            out.write(txt_path.name + "\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            if separator and i < len(txt_files) - 1:
                out.write(separator + "\n")

    print(f"Merged {len(txt_files)} file(s) → {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
