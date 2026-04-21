#!/usr/bin/env python3
"""Download page images from the Staatsbibliothek zu Berlin (SBB / Stabi) by PPN.

Uses the SBB METS/MODS API to retrieve image URLs and downloads them in parallel.
No OCR or layout analysis is performed — this is a pure download utility.

A ``metadata.json`` file is always written to the output directory alongside the
images.  It contains document-level metadata (title, author, date, …) plus the
ordered list of source image URLs so the transcript can be matched back to the
correct page later.

Example
-------
    python cli.py download-sbb-images --ppn 337138764X
    python cli.py download-sbb-images --ppn 337138764X --output-dir sbb_images/PPN337138764X
    python cli.py download-sbb-images --ppn 337138764X --max-pages 10
    python cli.py download-sbb-images --ppn 337138764X --workers 4 --no-verify-ssl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("download_sbb_images")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download-sbb-images",
        description=(
            "Download page images from the Staatsbibliothek zu Berlin (SBB / Stabi) "
            "digital collections by PPN (Pica Production Number).\n\n"
            "Images are fetched from the SBB METS/MODS API and saved to disk. "
            "No OCR or layout analysis is performed."
        ),
        add_help=add_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--ppn",
        type=str,
        required=True,
        metavar="PPN",
        help=(
            "PPN (Pica Production Number) of the SBB document to download. "
            "Example: 337138764X. "
            "Find PPNs at https://stabikat.de or https://digital.staatsbibliothek-berlin.de."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory to save downloaded images. "
            "Defaults to sbb_images/<PPN> (created automatically)."
        ),
    )
    parser.add_argument(
        "--max-pages",
        "--max_pages",
        dest="max_pages",
        type=int,
        default=0,
        metavar="N",
        help="Maximum number of pages to download (0 = all pages, default: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Number of parallel download threads (default: 8).",
    )
    parser.add_argument(
        "--no-verify-ssl",
        "--no_verify_ssl",
        dest="verify_ssl",
        action="store_false",
        default=True,
        help="Disable SSL certificate verification (use if you encounter SSL errors).",
    )
    parser.add_argument(
        "--show-metadata",
        "--show_metadata",
        dest="show_metadata",
        action="store_true",
        default=False,
        help="Print document metadata (title, author, date, pages) before downloading.",
    )

    return parser


def run(args: argparse.Namespace) -> int:
    """Download SBB images for the given PPN and save metadata.json.

    Returns 0 on success, 1 on failure.
    """
    from tibetan_utils.sbb_utils import (
        download_image,
        get_images_from_sbb,
        get_sbb_metadata,
    )
    from tibetan_utils.io_utils import ensure_dir
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from typing import Tuple

    ppn: str = args.ppn.strip()
    # Strip leading "PPN" prefix if the user included it (e.g. "PPN337138764X")
    if ppn.upper().startswith("PPN"):
        ppn = ppn[3:]

    output_dir: str = args.output_dir or f"sbb_images/{ppn}"
    max_pages: int = max(0, int(args.max_pages))
    workers: int = max(1, int(args.workers))
    verify_ssl: bool = bool(args.verify_ssl)

    LOGGER.info("PPN          : %s", ppn)
    LOGGER.info("Output dir   : %s", output_dir)
    LOGGER.info("Max pages    : %s", max_pages if max_pages > 0 else "all")
    LOGGER.info("Workers      : %d", workers)
    LOGGER.info("Verify SSL   : %s", verify_ssl)

    # Always fetch document metadata (needed for metadata.json)
    print("Fetching document metadata …")
    meta = get_sbb_metadata(ppn, verify_ssl=verify_ssl)

    # Optionally print metadata to stdout
    if args.show_metadata:
        print(f"  Title          : {meta.get('title') or '(unknown)'}")
        if meta.get('subtitle'):
            print(f"  Subtitle       : {meta['subtitle']}")
        if meta.get('title_part'):
            print(f"  Part           : {meta['title_part']}")
        authors = meta.get('authors') or []
        if authors:
            print(f"  Author(s)      : {'; '.join(authors)}")
        else:
            print(f"  Author         : (unknown)")
        contributors = meta.get('contributors') or []
        if contributors:
            for c in contributors:
                print(f"  Contributor    : {c['name']} [{c['role']}]")
        print(f"  Date           : {meta.get('date') or '(unknown)'}")
        if meta.get('date_other'):
            print(f"  Date (other)   : {meta['date_other']}")
        if meta.get('place'):
            print(f"  Place          : {meta['place']}")
        if meta.get('publisher'):
            print(f"  Publisher      : {meta['publisher']}")
        if meta.get('edition'):
            print(f"  Edition        : {meta['edition']}")
        if meta.get('extent'):
            print(f"  Extent         : {meta['extent']}")
        if meta.get('physical_description'):
            print(f"  Physical desc. : {meta['physical_description']}")
        languages = meta.get('languages') or []
        print(f"  Language(s)    : {', '.join(languages) if languages else '(unknown)'}")
        identifiers = meta.get('identifiers') or {}
        for id_type, id_val in identifiers.items():
            print(f"  Identifier     : [{id_type}] {id_val}")
        subjects = meta.get('subjects') or []
        for s in subjects:
            print(f"  Subject        : {s}")
        classifications = meta.get('classifications') or []
        for cls in classifications:
            print(f"  Classification : [{cls['authority']}] {cls['value']}")
        if meta.get('abstract'):
            print(f"  Abstract       : {meta['abstract']}")
        notes = meta.get('notes') or []
        for note in notes:
            print(f"  Note           : {note}")
        if meta.get('record_origin'):
            print(f"  Record origin  : {meta['record_origin']}")
        print(f"  Pages          : {meta.get('pages', 0)}")
        print(f"  URL            : {meta.get('url', '')}")
        print()

    # Retrieve image URLs from METS XML
    print(f"Retrieving image list for PPN {ppn} …")
    image_urls = get_images_from_sbb(ppn, verify_ssl=verify_ssl)

    if not image_urls:
        print("ERROR: No images found for this PPN. Check that the PPN is correct.", file=sys.stderr)
        return 1

    total_available = len(image_urls)
    if max_pages > 0 and total_available > max_pages:
        print(f"Limiting download to {max_pages} of {total_available} pages.")
        image_urls = image_urls[:max_pages]
    else:
        print(f"Found {total_available} page(s) to download.")

    # Ensure output directory exists
    ensure_dir(output_dir)
    out_path = Path(output_dir).resolve()
    print(f"Saving images to: {out_path}")

    # Download in parallel, tracking saved filenames in order
    total = len(image_urls)
    succeeded = 0
    failed = 0
    # ordered_filenames[i] = filename saved for image_urls[i], or None on failure
    ordered_filenames: list[Optional[str]] = [None] * total

    if workers == 1 or total == 1:
        for i, url in enumerate(image_urls):
            path = download_image(url, output_dir, verify_ssl=verify_ssl)
            if path:
                succeeded += 1
                fname = Path(path).name
                ordered_filenames[i] = fname
                print(f"  [{i + 1}/{total}] Saved: {fname}")
            else:
                failed += 1
                print(f"  [{i + 1}/{total}] FAILED: {url}", file=sys.stderr)
    else:
        effective_workers = min(workers, total)
        print(f"Downloading with {effective_workers} parallel workers …")

        def _download_one(idx_url: Tuple[int, str]) -> Tuple[int, Optional[str], str]:
            idx, url = idx_url
            path = download_image(url, output_dir, verify_ssl=verify_ssl)
            return idx, path, url

        results: list[Tuple[int, Optional[str], str]] = []
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(_download_one, pair) for pair in enumerate(image_urls)]
            for future in as_completed(futures):
                results.append(future.result())
                done = len(results)
                if done % 25 == 0 or done == total:
                    print(f"  Progress: {done}/{total}")

        # Collect results in original order
        results.sort(key=lambda r: r[0])
        for idx, path, url in results:
            if path:
                succeeded += 1
                ordered_filenames[idx] = Path(path).name
            else:
                failed += 1
                print(f"  FAILED [{idx + 1}/{total}]: {url}", file=sys.stderr)

    # Build and save metadata.json
    pages_info = [
        {
            "index": i,
            "filename": ordered_filenames[i],
            "source_url": image_urls[i],
        }
        for i in range(total)
    ]
    metadata_out = {
        "ppn": ppn,
        "title": meta.get("title"),
        "subtitle": meta.get("subtitle"),
        "title_part": meta.get("title_part"),
        "authors": meta.get("authors") or [],
        "author": meta.get("author"),
        "contributors": meta.get("contributors") or [],
        "date": meta.get("date"),
        "date_other": meta.get("date_other"),
        "place": meta.get("place"),
        "publisher": meta.get("publisher"),
        "edition": meta.get("edition"),
        "physical_description": meta.get("physical_description"),
        "extent": meta.get("extent"),
        "identifiers": meta.get("identifiers") or {},
        "subjects": meta.get("subjects") or [],
        "classifications": meta.get("classifications") or [],
        "abstract": meta.get("abstract"),
        "notes": meta.get("notes") or [],
        "language": meta.get("language"),
        "languages": meta.get("languages") or [],
        "record_origin": meta.get("record_origin"),
        "record_creation_date": meta.get("record_creation_date"),
        "total_pages_available": total_available,
        "pages_downloaded": succeeded,
        "catalogue_url": meta.get("url"),
        "pages": pages_info,
    }
    metadata_file = out_path / "metadata.json"
    metadata_file.write_text(json.dumps(metadata_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadata saved to: {metadata_file}")

    print()
    print(f"Download complete: {succeeded} succeeded, {failed} failed.")
    if failed > 0:
        print(f"WARNING: {failed} image(s) could not be downloaded.", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = create_parser(add_help=True)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
