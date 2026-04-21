from __future__ import annotations

from pechabridge.cli.batch_ocr import create_parser


def test_batch_ocr_parser_accepts_bdrc_parity_flags() -> None:
    parser = create_parser(add_help=False)
    args = parser.parse_args(
        [
            "--input-dir",
            "/tmp/in",
            "--layout-engine",
            "bdrc_line",
            "--ocr-engine",
            "bdrc_ocr",
            "--bdrc-line-no-merge-lines",
            "--bdrc-line-no-use-tps",
            "--bdrc-line-tps-threshold",
            "0.4",
            "--bdrc-line-k-factor",
            "2.9",
            "--bdrc-line-bbox-tolerance",
            "3.7",
        ]
    )

    assert args.bdrc_line_merge_lines is False
    assert args.bdrc_line_use_tps is False
    assert float(args.bdrc_line_tps_threshold) == 0.4
    assert float(args.bdrc_line_k_factor) == 2.9
    assert float(args.bdrc_line_bbox_tolerance) == 3.7
