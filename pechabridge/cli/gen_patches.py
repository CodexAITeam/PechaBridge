"""CLI entrypoint for generating line sub-patch training data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from pechabridge.data_gen.patch_sampler import PatchGenConfig, generate_patch_dataset

LOGGER = logging.getLogger("gen_patches")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Build parser for patch generation command."""
    parser = argparse.ArgumentParser(
        description="Generate multi-scale line sub-patch dataset with Option-A neighborhood metadata.",
        add_help=add_help,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/patch_gen.yaml",
        help="Path to YAML config. CLI flags override YAML values.",
    )
    parser.add_argument("--model", type=str, default=None, help="YOLO model path for textbox detection.")
    parser.add_argument("--input-dir", type=str, default=None, help="Input page image root directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dataset directory.")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size.")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cpu/cuda:0/mps).")
    parser.add_argument(
        "--no-samples",
        "--no_samples",
        dest="no_samples",
        type=int,
        default=None,
        help="Randomly sample at most N pages from input_dir (0 = all).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--target-height", type=int, default=None, help="Normalized line height.")
    parser.add_argument("--widths", type=str, default=None, help="Comma-separated patch widths, e.g. 256,384,512")
    parser.add_argument("--overlap", type=float, default=None, help="Dense sliding overlap in [0,1).")
    parser.add_argument("--jitter-frac", type=float, default=None, help="Aligned window jitter fraction.")
    parser.add_argument("--disable-jitter", action="store_true", help="Disable aligned-window jitter.")
    parser.add_argument("--rmin", type=float, default=None, help="Minimum ink ratio threshold.")
    parser.add_argument("--rmax", type=float, default=None, help="Maximum ink ratio threshold (<=0 disables).")
    parser.add_argument("--sigma-profile", type=float, default=None, help="Horizontal profile smoothing sigma.")
    parser.add_argument("--min-dist-px", type=int, default=None, help="Boundary minima min distance in px.")
    parser.add_argument("--min-dist-frac", type=float, default=None, help="Boundary minima min distance as fraction of target height.")
    parser.add_argument("--prominence", type=float, default=None, help="Minima prominence (relative if <=1).")
    parser.add_argument("--sigma-frac", type=float, default=None, help="Boundary score sigma fraction of scale width.")
    parser.add_argument("--n-per-line-per-scale", type=int, default=None, help="Sample count per line and scale.")
    parser.add_argument("--p-aligned", type=float, default=None, help="Probability to sample from aligned candidates.")
    parser.add_argument("--line-min-height", type=int, default=None, help="Classical line segmentation minimum line height.")
    parser.add_argument("--line-projection-smooth", type=int, default=None, help="Vertical profile smoothing rows.")
    parser.add_argument("--line-projection-threshold-rel", type=float, default=None, help="Vertical profile threshold.")
    parser.add_argument("--line-merge-gap-px", type=int, default=None, help="Vertical line run merge gap.")
    parser.add_argument("--use-clahe", action="store_true", help="Enable CLAHE before ink-map extraction.")
    parser.add_argument("--clahe-clip-limit", type=float, default=None, help="CLAHE clip limit.")
    parser.add_argument("--clahe-tile-grid-size", type=int, default=None, help="CLAHE tile grid size.")
    parser.add_argument("--binarize-ink", action="store_true", help="Binarize gray image before ink map.")
    parser.add_argument("--binarize-mode", type=str, default=None, choices=["otsu", "adaptive", "fixed"], help="Binarization mode.")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="Fixed threshold in [0,1] when --binarize-mode fixed.")
    parser.add_argument("--debug-dump", type=int, default=None, help="Dump N debug overlays under out_dataset/debug.")
    return parser


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse config YAML: {path}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return payload


def _parse_widths(raw: Any) -> List[int]:
    if isinstance(raw, (list, tuple)):
        vals = []
        for v in raw:
            try:
                vals.append(int(v))
            except Exception:
                continue
        return [int(v) for v in vals if int(v) > 1]
    txt = str(raw or "")
    vals = []
    for part in txt.split(","):
        tok = part.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except Exception:
            continue
    return [int(v) for v in vals if int(v) > 1]


def _resolve_config(args: argparse.Namespace) -> PatchGenConfig:
    cfg_path = Path(args.config).expanduser().resolve()
    yaml_cfg = _load_yaml_config(cfg_path)

    def pick(key: str, cli_value: Any, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if key in yaml_cfg:
            return yaml_cfg[key]
        return default

    widths_raw = pick("widths", args.widths, [256, 384, 512])
    widths = _parse_widths(widths_raw)
    if not widths:
        widths = [256, 384, 512]

    jitter_enabled = bool(pick("jitter_enabled", None, True))
    if bool(args.disable_jitter):
        jitter_enabled = False

    use_clahe = bool(pick("use_clahe", None, False)) or bool(args.use_clahe)
    binarize_ink = bool(pick("binarize_ink", None, False)) or bool(args.binarize_ink)

    cfg = PatchGenConfig(
        model_path=str(pick("model_path", args.model, "")),
        input_dir=str(pick("input_dir", args.input_dir, "")),
        output_dir=str(pick("output_dir", args.output_dir, "")),
        conf=float(pick("conf", args.conf, 0.25)),
        imgsz=int(pick("imgsz", args.imgsz, 1024)),
        device=str(pick("device", args.device, "")),
        seed=int(pick("seed", args.seed, 42)),
        no_samples=int(pick("no_samples", args.no_samples, 0)),
        target_height=int(pick("target_height", args.target_height, 112)),
        widths=widths,
        overlap=float(pick("overlap", args.overlap, 0.5)),
        jitter_enabled=bool(jitter_enabled),
        jitter_frac=float(pick("jitter_frac", args.jitter_frac, 0.04)),
        rmin=float(pick("rmin", args.rmin, 0.01)),
        rmax=float(pick("rmax", args.rmax, 1.0)),
        sigma_profile=float(pick("sigma_profile", args.sigma_profile, 2.0)),
        min_dist_frac=float(pick("min_dist_frac", args.min_dist_frac, 0.25)),
        min_dist_px=int(pick("min_dist_px", args.min_dist_px, 0)),
        prominence=float(pick("prominence", args.prominence, 0.08)),
        sigma_frac=float(pick("sigma_frac", args.sigma_frac, 0.15)),
        n_per_line_per_scale=int(pick("n_per_line_per_scale", args.n_per_line_per_scale, 12)),
        p_aligned=float(pick("p_aligned", args.p_aligned, 0.6)),
        line_min_height=int(pick("line_min_height", args.line_min_height, 10)),
        line_projection_smooth=int(pick("line_projection_smooth", args.line_projection_smooth, 9)),
        line_projection_threshold_rel=float(
            pick("line_projection_threshold_rel", args.line_projection_threshold_rel, 0.20)
        ),
        line_merge_gap_px=int(pick("line_merge_gap_px", args.line_merge_gap_px, 5)),
        use_clahe=bool(use_clahe),
        clahe_clip_limit=float(pick("clahe_clip_limit", args.clahe_clip_limit, 2.0)),
        clahe_tile_grid_size=int(pick("clahe_tile_grid_size", args.clahe_tile_grid_size, 8)),
        binarize_ink=bool(binarize_ink),
        binarize_mode=str(pick("binarize_mode", args.binarize_mode, "otsu")),
        fixed_threshold=float(pick("fixed_threshold", args.fixed_threshold, 0.5)),
        debug_dump=int(pick("debug_dump", args.debug_dump, 0)),
    )
    if not cfg.model_path:
        raise ValueError("Missing model path. Set --model or model_path in YAML.")
    if not cfg.input_dir:
        raise ValueError("Missing input directory. Set --input-dir or input_dir in YAML.")
    if not cfg.output_dir:
        raise ValueError("Missing output directory. Set --output-dir or output_dir in YAML.")
    return cfg


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run patch generation from parsed args."""
    cfg = _resolve_config(args)
    LOGGER.info(
        "Patch generation: model=%s input=%s output=%s widths=%s",
        cfg.model_path,
        cfg.input_dir,
        cfg.output_dir,
        ",".join(str(v) for v in cfg.widths),
    )
    summary = generate_patch_dataset(cfg)
    LOGGER.info("Done: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = create_parser(add_help=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

