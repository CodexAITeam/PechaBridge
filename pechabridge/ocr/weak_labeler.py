"""Weak OCR labeling pipeline for patch datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd
from PIL import Image

from .backends import OCRBackend, OCRResult, TesseractBackend, VLMBackendStub
from .preprocess import PreprocessConfig, preprocess_patch_image

LOGGER = logging.getLogger("weak_ocr_labeler")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_config_hash(payload: Mapping[str, Any]) -> str:
    """Stable sha256 hash for reproducibility metadata."""
    normalized = _jsonable(dict(payload or {}))
    txt = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def belongs_to_shard(patch_id: int, shard_id: int, num_shards: int) -> bool:
    """Deterministic shard assignment."""
    ns = max(1, int(num_shards))
    sid = int(shard_id)
    if sid < 0 or sid >= ns:
        raise ValueError(f"Invalid shard_id={sid} for num_shards={ns}")
    return int(patch_id) % ns == sid


def filter_patch_ids_for_resume(
    patch_ids: Sequence[int],
    *,
    existing_patch_ids: Set[int],
    resume: bool,
    overwrite: bool,
) -> List[int]:
    """Filter patch_ids according to resume/overwrite policy."""
    out: List[int] = []
    for pid in patch_ids:
        patch_id = int(pid)
        if bool(overwrite):
            out.append(patch_id)
            continue
        if bool(resume) and patch_id in existing_patch_ids:
            continue
        out.append(patch_id)
    return out


def load_existing_patch_ids(out_parquet: Path) -> Set[int]:
    """Read existing patch ids from weak_ocr parquet."""
    p = Path(out_parquet).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return set()
    try:
        df = pd.read_parquet(p, columns=["patch_id"])
    except Exception:
        df = pd.read_parquet(p)
    if df is None or df.empty or "patch_id" not in df.columns:
        return set()
    out: Set[int] = set()
    for raw in df["patch_id"].tolist():
        try:
            out.add(int(raw))
        except Exception:
            continue
    return out


def _normalize_id(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _sanitize_id(value: Any) -> str:
    txt = _normalize_id(value)
    if not txt:
        return "unknown"
    out: List[str] = []
    for ch in txt:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _resolve_patch_path(dataset_dir: Path, row: Mapping[str, Any]) -> Optional[Path]:
    for col in ("patch_image_path", "patch_path"):
        raw = str(row.get(col, "") or "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        resolved = p.resolve() if p.is_absolute() else (dataset_dir / p).resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    try:
        patch_id = int(row.get("patch_id", -1))
        line_id = int(row.get("line_id", -1))
        scale_w = int(row.get("scale_w", -1))
    except Exception:
        return None
    if patch_id < 0 or line_id < 0 or scale_w <= 0:
        return None

    doc = _sanitize_id(row.get("doc_id", ""))
    page = _sanitize_id(row.get("page_id", ""))
    derived = (
        dataset_dir
        / "patches"
        / f"doc={doc}"
        / f"page={page}"
        / f"line={line_id}"
        / f"scale={scale_w}"
        / f"patch_{patch_id}.png"
    ).resolve()
    if derived.exists() and derived.is_file():
        return derived
    return None


@dataclass(frozen=True)
class OutputConfig:
    min_confidence: float = 0.0
    store_raw: bool = False

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "OutputConfig":
        p = dict(payload or {})
        return OutputConfig(
            min_confidence=float(p.get("min_confidence", 0.0)),
            store_raw=bool(p.get("store_raw", False)),
        )


@dataclass(frozen=True)
class PerformanceConfig:
    batch_read: int = 256
    num_workers: int = 8

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "PerformanceConfig":
        p = dict(payload or {})
        return PerformanceConfig(
            batch_read=max(1, int(p.get("batch_read", 256))),
            num_workers=max(1, int(p.get("num_workers", 8))),
        )


@dataclass(frozen=True)
class WeakOCRConfig:
    backend_name: str = "tesseract"
    backend_config: Mapping[str, Any] = field(default_factory=dict)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @staticmethod
    def from_dict(
        payload: Mapping[str, Any],
        *,
        backend_override: Optional[str] = None,
        num_workers_override: Optional[int] = None,
    ) -> "WeakOCRConfig":
        p = dict(payload or {})
        backend_payload = dict(p.get("backend") or {})
        backend_name = str(backend_override or backend_payload.get("name", "tesseract")).strip().lower()
        backend_cfg = {k: v for k, v in backend_payload.items() if str(k) != "name"}

        perf = PerformanceConfig.from_dict(p.get("performance") or {})
        if num_workers_override is not None and int(num_workers_override) > 0:
            perf = PerformanceConfig(batch_read=perf.batch_read, num_workers=int(num_workers_override))

        return WeakOCRConfig(
            backend_name=backend_name,
            backend_config=backend_cfg,
            preprocess=PreprocessConfig.from_dict(p.get("preprocess") or {}),
            output=OutputConfig.from_dict(p.get("output") or {}),
            performance=perf,
        )


def _build_backend(backend_name: str, cfg: Mapping[str, Any]) -> OCRBackend:
    name = str(backend_name or "").strip().lower()
    payload = dict(cfg or {})
    if name == "tesseract":
        return TesseractBackend(
            tesseract_cmd=payload.get("tesseract_cmd"),
            lang=str(payload.get("lang", "bod")),
            oem=int(payload.get("oem", 1)),
            psm=int(payload.get("psm", 6)),
            extra_config=str(payload.get("extra_config", "")),
        )
    if name == "vlm":
        return VLMBackendStub(config=payload)
    raise ValueError(f"Unsupported OCR backend: {backend_name}")


@dataclass
class _RateLimiter:
    qps: float = 0.0
    _next_allowed: float = 0.0

    def wait(self) -> None:
        rate = float(self.qps)
        if rate <= 0.0:
            return
        now = time.monotonic()
        if now < self._next_allowed:
            time.sleep(self._next_allowed - now)
            now = time.monotonic()
        self._next_allowed = max(self._next_allowed, now) + (1.0 / rate)


def _short_error(exc: Exception, limit: int = 500) -> str:
    msg = str(exc).strip()
    if len(msg) <= limit:
        return msg
    return msg[: max(0, limit - 3)] + "..."


_WORKER_BACKEND: Optional[OCRBackend] = None
_WORKER_PREPROCESS: Optional[PreprocessConfig] = None
_WORKER_STORE_RAW: bool = False
_WORKER_RATE_LIMITER: Optional[_RateLimiter] = None


def _worker_init(
    backend_name: str,
    backend_cfg: Mapping[str, Any],
    preprocess_cfg: Mapping[str, Any],
    store_raw: bool,
    rate_limit_qps: float,
) -> None:
    global _WORKER_BACKEND, _WORKER_PREPROCESS, _WORKER_STORE_RAW, _WORKER_RATE_LIMITER
    _WORKER_BACKEND = _build_backend(backend_name, backend_cfg)
    _WORKER_PREPROCESS = PreprocessConfig.from_dict(preprocess_cfg)
    _WORKER_STORE_RAW = bool(store_raw)
    _WORKER_RATE_LIMITER = _RateLimiter(qps=float(rate_limit_qps))


def _error_row(task: Mapping[str, Any], *, lang_used: str, error_code: str, error_msg: str) -> Dict[str, Any]:
    return {
        "patch_id": int(task.get("patch_id", -1)),
        "doc_id": str(task.get("doc_id", "")),
        "page_id": str(task.get("page_id", "")),
        "line_id": int(task.get("line_id", -1)),
        "scale_w": int(task.get("scale_w", -1)),
        "text": "",
        "confidence": float(0.0),
        "char_count": int(0),
        "word_count": int(0),
        "lang_used": str(lang_used or ""),
        "backend": str(task.get("backend", "")),
        "preprocess_hash": str(task.get("preprocess_hash", "")),
        "ocr_config_hash": str(task.get("ocr_config_hash", "")),
        "error_code": str(error_code or ""),
        "error_msg": str(error_msg or ""),
        "raw_json": None,
    }


def _process_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    if _WORKER_BACKEND is None or _WORKER_PREPROCESS is None:
        raise RuntimeError("OCR worker was not initialized.")

    lang_fallback = str(getattr(_WORKER_BACKEND, "lang_used", "") or "")
    try:
        if _WORKER_RATE_LIMITER is not None:
            _WORKER_RATE_LIMITER.wait()

        image_path = Path(str(task.get("image_path", ""))).expanduser().resolve()
        with Image.open(image_path) as im:
            original = im.convert("RGB")
        preprocessed = preprocess_patch_image(original, _WORKER_PREPROCESS)
        result: OCRResult = _WORKER_BACKEND.ocr_image(preprocessed, meta=task)
        text = str(result.text or "")
        conf = float(result.confidence)
        if np.isfinite(conf):
            conf = float(max(0.0, min(1.0, conf)))
        else:
            conf = float("nan")

        tokens = list(result.tokens or [])
        char_count = int(result.char_count) if int(result.char_count) > 0 else int(len(text))
        if int(result.word_count) > 0:
            word_count = int(result.word_count)
        elif tokens:
            word_count = int(len(tokens))
        else:
            word_count = int(len([w for w in text.split() if w]))
        lang_used = str(result.lang_used or lang_fallback)

        return {
            "patch_id": int(task["patch_id"]),
            "doc_id": str(task.get("doc_id", "")),
            "page_id": str(task.get("page_id", "")),
            "line_id": int(task.get("line_id", -1)),
            "scale_w": int(task.get("scale_w", -1)),
            "text": text,
            "confidence": conf,
            "char_count": char_count,
            "word_count": word_count,
            "lang_used": lang_used,
            "backend": str(task.get("backend", "")),
            "preprocess_hash": str(task.get("preprocess_hash", "")),
            "ocr_config_hash": str(task.get("ocr_config_hash", "")),
            "error_code": None,
            "error_msg": None,
            "raw_json": (result.raw_json if _WORKER_STORE_RAW else None),
        }
    except Exception as exc:
        return _error_row(
            task,
            lang_used=lang_fallback,
            error_code=type(exc).__name__,
            error_msg=_short_error(exc),
        )


def _typed_output_dataframe(rows: Iterable[Mapping[str, Any]], *, include_raw: bool) -> pd.DataFrame:
    base_cols = [
        "patch_id",
        "doc_id",
        "page_id",
        "line_id",
        "scale_w",
        "text",
        "confidence",
        "char_count",
        "word_count",
        "lang_used",
        "backend",
        "preprocess_hash",
        "ocr_config_hash",
        "error_code",
        "error_msg",
    ]
    cols = list(base_cols)
    if include_raw:
        cols.append("raw_json")

    rows_list = [dict(r) for r in rows]
    if not rows_list:
        data: Dict[str, pd.Series] = {
            "patch_id": pd.Series([], dtype="int64"),
            "doc_id": pd.Series([], dtype="string"),
            "page_id": pd.Series([], dtype="string"),
            "line_id": pd.Series([], dtype="int32"),
            "scale_w": pd.Series([], dtype="int16"),
            "text": pd.Series([], dtype="string"),
            "confidence": pd.Series([], dtype="float32"),
            "char_count": pd.Series([], dtype="int32"),
            "word_count": pd.Series([], dtype="int32"),
            "lang_used": pd.Series([], dtype="string"),
            "backend": pd.Series([], dtype="string"),
            "preprocess_hash": pd.Series([], dtype="string"),
            "ocr_config_hash": pd.Series([], dtype="string"),
            "error_code": pd.Series([], dtype="string"),
            "error_msg": pd.Series([], dtype="string"),
        }
        if include_raw:
            data["raw_json"] = pd.Series([], dtype="string")
        return pd.DataFrame(data, columns=cols)

    df = pd.DataFrame(rows_list)
    for col in cols:
        if col not in df.columns:
            if col in {"confidence"}:
                df[col] = np.nan
            elif col in {"error_code", "error_msg", "raw_json"}:
                df[col] = None
            elif col in {"patch_id", "line_id", "scale_w", "char_count", "word_count"}:
                df[col] = 0
            else:
                df[col] = ""

    for col in ["patch_id", "line_id", "scale_w", "char_count", "word_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["patch_id"] = df["patch_id"].fillna(-1).astype("int64")
    df["line_id"] = df["line_id"].fillna(-1).astype("int32")
    df["scale_w"] = df["scale_w"].fillna(-1).astype("int16")
    df["char_count"] = df["char_count"].fillna(0).astype("int32")
    df["word_count"] = df["word_count"].fillna(0).astype("int32")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").astype("float32")

    for col in ["doc_id", "page_id", "text", "lang_used", "backend", "preprocess_hash", "ocr_config_hash"]:
        df[col] = df[col].fillna("").astype("string")
    df["error_code"] = df["error_code"].astype("string")
    df["error_msg"] = df["error_msg"].astype("string")
    if include_raw:
        df["raw_json"] = df["raw_json"].astype("string")

    return df[cols]


def _write_debug_dump(
    *,
    dataset_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    patch_to_image_path: Mapping[int, str],
    preprocess_cfg: PreprocessConfig,
    limit: int,
) -> int:
    take = max(0, int(limit))
    if take <= 0:
        return 0

    good = []
    for row in rows:
        if str(row.get("error_code", "") or "").strip():
            continue
        try:
            patch_id = int(row.get("patch_id", -1))
        except Exception:
            continue
        if patch_id in patch_to_image_path:
            good.append((patch_id, row))
    if not good:
        return 0

    good.sort(key=lambda x: x[0])
    out_dir = (dataset_dir / "debug" / "weak_ocr").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, (patch_id, row) in enumerate(good[:take], start=1):
        image_path = Path(str(patch_to_image_path[patch_id])).expanduser().resolve()
        try:
            with Image.open(image_path) as im:
                original = im.convert("RGB")
            prep = preprocess_patch_image(original, preprocess_cfg)
            stem = f"{idx:05d}_patch_{patch_id}"
            original.save(out_dir / f"{stem}_orig.png")
            prep.save(out_dir / f"{stem}_prep.png")
            txt = (
                f"patch_id: {patch_id}\n"
                f"confidence: {row.get('confidence')}\n"
                f"error_code: {row.get('error_code')}\n"
                "text:\n"
                f"{str(row.get('text', ''))}\n"
            )
            (out_dir / f"{stem}_text.txt").write_text(txt, encoding="utf-8")
            count += 1
        except Exception:
            continue
    return int(count)


def run_weak_ocr_labeler(
    *,
    dataset_dir: Path,
    meta_path: Path,
    out_path: Path,
    config: WeakOCRConfig,
    shard_id: int = 0,
    num_shards: int = 1,
    resume: bool = False,
    overwrite: bool = False,
    debug_dump: int = 0,
) -> Dict[str, Any]:
    """Run weak OCR labeling and write output parquet."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    meta_path = Path(meta_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not meta_path.exists() or not meta_path.is_file():
        raise FileNotFoundError(f"Metadata parquet not found: {meta_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if int(num_shards) <= 0:
        raise ValueError("num_shards must be > 0")
    if int(shard_id) < 0 or int(shard_id) >= int(num_shards):
        raise ValueError(f"Invalid shard_id={shard_id} for num_shards={num_shards}")

    backend_probe = _build_backend(config.backend_name, config.backend_config)
    backend_lang = str(getattr(backend_probe, "lang_used", "") or "")
    del backend_probe

    out_exists = out_path.exists() and out_path.is_file()
    if out_exists and (not bool(resume)) and (not bool(overwrite)):
        raise FileExistsError(f"Output already exists: {out_path}. Use --resume or --overwrite.")

    existing_df: Optional[pd.DataFrame] = None
    existing_patch_ids: Set[int] = set()
    if out_exists:
        existing_df = pd.read_parquet(out_path)
        if existing_df is not None and (not existing_df.empty) and "patch_id" in existing_df.columns:
            for raw in existing_df["patch_id"].tolist():
                try:
                    existing_patch_ids.add(int(raw))
                except Exception:
                    continue

    meta_df = pd.read_parquet(meta_path)
    if meta_df is None or meta_df.empty:
        raise RuntimeError(f"No rows in metadata parquet: {meta_path}")
    if "patch_id" not in meta_df.columns:
        raise ValueError("Metadata parquet missing required column: patch_id")

    preprocess_hash = stable_config_hash(config.preprocess.to_dict())
    ocr_config_hash = stable_config_hash({"backend": config.backend_name, "config": dict(config.backend_config)})

    meta_sorted = meta_df.sort_values("patch_id", kind="mergesort")
    rows = meta_sorted.to_dict(orient="records")

    tasks: List[Dict[str, Any]] = []
    patch_to_image_path: Dict[int, str] = {}
    immediate_rows: List[Dict[str, Any]] = []

    seen_patch_ids: Set[int] = set()
    invalid_meta_rows = 0
    skipped_resume = 0
    skipped_shard = 0
    deduped = 0
    missing_paths = 0

    for row in rows:
        try:
            patch_id = int(row.get("patch_id"))
        except Exception:
            invalid_meta_rows += 1
            continue
        if patch_id in seen_patch_ids:
            deduped += 1
            continue
        seen_patch_ids.add(patch_id)

        try:
            if not belongs_to_shard(patch_id, int(shard_id), int(num_shards)):
                skipped_shard += 1
                continue
        except Exception:
            skipped_shard += 1
            continue

        if (not bool(overwrite)) and bool(resume) and patch_id in existing_patch_ids:
            skipped_resume += 1
            continue

        doc_id = _normalize_id(row.get("doc_id", ""))
        page_id = _normalize_id(row.get("page_id", ""))
        try:
            line_id = int(row.get("line_id", -1))
        except Exception:
            line_id = -1
        try:
            scale_w = int(row.get("scale_w", -1))
        except Exception:
            scale_w = -1

        task_base = {
            "patch_id": patch_id,
            "doc_id": doc_id,
            "page_id": page_id,
            "line_id": line_id,
            "scale_w": scale_w,
            "backend": config.backend_name,
            "preprocess_hash": preprocess_hash,
            "ocr_config_hash": ocr_config_hash,
        }

        image_path = _resolve_patch_path(dataset_dir, row)
        if image_path is None:
            immediate_rows.append(
                _error_row(
                    task_base,
                    lang_used=backend_lang,
                    error_code="missing_image",
                    error_msg="Could not resolve patch image path for metadata row.",
                )
            )
            missing_paths += 1
            continue

        task = dict(task_base)
        task["image_path"] = str(image_path)
        patch_to_image_path[int(patch_id)] = str(image_path)
        tasks.append(task)

    produced_rows: List[Dict[str, Any]] = list(immediate_rows)
    if tasks:
        workers = max(1, int(config.performance.num_workers))
        batch = max(1, int(config.performance.batch_read))
        rate_limit_qps = float((config.backend_config or {}).get("rate_limit_qps", 0.0) or 0.0)

        LOGGER.info(
            "Weak OCR start: backend=%s tasks=%d workers=%d shard=%d/%d",
            config.backend_name,
            len(tasks),
            workers,
            int(shard_id),
            int(num_shards),
        )

        if workers <= 1:
            _worker_init(
                config.backend_name,
                dict(config.backend_config),
                dict(config.preprocess.to_dict()),
                bool(config.output.store_raw),
                rate_limit_qps,
            )
            for task in tasks:
                produced_rows.append(_process_task(task))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=workers,
                initializer=_worker_init,
                initargs=(
                    config.backend_name,
                    dict(config.backend_config),
                    dict(config.preprocess.to_dict()),
                    bool(config.output.store_raw),
                    rate_limit_qps,
                ),
            ) as pool:
                for out_row in pool.imap_unordered(_process_task, tasks, chunksize=batch):
                    produced_rows.append(out_row)

    include_raw = bool(config.output.store_raw)
    if existing_df is not None and "raw_json" in existing_df.columns:
        include_raw = True

    new_df = _typed_output_dataframe(produced_rows, include_raw=include_raw)

    if existing_df is not None and not existing_df.empty:
        existing_typed = _typed_output_dataframe(existing_df.to_dict(orient="records"), include_raw=include_raw)
        if bool(overwrite):
            replace_ids = set(int(v) for v in new_df["patch_id"].tolist())
            if replace_ids:
                existing_typed = existing_typed[~existing_typed["patch_id"].isin(sorted(replace_ids))]
            final_df = pd.concat([existing_typed, new_df], ignore_index=True)
        else:
            final_df = pd.concat([existing_typed, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df = final_df.sort_values("patch_id", kind="mergesort")
    final_df = final_df.drop_duplicates(subset=["patch_id"], keep="last")
    final_df.to_parquet(out_path, index=False)

    dumped = _write_debug_dump(
        dataset_dir=dataset_dir,
        rows=produced_rows,
        patch_to_image_path=patch_to_image_path,
        preprocess_cfg=config.preprocess,
        limit=int(debug_dump),
    )

    conf_vals = pd.to_numeric(new_df["confidence"], errors="coerce")
    finite_conf = conf_vals[np.isfinite(conf_vals.to_numpy(dtype=np.float32, copy=False))]
    low_conf = int(
        np.sum(
            np.isfinite(conf_vals.to_numpy(dtype=np.float32, copy=False))
            & (conf_vals.to_numpy(dtype=np.float32, copy=False) < float(config.output.min_confidence))
        )
    )
    errors_in_new = int(new_df["error_code"].fillna("").astype(str).str.len().gt(0).sum())
    summary: Dict[str, Any] = {
        "backend": config.backend_name,
        "lang_used": backend_lang,
        "dataset_dir": str(dataset_dir),
        "meta_path": str(meta_path),
        "out_path": str(out_path),
        "total_meta_rows": int(len(meta_df)),
        "total_unique_patch_ids": int(len(seen_patch_ids)),
        "tasks_sent_to_ocr": int(len(tasks)),
        "rows_produced_new": int(len(new_df)),
        "rows_written_total": int(len(final_df)),
        "invalid_meta_rows": int(invalid_meta_rows),
        "deduped_meta_rows": int(deduped),
        "skipped_shard": int(skipped_shard),
        "skipped_resume": int(skipped_resume),
        "missing_image_paths": int(missing_paths),
        "error_rows_new": int(errors_in_new),
        "debug_dump_written": int(dumped),
        "preprocess_hash": preprocess_hash,
        "ocr_config_hash": ocr_config_hash,
        "confidence": {
            "min": (float(np.min(finite_conf)) if len(finite_conf) else None),
            "max": (float(np.max(finite_conf)) if len(finite_conf) else None),
            "mean": (float(np.mean(finite_conf)) if len(finite_conf) else None),
            "below_min_confidence": int(low_conf),
        },
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = [
    "OutputConfig",
    "PerformanceConfig",
    "WeakOCRConfig",
    "belongs_to_shard",
    "filter_patch_ids_for_resume",
    "load_existing_patch_ids",
    "run_weak_ocr_labeler",
    "stable_config_hash",
]
