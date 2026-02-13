"""
Pydantic schemas for Tibetan n-gram retrieval API responses.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, model_validator
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "tibetan_utils.retrieval_schema requires pydantic>=2. "
        "Install with: pip install pydantic"
    ) from exc


NormalizationType = Literal["NFC", "NFKC", "NONE"]


class QueryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str = Field(..., description="User-provided query string.")
    normalized: str = Field(..., description="Normalized query string used for encoding.")
    normalization: NormalizationType = Field(
        ...,
        description="Normalization strategy applied to query and corpus.",
    )
    ngrams: List[str] = Field(default_factory=list, description="Derived n-grams from normalized query.")


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_encoder: str = Field(..., description="Image encoder identifier/version.")
    text_encoder: str = Field(..., description="Text encoder identifier/version.")
    index_id: str = Field(..., description="ANN index identifier/version.")


class RetrievalMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    split: Optional[str] = Field(default=None, description="Dataset split, e.g. real/synth.")
    collection: Optional[str] = Field(default=None, description="Collection origin, e.g. sbb.")
    index_version: Optional[str] = Field(default=None, description="Index build date or semantic version.")


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., ge=1)
    score: float = Field(..., description="Higher-is-better similarity score.")
    distance: Optional[float] = Field(default=None, description="Lower-is-better ANN distance.")
    id: str = Field(..., description="Stable retrieval item id.")

    ppn: str = Field(..., description="SBB PPN.")
    page_id: str = Field(..., description="Page identifier in source document.")
    line_id: Optional[str] = Field(default=None, description="Line/region id within page.")
    bbox_xyxy: List[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")

    image_path: str = Field(..., description="Path to source page image.")
    crop_path: Optional[str] = Field(default=None, description="Optional path to indexed crop/line image.")
    source_url: Optional[str] = Field(default=None, description="Source image URL (if known).")
    viewer_url: Optional[str] = Field(default=None, description="Viewer URL for human verification.")

    metadata: RetrievalMetadata = Field(default_factory=RetrievalMetadata)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: QueryPayload
    model_info: ModelInfo
    top_k: int = Field(..., ge=1)
    results: List[RetrievalHit] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranking(self) -> "RetrievalResult":
        if len(self.results) > self.top_k:
            raise ValueError("results length must be <= top_k")
        ranks = [item.rank for item in self.results]
        if len(set(ranks)) != len(ranks):
            raise ValueError("result ranks must be unique")
        if ranks and ranks != sorted(ranks):
            raise ValueError("result ranks must be sorted ascending")
        return self

    def to_dict(self) -> Dict:
        return self.model_dump()


__all__ = [
    "NormalizationType",
    "QueryPayload",
    "ModelInfo",
    "RetrievalMetadata",
    "RetrievalHit",
    "RetrievalResult",
]

