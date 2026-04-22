"""Custom LangChain embedding wrapper for a Tibetan CLIP-style Hugging Face model."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .compat import Embeddings
from .config import EmbeddingConfig


class TibetanClipEmbeddings(Embeddings):
    """Embeds Tibetan transcript lines via a Hugging Face text encoder."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        tokenizer_kwargs = {
            "trust_remote_code": config.trust_remote_code,
        }
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
        }
        if config.cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = str(config.cache_dir)
            model_kwargs["cache_dir"] = str(config.cache_dir)

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.huggingface_model_id,
            **tokenizer_kwargs,
        )
        tokenizer_was_resized = False
        if self._tokenizer.pad_token is None:
            for token_name in ("eos_token", "sep_token", "unk_token", "bos_token"):
                fallback_token = getattr(self._tokenizer, token_name, None)
                if fallback_token is not None:
                    self._tokenizer.pad_token = fallback_token
                    break
            if self._tokenizer.pad_token is None:
                self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                tokenizer_was_resized = True

        self._device = torch.device(config.device)
        self._model = AutoModel.from_pretrained(
            config.huggingface_model_id,
            **model_kwargs,
        ).to(self._device)
        if tokenizer_was_resized:
            self._model.resize_token_embeddings(len(self._tokenizer))
        self._model.eval()
        self._embedding_size = self._infer_embedding_size()

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_texts(texts, prefix=self.config.document_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_texts([text], prefix=self.config.query_prefix)[0]

    def _embed_texts(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in self._batched(texts, self.config.batch_size):
            prefixed_batch = [f"{prefix}{text}" for text in batch]
            encoded = self._tokenizer(
                prefixed_batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
                features = self._extract_features(encoded, outputs)
                if self.config.normalize_embeddings:
                    features = F.normalize(features, p=2, dim=1)
            vectors.extend(features.detach().cpu().tolist())
        return vectors

    def _extract_features(self, encoded_inputs: dict[str, torch.Tensor], outputs: object) -> torch.Tensor:
        if hasattr(self._model, "get_text_features"):
            return self._model.get_text_features(**encoded_inputs)

        text_embeds = getattr(outputs, "text_embeds", None)
        if text_embeds is not None:
            return text_embeds

        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError(
                "Unable to extract embeddings from the configured Hugging Face model. "
                "Expected one of get_text_features, text_embeds, pooler_output, or last_hidden_state."
            )

        attention_mask = encoded_inputs.get("attention_mask")
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _infer_embedding_size(self) -> int:
        probe_vector = self.embed_query("བོད")
        return len(probe_vector)

    @staticmethod
    def _batched(values: list[str], batch_size: int) -> Iterable[list[str]]:
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]
