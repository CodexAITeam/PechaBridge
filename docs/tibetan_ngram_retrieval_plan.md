# Tibetan n-gram Retrieval Plan (ohne echte Paare, mit späterem Alignment)

## Ziel
Text-Query (n-gram, Tibetan) soll relevante Seiten/Zeilen aus Pecha-Bildern finden, obwohl zunächst keine echten Bild-Text-Paare vorliegen.

## Kurzstrategie
1. Bild-Encoder auf Stabi-Bildern selbstüberwacht vortrainieren.
2. Text-Encoder auf Tibetan-Texten vortrainieren (char/subword-fokussiert, Unicode-sicher).
3. Mit synthetischen und pseudo-gelabelten Paaren einen gemeinsamen Embedding-Raum lernen.
4. Retrieval-Index bauen und iterativ mit besseren Paaren nachschärfen.

---

## Modellwahl

## 1) Image Encoder (Empfehlung)
### Primär: `DINOv2 ViT-B/14` als Backbone
Warum:
- Sehr stark für Dokument-/Struktur-Features ohne Labels.
- Stabil beim Self-Supervised-Finetuning auf eigenen Scans.
- Gute Basis für spätere Zeilen-/Region-Embeddings.

Praktische Konfiguration:
- Input: zuerst `518` oder `448`, später optional `672` für feine Druckdetails.
- Pooling: CLS + mean pooling testen, meist mean stabiler für Retrieval.
- Output-Dim für Retrieval-Head: `768 -> 256` oder `384` Projektion.

Alternative:
- `SigLIP`/`CLIP`-Vision-Backbone, falls du schon CLIP-Tooling nutzt.

### Wichtig: Page Encoder vs. Line Encoder
- Für echtes **n-gram retrieval** reicht ein reiner Page-Encoder meist nicht.
- Empfohlen ist ein **Line Encoder** (oder Region-Encoder) als primärer Retrieval-Encoder.
- Praktischer Kompromiss:
  - erst Page-level self-supervised pretraining,
  - dann Line-level fine-tuning auf synthetischen Line-Crops (mit bekanntem Text),
  - optional zwei Heads behalten:
    - Page-Head für coarse recall,
    - Line-Head für präzises final retrieval.

## 2) Text Encoder (Empfehlung)
### Primär: Char-/Subword-Transformer mit ByT5-ähnlichem Ansatz
Warum:
- Tibetan hat komplexe Unicode-Zeichenfolgen und Kombinationszeichen.
- Byte/Char-nahe Modelle sind robuster als reine ASCII-/Word-Pipelines.
- Funktioniert besser bei n-gram retrieval, wenn Orthographie variiert.

Praktische Optionen:
1. **ByT5-small/base** (byte-level, Unicode-robust) + contrastive Head.
2. **SentencePiece-Unigram** auf Tibetan-Korpus (8k-16k vocab) + Transformer Encoder.
3. Hybrid: Char CNN/Transformer + Subword-Encoder fusionieren (optional später).

Empfohlene Startwahl:
- Wenn schnell robust gewünscht: **ByT5-small**.
- Wenn du volle Kontrolle willst: eigenes **SentencePiece-Unigram + Encoder**.

---

## Tibetan Unicode / Nicht-ASCII: Muss-Regeln

## 1) Normalisierung
- Verwende **NFC** oder **NFKC** konsistent (nicht mischen).
- Pipeline-weit genau eine Norm erzwingen (Training + Inferenz + Indexing).

## 2) Segmentierung
- Nicht auf ASCII-Grenzen verlassen.
- Tibetan Trenner (`་`, `།`) explizit behandeln.
- n-grams auf **Unicode-Grapheme-Cluster** oder normalisierten Zeichenfolgen bauen, nicht auf Bytestring-Slices.

## 3) Tokenizer-Training
- SentencePiece direkt auf normalisiertem Tibetan trainieren.
- Kontrolliere, dass seltene kombinierte Sequenzen nicht zerstört werden.

## 4) Retrieval-Vorverarbeitung
- Gleiche Normalize/Canonicalize-Logik für:
  - Korpusindex
  - Query
  - Eval-Daten
- Sonst sinkt Recall künstlich stark.

---

## Phase A (jetzt, ohne Paare): Separate Vortrainings

## A1) Image Encoder Pretraining (self-supervised)
Daten:
- Stabi-Rohbilder (`sbb_images`) + crops/lines (falls vorhanden).

Methoden:
- DINO/SimCLR-style augmentations (crop, blur, contrast, JPEG artifacts).
- Dokument-spezifische Augs moderat halten (nicht zu starke geometrische Verzerrung).

Output:
- Checkpoints: `image_encoder_best.pt`, `image_encoder_last.pt`
- Embedding-Head checkpoint (für Retrieval kompatibel).

## A2) Text Encoder Pretraining (unpaired text)
Daten:
- Alle verfügbaren Tibetan-Texte (auch klein ist ok).
- Optional synthetisch erzeugte Tibetan-Zeilen für Datenaufblähung.

Methoden:
- MLM/denoising oder contrastive text objectives.
- Ziel: robuste orthographische und n-gram-nahe Repräsentationen.

Output:
- `text_encoder_best.pt`
- Tokenizer-Artefakte (`sentencepiece.model` oder byte-level config).

---

## Phase B (Brücke ohne echte Paare): Weak Supervision Alignment

## B1) Synthetic Paare erzeugen (wichtig)
- Rendered line image + exakt bekannter Text.
- Variiere Schrift, Dicke, Noise, Background, Scan-Artefakte.
- Nutze vorhandene Texture-LoRA für realistischere Bilder.

## B2) Pseudo-Paare auf Realbildern
- Segmentierung/Line detector auf Stabi.
- Schwache Labels über:
  - heuristische Zuordnung,
  - VLM/OCR-Teacher (noisy),
  - nearest-neighbor propagation.

## B3) Joint Dual-Encoder Training
- Image/Text jeweils eigener Encoder + Projektionskopf in gemeinsamen Raum.
- Loss: **InfoNCE contrastive** (+ hard negatives).
- Batch negatives + mined hard negatives aus ähnlichen Seiten.

Output:
- `dual_encoder_best` (image/text + projection heads)
- retrieval-ready embedding pipeline.

---

## Phase C: Retrieval-System

## C1) Index-Aufbau
- Segmentiere Seite -> Zeilen/Blöcke.
- Bilde Image-Embeddings, speichere mit Metadaten (PPN, Seite, bbox).
- FAISS/HNSW Index bauen.

### Pflicht: Source/Provenance pro Treffer speichern
Nur \"ähnliches Bild\" ist nicht ausreichend. Jeder Index-Eintrag muss Quelle und Position mitführen.

Empfohlenes Record-Schema (pro Line/Region):

```json
{
  "id": "ppn337138764X_p0042_l0017",
  "ppn": "337138764X",
  "page_id": "0042",
  "image_path": "sbb_images/default_abc123.jpg",
  "bbox_xyxy": [312, 840, 1580, 932],
  "split": "real",
  "source_url": "https://content.staatsbibliothek-berlin.de/...",
  "viewer_url": "https://digital.staatsbibliothek-berlin.de/werkansicht?PPN=337138764X&PHYSID=PHYS_0042"
}
```

Bei Retrieval gibst du genau dieses Record zurück, damit der Treffer zitierbar und nachvollziehbar ist.

## C2) Query-Pipeline
- Query normalisieren (Unicode-Regeln).
- n-grams erzeugen.
- Text-Embeddings rechnen.
- ANN Search + optional reranking.
- Output immer mit `ppn`, `page_id`, `bbox_xyxy`, `viewer_url`.

### Konkretes Ausgabeformat: `retrieval_result.json`
Empfohlenes Top-K Response-Schema:

```json
{
  "query": {
    "raw": "བཀྲ་ཤིས",
    "normalized": "བཀྲ་ཤིས",
    "normalization": "NFC",
    "ngrams": ["བཀ", "ཀྲ", "ྲ་", "་ཤ", "ཤི", "ིས"]
  },
  "model_info": {
    "image_encoder": "dinov2_vitb14_line_head_v3",
    "text_encoder": "byt5_small_retrieval_v2",
    "index_id": "faiss_lines_2026_02_13"
  },
  "top_k": 10,
  "results": [
    {
      "rank": 1,
      "score": 0.8731,
      "distance": 0.1269,
      "id": "ppn337138764X_p0042_l0017",
      "ppn": "337138764X",
      "page_id": "0042",
      "line_id": "l0017",
      "bbox_xyxy": [312, 840, 1580, 932],
      "image_path": "sbb_images/default_abc123.jpg",
      "crop_path": "index_crops/ppn337138764X_p0042_l0017.png",
      "source_url": "https://content.staatsbibliothek-berlin.de/...",
      "viewer_url": "https://digital.staatsbibliothek-berlin.de/werkansicht?PPN=337138764X&PHYSID=PHYS_0042",
      "metadata": {
        "split": "real",
        "collection": "sbb",
        "index_version": "2026-02-13"
      }
    }
  ]
}
```

Hinweise:
- `score` und `distance` beide speichern (für Debugging/Monitoring).
- `id` muss stabil sein (`ppn + page + line`), damit Treffer reproduzierbar bleiben.
- `viewer_url` direkt klickbar halten (für manuelle Verifikation).
- Optional später Felder ergänzen:
  - `ocr_text` (falls verfügbar),
  - `rerank_score`,
  - `evidence_spans` für explainability.

Code-Referenz:
- Pydantic-Schema liegt unter `tibetan_utils/retrieval_schema.py`
- Zentrales Root-Modell: `RetrievalResult`

## C3) Evaluierung
Metriken:
- Recall@K (K=1,5,10,50)
- MRR
- nDCG

Eval-Sets:
- Kleine manuelle Goldmenge (auch 200-500 Queries reichen für Start).
- Mix aus kurzen und längeren n-grams.

---

## Konkreter Compute-Plan (1,5 Wochen auf 4x V100)

## Woche 1
1. Tag 1-2: Unicode-/Tokenizer-Pipeline finalisieren, Eval-Protokoll fixieren.
2. Tag 2-4: Image encoder self-supervised pretraining.
3. Tag 3-5: Text encoder pretraining.
4. Tag 5-7: Synthetic paired data groß erzeugen.

## Woche 2 (Restzeit)
1. Tag 8-10: Dual-encoder alignment (synthetic + pseudo pairs).
2. Tag 10-11: Hard-negative mining + zweiter Alignment-Run.
3. Tag 11: Export `best` + `fast` Varianten und Index-Prototyp.

---

## Was du zwingend sichern solltest (für 1070 Ti danach)
- `image_encoder_best`
- `text_encoder_best`
- `dual_encoder_best`
- Tokenizer + Unicode normalization code (versioniert)
- Retrieval index builder scripts + eval scripts
- Kleinere distilled/fast Variante für lokale Iterationen

---

## Risiken und Gegenmaßnahmen
- Risiko: Kein echtes Alignment ohne Paare.
  - Gegenmaßnahme: synthetic + pseudo pairs früh einbauen.
- Risiko: Unicode mismatch zwischen train/infer/index.
  - Gegenmaßnahme: eine zentrale Normalisierungsfunktion, überall identisch nutzen.
- Risiko: Overfit auf Synthetic Look.
  - Gegenmaßnahme: realistische Texture/scan-Augmentations + echte SBB crops beim Fine-Tuning.

---

## Entscheidungsempfehlung (kurz)
- **Image Encoder:** DINOv2 ViT-B/14 (self-supervised auf Stabi).
- **Text Encoder:** ByT5-small (Unicode-robust) oder SP-Unigram + Transformer.
- **Alignment:** Dual-Encoder mit InfoNCE auf synthetic + pseudo pairs.
- **Priorität:** erst robuste Unicode/Text-Pipeline, dann Alignment.
