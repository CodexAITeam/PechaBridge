# line_clip Dual Encoder Probe Guide

Diese Anleitung beschreibt die praktische Bewertung des `line_clip` Dual Encoders mit den neuen CLI-Tools:

- Cache-Warmup: `warm-line-clip-workbench-cache`
- Probe/Evaluation: `probe-line-clip-workbench-random-samples`

Die Tools nutzen dieselbe Modellauswahl wie die Workbench (`bestes line_clip Bundle` nach Val-Retrieval-Performance).

## 1) Voraussetzungen

- OCR-Dataset mit Split-Struktur:
  - `<dataset_root>/train/meta/lines.jsonl`
  - `<dataset_root>/val/meta/lines.jsonl`
  - `<dataset_root>/eval/meta/lines.jsonl`
  - `<dataset_root>/test/meta/lines.jsonl`
- line_clip Modelle unter `models_dir`
- Python-Umgebung mit allen Abhängigkeiten (z. B. `TibetanLayoutAnalyzer`)

Beispielpfade:

- `models_dir=/home/ubuntu/data/PechaBridge/models`
- `dataset_root=/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines`

## 2) Cache aufbauen (empfohlen vor Probes)

Der Probe-Command kann Cache selbst bauen. Für reproduzierbare und schnellere Evaluierung ist Warmup vorher sinnvoll.

### 2.1 Beide Banken (Bild+Text)

```bash
python /home/ubuntu/data/PechaBridge/cli.py warm-line-clip-workbench-cache \
  --models-dir /home/ubuntu/data/PechaBridge/models \
  --dataset-root /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines \
  --splits train,val,eval,test \
  --device cpu \
  --batch-size 16 \
  --only both \
  --progress-every-batches 50
```

### 2.2 Nur Bildbank (schneller, ausreichend für viele Text->Image Workflows)

```bash
python /home/ubuntu/data/PechaBridge/cli.py warm-line-clip-workbench-cache \
  --splits eval,test \
  --device cpu \
  --only image \
  --progress-every-batches 20
```

## 3) In-Split Probe

In-Split bedeutet `query_split == index_split` (z. B. `eval->eval`).

```bash
python /home/ubuntu/data/PechaBridge/cli.py probe-line-clip-workbench-random-samples \
  --models-dir /home/ubuntu/data/PechaBridge/models \
  --dataset-root /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines \
  --splits eval \
  --samples-per-split 200 \
  --top-k 5 \
  --device cpu \
  --summary-only
```

### Ergebnis

- Sehr kompakte Kennzahlen pro Probe:
  - `t2i@1`, `t2i@5`, `t2i_mrr`
  - `i2t@1`, `i2t@5`, `i2t_mrr`

## 4) Cross-Split Probe

Cross-Split bedeutet `query_split != index_split`, z. B. `eval->test`.

```bash
python /home/ubuntu/data/PechaBridge/cli.py probe-line-clip-workbench-random-samples \
  --cross-split eval:test,val:test \
  --samples-per-split 300 \
  --top-k 5 \
  --device cpu \
  --summary-only \
  --json-out /home/ubuntu/data/PechaBridge/output/line_clip_cross_probe.json
```

### Standardverhalten bei `--cross-split`

- Wenn `--cross-split` gesetzt ist und `--splits` auf Default steht, werden nur Cross-Split-Probes ausgeführt.
- Falls zusätzlich In-Split gewünscht:
  - `--include-in-split`

Beispiel:

```bash
python /home/ubuntu/data/PechaBridge/cli.py probe-line-clip-workbench-random-samples \
  --cross-split eval:test \
  --splits eval,test \
  --include-in-split \
  --samples-per-split 200 \
  --summary-only
```

## 5) Wie Cross-Split Metriken definiert sind

Da es bei Cross-Split kein 1:1 Index-Match gibt, wird ein Query als positiv gewertet, wenn im Index ein Eintrag mit identischem normalisiertem Text (`strip`) existiert.

- `Text -> Image`:
  - Query: Text-Embedding aus Query-Split
  - Target: Bild-Embeddings aus Index-Split
  - Positiv-Set: alle Index-Zeilen mit gleichem Text
- `Image -> Text`:
  - Query: Bild-Embedding aus Query-Split
  - Target: Text-Embeddings aus Index-Split
  - Positiv-Set: alle Index-Zeilen mit gleichem Text

Wichtig: Bei Queries ohne positives Text-Match im Index sinkt `evaluable_n`. Diese Queries gehen nicht in Recall/MRR ein.

## 6) Output lesen und bewerten

Relevante Felder im JSON:

- `results[*].kind`
  - `in_split` oder `cross_split`
- `results[*].name`
  - z. B. `eval->eval` oder `eval->test`
- `results[*].metrics.text_to_image` / `image_to_text`
  - `n`, `r1`, `r5`, `mrr`, `mean_rank`, `median_rank`
- nur `cross_split`:
  - `evaluable_n.text_to_image`
  - `evaluable_n.image_to_text`

Interpretation:

- In-Split mit kleinem `samples_per_split` kann schnell künstlich sehr hoch sein (bis 1.0).
- Für belastbare Aussage:
  - `samples_per_split` hoch setzen (mind. 200, besser 500+)
  - Cross-Split gegen held-out Index messen
  - mehrere Seeds vergleichen (`--seed`)

## 7) Cache-Hit vs Cache-Miss prüfen

Die Probe loggt pro Split:

- `checking corpus cache at: <pfad>`
- `cache HIT on disk ...` oder `cache MISS -> building ...`

Typische Ursachen für Miss trotz vorhandener Cache-Dateien:

- anderer Split (`eval` vs `test`)
- anderes `text_max_length`
- anderes `l2_normalize`
- anderes Modellbundle/Artefaktpfad
- anderer `dataset_root`

## 8) Empfohlene Evaluationsroutine (praktisch)

1. Cache warmen (mind. `eval,test`).
2. In-Split Probe als Sanity (`eval->eval`, `test->test`).
3. Cross-Split Probe als Generalisierung (`eval->test`, optional `val->test`).
4. Ergebnisse mit `--json-out` persistieren.
5. Bei Modellvergleich:
   - gleiche Parameter verwenden (`text_max_length`, `l2_normalize`, `top_k`, `samples_per_split`, `seed`)
   - gleiche Probesets (`cross-split` Paare)

## 9) Troubleshooting

### Probe baut immer neu

- Prüfe:
  - geloggten Cache-Pfad
  - `meta.json` `key` gegen aktuelle Probe-Parameter
- Danach ggf. exakt mit denselben Parametern neu warmen.

### Zu viel Konsolen-Output

- `--summary-only` verwenden
- optional weiter reduzieren:
  - `--examples 0`
  - `--progress-every-batches 0` (wenn bereits gecacht)

