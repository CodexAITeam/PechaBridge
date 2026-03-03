# DONUT OCR Trainingsanleitung (Tiny-Pretraining + Full Run)

Stand: 2026-03-03

Diese Anleitung ist ein praxisnahes Playbook fuer DONUT/TroCR-Training in PechaBridge.
Sie ist bewusst sehr detailliert und erklaert:

1. wie du schnell einen stabilen Tiny-Run aufsetzt,
2. warum dieser Tiny-Schritt gegen Empty-Predictions/Collapse hilft,
3. wie du danach sauber in einen Full-Run gehst (gray oder rgb),
4. wie du Logs richtig interpretierst.

Die Befehle beziehen sich auf `python cli.py train-donut-ocr`.

---

## 1) Zielbild und typische Failure-Modes

Beim DONUT OCR Training gibt es drei haeufige Probleme:

1. Empty Predictions:
Das Modell gibt nach Decoding leere Strings aus (`empty_pred` hoch, CER oft 100%).

2. Special-Token-Loop:
Das Modell wiederholt fast nur `<s_ocr>` / `</s_ocr>` / `<pad>` oder beendet sofort.

3. Repetitive Text-Loops:
Das Modell produziert lange Wiederholungen (z. B. immer dieselbe Silbe), CER kann > 100% werden.

Der Tiny-Schritt ist ein frueher Stresstest, der genau diese Failure-Modes sichtbar macht,
bevor du sehr viel GPU-Zeit in einen Full-Run investierst.

---

## 2) Warum Tiny-Pretraining gegen Collapse hilft

Tiny-Pretraining ist kein "magischer Trick", sondern eine kontrollierte Diagnose- und Stabilitaetsphase:

1. Du validierst Datenpfade, Manifest-Felder, Tokenizer und Label-Format auf kleinem Datensatz.
2. Du siehst frueh, ob das Modell in Empty-/Special-Token-Loops kippt.
3. Du erhaeltst einen warmen, bereits stabilen Start-Checkpoint fuer den Full-Run.

In der Praxis reduziert das das Risiko, dass Full-Runs frueh kollabieren oder stundenlang in unbrauchbaren Regimen trainieren.

---

## 3) Voraussetzungen

1. Installierte Abhaengigkeiten:
```bash
pip install -r requirements.txt
```

2. Lokale BoSentencePiece-Tokenizer-Dateien:
- In diesem Training ist BoSentencePiece verpflichtend.
- Fallback auf beliebige AutoTokenizer ist absichtlich deaktiviert.
- Wenn der Tokenizer nicht passt, bricht das Skript frueh mit Fehler ab (gewollt).

3. Verfuegbare Manifeste:
- Train JSONL (line-image + text)
- Val JSONL

4. Wichtige Eingabefelder in JSONL:
- Bild: `line_path`, `src__image`, `image`, `image_path`, ...
- Text: `text`, `src__label`, `label`, `transcription`, ...

---

## 4) Pipeline-Wahl kurz erklaert

`--image_preprocess_pipeline` unterstuetzt:

1. `gray`:
- Graustufe via `min_rgb`, ohne harte Binarisierung.
- Oft ein robuster Baseline-Start.

2. `rgb`:
- RGB-Line-Scan Cleanup, farberhaltend (z. B. rot + schwarz).
- Keine harte Binarisierung per Default.
- Typisch fuer gemischte Tintenfarben.

3. `bdrc`:
- Graustufe + adaptive Binarisierung.
- Eher klassischer OCR-Preproc-Stil.

Empfehlung:
1. Wenn rot/schwarz wichtig: `rgb` testen.
2. Parallel immer `gray` als solide Vergleichsbasis laufen lassen.

---

## 5) Tiny-Datensatz bauen (deterministisch)

Beispiel: aus grossen Manifests einen kleinen, reproduzierbaren Tiny-Split erzeugen.

```bash
export TRAIN_MANIFEST=/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/train/meta/lines.jsonl
export VAL_MANIFEST=/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/eval/meta/lines.jsonl
export TINY_DIR=/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines_tiny

mkdir -p "$TINY_DIR/train/meta" "$TINY_DIR/eval/meta"

python - <<'PY'
import random
from pathlib import Path

seed = 42
train_n = 512
val_n = 128

train_src = Path("/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/train/meta/lines.jsonl")
val_src = Path("/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/eval/meta/lines.jsonl")
train_dst = Path("/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines_tiny/train/meta/lines.jsonl")
val_dst = Path("/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines_tiny/eval/meta/lines.jsonl")

def sample_jsonl(src: Path, dst: Path, n: int, seed: int) -> None:
    rows = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(seed)
    if len(rows) <= n:
        picked = rows
    else:
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        picked = [rows[i] for i in sorted(idx[:n])]
    dst.write_text("\n".join(picked) + "\n", encoding="utf-8")

sample_jsonl(train_src, train_dst, train_n, seed)
sample_jsonl(val_src, val_dst, val_n, seed + 1)
print("wrote", train_dst, val_dst)
PY
```

---

## 6) Tiny-Run (from scratch)

Ziel: pruefen, ob Setup stabil lernt und nicht kollabiert.

Beispiel `gray`:

```bash
CUDA_VISIBLE_DEVICES=0 python cli.py train-donut-ocr \
  --train_manifest /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines_tiny/train/meta/lines.jsonl \
  --val_manifest /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines_tiny/eval/meta/lines.jsonl \
  --output_dir /dev/shm/donut_tiny_overfit_from_scratch_gray \
  --model_name_or_path microsoft/trocr-base-stage1 \
  --tokenizer_path /home/ubuntu/data/PechaBridge/ext/BoSentencePiece \
  --image_preprocess_pipeline gray \
  --image_size 384 \
  --max_target_length 160 \
  --generation_max_length 160 \
  --generation_min_new_tokens 0 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 30 \
  --warmup_steps 40 \
  --eval_steps 20 \
  --save_steps 20 \
  --save_total_limit 10 \
  --num_workers 8 \
  --seed 42 \
  --val_eval_max_samples 128
```

Analog fuer `rgb` nur `--image_preprocess_pipeline rgb` und anderer `--output_dir`.

---

## 7) Tiny-Run interpretieren (wichtige Signale)

### Gesund

1. `eval_cer` sinkt ueber Zeit.
2. `empty_pred=0`.
3. `eval_pred_repetitive_ratio` bleibt niedrig.
4. `eval_pred_avg_len` naehrt sich `eval_ref_avg_len`.
5. `eval_token_forensics` zeigt:
- hohe Token-Diversitaet (`token_unique_nonpad`),
- Special-Token-Anteil in Pred nahe Label-Verteilung.

### Ungesund / Warnzeichen

1. `empty_pred_ratio` hoch oder oszillierend Richtung 1.0.
2. `eval_pred_repetitive_ratio` stark steigend.
3. `top_tokens_nonpad` wird von wenigen Tokens dominiert.
4. CER bleibt lange bei ~100% oder springt auf sehr hohe Werte mit Loops.

---

## 8) Bester Tiny-Checkpoint waehlen

Nimm den Checkpoint mit:

1. niedrigster stabiler CER (nicht nur ein Ausreisser),
2. `empty_pred=0`,
3. `repetitive_ratio` nahe 0,
4. plausibler Pred-Length (in der Naehe der Ref-Length).

Typisch:
- `/dev/shm/donut_tiny_overfit_from_scratch_gray/checkpoint-XXX`
- `/dev/shm/donut_tiny_overfit_from_scratch_rgb/checkpoint-YYY`

---

## 9) Full-Run aus Tiny-Bestcheckpoint starten

### 9.1 Single-GPU Beispiel (gray)

```bash
export TRAIN_MANIFEST=/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/train/meta/lines.jsonl
export VAL_MANIFEST=/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/eval/meta/lines.jsonl
export TOKENIZER=/home/ubuntu/data/PechaBridge/ext/BoSentencePiece
export BEST_GRAY=/dev/shm/donut_tiny_overfit_from_scratch_gray/checkpoint-480

CUDA_VISIBLE_DEVICES=0 python cli.py train-donut-ocr \
  --train_manifest "$TRAIN_MANIFEST" \
  --val_manifest "$VAL_MANIFEST" \
  --output_dir /home/ubuntu/data/PechaBridge/models/donut_full_gray_from_tiny \
  --model_name_or_path "$BEST_GRAY" \
  --image_processor_path /dev/shm/donut_tiny_overfit_from_scratch_gray/image_processor \
  --tokenizer_path "$TOKENIZER" \
  --image_preprocess_pipeline gray \
  --image_size 384 \
  --max_target_length 160 \
  --generation_max_length 160 \
  --generation_min_new_tokens 0 \
  --per_device_train_batch_size 40 \
  --per_device_eval_batch_size 40 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 80 \
  --warmup_steps 1000 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --num_workers 8 \
  --seed 42 \
  --val_eval_max_samples 300
```

### 9.2 Single-GPU Beispiel (rgb)

Nur diese Werte wechseln:

1. `--model_name_or_path "$BEST_RGB"`
2. `--image_processor_path /dev/shm/donut_tiny_overfit_from_scratch_rgb/image_processor`
3. `--image_preprocess_pipeline rgb`
4. eigener `--output_dir`

### 9.3 Multi-GPU (1,2,3) mit `torchrun`

```bash
export TRAIN_MANIFEST=/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/train/meta/lines.jsonl
export VAL_MANIFEST=/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/eval/meta/lines.jsonl
export TOKENIZER=/home/ubuntu/data/PechaBridge/ext/BoSentencePiece
export BEST_RGB=/dev/shm/donut_tiny_overfit_from_scratch_rgb/checkpoint-480

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 cli.py train-donut-ocr \
  --train_manifest "$TRAIN_MANIFEST" \
  --val_manifest "$VAL_MANIFEST" \
  --output_dir /home/ubuntu/data/PechaBridge/models/donut_full_rgb_from_tiny \
  --model_name_or_path "$BEST_RGB" \
  --image_processor_path /dev/shm/donut_tiny_overfit_from_scratch_rgb/image_processor \
  --tokenizer_path "$TOKENIZER" \
  --image_preprocess_pipeline rgb \
  --image_size 384 \
  --max_target_length 160 \
  --generation_max_length 160 \
  --generation_min_new_tokens 0 \
  --per_device_train_batch_size 40 \
  --per_device_eval_batch_size 40 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 80 \
  --warmup_steps 1000 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --num_workers 8 \
  --seed 42 \
  --val_eval_max_samples 300
```

Hinweis:
`--save_total_limit 5` bedeutet, dass nur die 5 neuesten Checkpoints im Output-Verzeichnis behalten werden.

---

## 10) Was bedeuten die wichtigsten Logs?

1. `eval_cer`:
- Hauptqualitaetsmetrik.
- kleiner ist besser.

2. `eval_empty_pred_count`:
- sollte 0 sein.
- hoher Wert ist akutes Warnsignal.

3. `eval_pred_repetitive_ratio`:
- misst Wiederholungsmodus.
- sollte niedrig bleiben.

4. `eval_pred_avg_len` vs `eval_ref_avg_len`:
- starke Laengendifferenz ist oft ein Fruehwarnzeichen fuer Decoder-Probleme.

5. `eval_token_forensics`:
- sehr wichtig fuer Diagnose.
- vergleiche Pred/Label bei:
  - `token_special_ratio`,
  - `token_unique_nonpad`,
  - Top-Tokens,
  - Start-Token-Verhalten.

---

## 11) Diagnose-Checkliste bei Problemen

Wenn Training instabil wird, in dieser Reihenfolge pruefen:

1. Tokenizer korrekt?
- Muss BoSentencePiece sein.
- Token audit im Log pruefen.

2. Manifest-Felder/Paths korrekt?
- Keine leeren Texte.
- Bildpfade aufloesbar.

3. Pipeline passend?
- Bei Farbtinte `rgb` statt harter Binarisierung.

4. Tiny erneut laufen lassen:
- Reproduzierbar kleiner Datensatz,
- kurze Eval-Intervalle,
- Failure-Mode zuerst dort fixen.

5. LR und Warmup:
- Bei instabilem Decoder ggf. niedrigere LR oder laengeres Warmup testen.

6. Checkpoint-Wahl:
- Nicht nur nach niedrigster CER gehen,
- auch `empty_pred` und `repetitive_ratio` beachten.

---

## 12) Praktische Empfehlungen fuer stabile Runs

1. Immer zuerst Tiny-Run pro Pipeline (`gray`, `rgb`).
2. Full-Runs aus stabilem Tiny-Checkpoint starten, nicht blind from scratch.
3. Eval-Intervalle anfangs kuerzer halten (fruehe Fehler sichtbar machen).
4. `val_eval_max_samples` fuer schnelle Iteration begrenzen, spaeter ggf. erhoehen.
5. Zwei parallele Full-Runs (gray vs rgb) sind sinnvoll, wenn Farbinformation relevant ist.

---

## 13) Kurzfazit

Tiny-Pretraining ist in diesem Setup die wichtigste Schutzmassnahme gegen:

1. Empty-Prediction-Collapse,
2. Special-Token-Schleifen,
3. repetitive Decoder-Ausgaben.

Es kostet wenig Zeit, spart aber sehr viele Stunden fehlgeschlagenes Full-Training.
