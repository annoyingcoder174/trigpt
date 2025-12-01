# src/train_irene_from_feedback.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


FEEDBACK_FILE = Path("data/chat_feedback.jsonl")
SFT_DATASET_FILE = Path("data/irene_sft_dataset.jsonl")
STYLE_EXAMPLES_FILE = Path("data/irene_style_from_feedback.json")


def looks_vietnamese(text: str) -> bool:
  """
  Very rough language guess. Same spirit as api.py.
  """
  if not text:
    return False
  t = text.lower()
  vi_markers = [
      " không", " ko ", " là ", "của ", "nhé", "nh nha", " nha", " ạ",
      "và", "được", "mình", "cậu", "tớ", "anh ", "chị ", "em ", "ơi",
  ]
  return any(m in t for m in vi_markers) or "đ" in t


def load_feedback() -> List[Dict[str, Any]]:
  if not FEEDBACK_FILE.exists():
    print(f"[train_irene] No feedback file found at {FEEDBACK_FILE}")
    return []

  records: List[Dict[str, Any]] = []
  with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rec = json.loads(line)
        records.append(rec)
      except json.JSONDecodeError:
        print(f"[train_irene] Skipping bad JSON line: {line[:80]}...")
  return records


def build_datasets(
    records: List[Dict[str, Any]],
    min_good_rating: int = 4,
    max_good_per_lang: int = 200,
) -> None:
  """
  - All feedback -> SFT dataset (prompt/completion pairs).
  - High-rated ones -> style examples (few-shot bank).
  """
  sft_lines: List[str] = []
  style_examples: Dict[str, List[Dict[str, str]]] = {"en": [], "vi": []}

  # First pass: build SFT dataset
  for rec in records:
    question = (rec.get("question") or "").strip()
    answer = (rec.get("answer") or "").strip()
    rating = int(rec.get("rating") or 0)
    mode = rec.get("mode") or "unknown"
    comment = (rec.get("comment") or "").strip() or None

    if not question or not answer:
      continue

    lang = "vi" if looks_vietnamese(question + " " + answer) else "en"

    # Generic SFT-style item (compatible with many trainers later)
    item = {
        "lang": lang,
        "mode": mode,
        "rating": rating,
        "comment": comment,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }
    sft_lines.append(json.dumps(item, ensure_ascii=False))

  # Second pass: high-rated -> style examples (shortened)
  # We do a separate loop so style filter logic is easier.
  for rec in records:
    question = (rec.get("question") or "").strip()
    answer = (rec.get("answer") or "").strip()
    rating = int(rec.get("rating") or 0)

    if rating < min_good_rating:
      continue
    if not question or not answer:
      continue

    lang = "vi" if looks_vietnamese(question + " " + answer) else "en"
    lang_list = style_examples[lang]

    if len(lang_list) >= max_good_per_lang:
      continue

    # Keep it short-ish: truncate extra-long answers just for style bank
    max_chars = 900
    if len(answer) > max_chars:
      answer = answer[: max_chars].rstrip() + " …"

    lang_list.append(
        {
            "user": question,
            "assistant": answer,
        }
    )

  # --- Write outputs ---
  SFT_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)

  with SFT_DATASET_FILE.open("w", encoding="utf-8") as f:
    for line in sft_lines:
      f.write(line + "\n")
  print(
      f"[train_irene] Wrote SFT dataset with {len(sft_lines)} items to "
      f"{SFT_DATASET_FILE}"
  )

  with STYLE_EXAMPLES_FILE.open("w", encoding="utf-8") as f:
    json.dump(style_examples, f, ensure_ascii=False, indent=2)
  print(
      f"[train_irene] Wrote style examples from feedback to "
      f"{STYLE_EXAMPLES_FILE}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Build Irene training data from chat_feedback.jsonl"
  )
  parser.add_argument(
      "--min-good-rating",
      type=int,
      default=4,
      help="Minimum rating (1–5) to treat a sample as 'good style'. Default: 4",
  )
  parser.add_argument(
      "--max-good-per-lang",
      type=int,
      default=200,
      help="Maximum high-rated examples per language for style bank.",
  )
  args = parser.parse_args()

  records = load_feedback()
  if not records:
    print("[train_irene] No feedback records found. Nothing to do.")
    return

  print(f"[train_irene] Loaded {len(records)} feedback records.")
  build_datasets(
      records,
      min_good_rating=args.min_good_rating,
      max_good_per_lang=args.max_good_per_lang,
  )


if __name__ == "__main__":
  main()
