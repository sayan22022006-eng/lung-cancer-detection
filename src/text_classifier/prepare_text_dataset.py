import json
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/text/xraydar-reports.jsonl")
OUTPUT_FILE = Path("data/text/lung_finding_dataset.csv")

TARGET_LABEL = "parenchymal_lesion"

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        text = record.get("text", "").strip()
        labels = record.get("labels", [])

        if not text:
            continue

        label = 1 if TARGET_LABEL in labels else 0

        records.append({
            "xray_id": record.get("xray_id"),
            "text": text,
            "label": label
        })

df = pd.DataFrame(records)

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset created successfully")
print(f"Total reports: {len(df)}")
print(f"Positive reports: {(df['label'] == 1).sum()}")
print(f"Negative reports: {(df['label'] == 0).sum()}")
print("\nClass distribution:")
print(df["label"].value_counts())
print(f"\nSaved to: {OUTPUT_FILE}")
