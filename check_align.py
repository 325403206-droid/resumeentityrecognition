import sys
from data.training_data import TRAINING_DATA

with open("check_align_out.txt", "w", encoding="utf-8") as f:
    for i, (text, annotations) in enumerate(TRAINING_DATA):
        f.write(f"Sample {i+1}:\n")
        f.write(f"Text length: {len(text)}\n")
        for start, end, label in annotations["entities"]:
            snippet = text[start:end]
            f.write(f"  [{label}] ({start}, {end}): '{snippet}'\n")
