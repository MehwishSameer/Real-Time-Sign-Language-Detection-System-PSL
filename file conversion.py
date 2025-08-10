import os
import cv2
import shutil
import pandas as pd
from pathlib import Path

# === CONFIG ===
source_dir = Path("dataset")             # your folder with class-wise folders
target_dir = Path("yolo_dataset")        # target YOLO folder
img_dir = target_dir / "images" / "train"
lbl_dir = target_dir / "labels" / "train"

# === PREPARE FOLDERS ===
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)

# === Get class names from folder names ===
class_names = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
class_to_id = {cls: i for i, cls in enumerate(class_names)}

print("Detected Classes:")
for cls, idx in class_to_id.items():
    print(f"  {idx}: {cls}")

# === Process all images ===
for cls_name in class_names:
    cls_path = source_dir / cls_name
    class_id = class_to_id[cls_name]

    for file in cls_path.glob("*.*"):
        if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        img = cv2.imread(str(file))
        if img is None:
            continue
        h, w, _ = img.shape

        # Copy image
        new_name = f"{cls_name}_{file.stem}{file.suffix}"
        new_path = img_dir / new_name
        shutil.copy(file, new_path)

        # Write dummy label (full image box)
        label_path = lbl_dir / f"{new_path.stem}.txt"
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# === Save class map ===
map_path = target_dir / "class_map.csv"
pd.DataFrame(list(class_to_id.items()), columns=["class", "id"]).to_csv(map_path, index=False)

print("\n✅ Conversion completed!")
print(f"YOLOv8 images in: {img_dir}")
print(f"YOLOv8 labels in: {lbl_dir}")
print(f"Class map saved to: {map_path}")
