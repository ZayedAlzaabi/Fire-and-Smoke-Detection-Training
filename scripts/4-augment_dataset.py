import os
import cv2
import albumentations as A
import numpy as np
from pathlib import Path

# THIS SCRIPT WILL ADD RANDOM AUGMENTATIONS TO THE DATA


# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input folders
img_dir = os.path.join(SCRIPT_DIR, "..", "dataset", "images", "train")
lbl_dir = os.path.join(SCRIPT_DIR, "..", "dataset", "labels", "train")

# Output folders
aug_img_dir = os.path.join(SCRIPT_DIR, "..", "dataset", "images", "train")
aug_lbl_dir = os.path.join(SCRIPT_DIR, "..", "dataset", "labels", "train")
# os.makedirs(aug_img_dir, exist_ok=True)
# os.makedirs(aug_lbl_dir, exist_ok=True)

# Albumentations augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Rotate(limit=15, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Process images
for img_file in os.listdir(img_dir):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(lbl_dir, img_file.rsplit(".", 1)[0] + ".txt")

    # Load image and labels
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    with open(label_path, "r") as f:
        lines = f.read().strip().split("\n")
    
    # Parse YOLO bboxes
    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id, bbox = int(parts[0]), list(map(float, parts[1:]))
        bboxes.append(bbox)
        class_labels.append(class_id)
    
    # Skip images with no bounding boxes
    if not bboxes:
        continue

    # Apply augmentation
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

    # Save augmented image
    aug_img_name = "aug_" + img_file
    cv2.imwrite(os.path.join(aug_img_dir, aug_img_name), augmented['image'])

    # Save augmented labels
    aug_label_name = "aug_" + img_file.rsplit(".", 1)[0] + ".txt"
    with open(os.path.join(aug_lbl_dir, aug_label_name), "w") as f:
        for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")
