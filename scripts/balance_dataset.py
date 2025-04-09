import os
import shutil
from collections import defaultdict

# THIS SCRIPT WILL GET AN EQUAL NUMBER OF ALL TYPES


# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths relative to the script location
images_dir = os.path.join(SCRIPT_DIR, "..", "data", "all_data", "images")
labels_dir = os.path.join(SCRIPT_DIR, "..", "data", "all_data", "labels")

# Output
output_base = "balanced_subset"
output_images_dir = os.path.join(SCRIPT_DIR, "..", "data", output_base, "images")
output_labels_dir = os.path.join(SCRIPT_DIR, "..", "data", output_base, "labels")
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Counters
counts = {
    "only_fire": 0,
    "only_smoke": 0,
    "both": 0
}
limits = {
    "only_fire": 500,
    "only_smoke": 500,
    "both": 500
}

# Function to categorize
def categorize_label(classes):
    unique = set(classes)
    if unique == {"0"}:
        return "only_fire"
    elif unique == {"1"}:
        return "only_smoke"
    elif "0" in unique and "1" in unique:
        return "both"
    else:
        return None

# Loop through labels
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    with open(os.path.join(labels_dir, label_file), "r") as f:
        lines = f.read().strip().split("\n")
        classes = [line.split()[0] for line in lines if line.strip()]

    category = categorize_label(classes)
    if category and counts[category] < limits[category]:
        # Copy label
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(output_labels_dir, label_file)
        shutil.copy(src_label, dst_label)

        # Copy image
        img_name = label_file.rsplit(".", 1)[0] + ".jpg"
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(output_images_dir, img_name)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
            counts[category] += 1

    # Stop if all categories are full
    if all(counts[c] >= limits[c] for c in counts):
        break

print("✅ Extraction complete:")
for k, v in counts.items():
    print(f"{k}: {v} images")
