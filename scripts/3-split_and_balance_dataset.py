import os
import shutil
import random
from collections import defaultdict

# THIS SCRIPT WILL SPLIT AND BALANCE THE DATASET


# === CONFIG ===
image_dir = "merged_dataset/images"
label_dir = "merged_dataset/labels"
output_base = "dataset"
split_ratio = {"train": 0.7, "val": 0.2, "test": 0.1}
class_ids = {"fire": "0", "smoke": "1"}
max_total_samples = 4000  # Set to None for no limit, or a number to limit total samples across all categories

# === Prepare output folders ===
for split in split_ratio:
    os.makedirs(f"{output_base}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_base}/labels/{split}", exist_ok=True)

# === Categorize images ===
categories = {
    "only_fire": [],
    "only_smoke": [],
    "both": [],
    "other": []
}

for lbl_file in os.listdir(label_dir):
    if not lbl_file.endswith(".txt"):
        continue

    with open(os.path.join(label_dir, lbl_file), "r") as f:
        labels = [line.split()[0] for line in f if line.strip()]
        label_set = set(labels)

        base_name = os.path.splitext(lbl_file)[0]
        img_file = base_name + ".jpg"
        if not os.path.exists(os.path.join(image_dir, img_file)):
            img_file = base_name + ".png"
            if not os.path.exists(os.path.join(image_dir, img_file)):
                continue  # skip if image not found

        if label_set == {class_ids["fire"]}:
            categories["only_fire"].append(base_name)
        elif label_set == {class_ids["smoke"]}:
            categories["only_smoke"].append(base_name)
        elif class_ids["fire"] in label_set and class_ids["smoke"] in label_set:
            categories["both"].append(base_name)
        else:
            categories["other"].append(base_name)

# === Balance train categories ===
min_train_per_category = min(len(categories["only_fire"]),
                             len(categories["only_smoke"]),
                             len(categories["both"]))

# Calculate total samples across all categories
total_categories = len(categories["only_fire"]) + len(categories["only_smoke"]) + len(categories["both"])

if max_total_samples is not None:
    # Adjust min_train_per_category based on max_total_samples
    max_per_category = max_total_samples // 3  # Divide total limit by number of categories
    min_train_per_category = min(min_train_per_category, max_per_category)

train_count_per_category = int(min_train_per_category * split_ratio["train"])
val_count_per_category = int(min_train_per_category * split_ratio["val"])
test_count_per_category = int(min_train_per_category * split_ratio["test"])

def distribute(files, split, count):
    for base_name in random.sample(files, count):
        img_ext = ".jpg" if os.path.exists(os.path.join(image_dir, base_name + ".jpg")) else ".png"
        shutil.copy(os.path.join(image_dir, base_name + img_ext), f"{output_base}/images/{split}/{base_name + img_ext}")
        shutil.copy(os.path.join(label_dir, base_name + ".txt"), f"{output_base}/labels/{split}/{base_name + '.txt'}")

for category in ["only_fire", "only_smoke", "both"]:
    distribute(categories[category], "train", train_count_per_category)
    distribute(categories[category], "val", val_count_per_category)
    distribute(categories[category], "test", test_count_per_category)

print("✅ Split complete.")
print(f"Train: {3 * train_count_per_category}, Val: {3 * val_count_per_category}, Test: {3 * test_count_per_category}")
