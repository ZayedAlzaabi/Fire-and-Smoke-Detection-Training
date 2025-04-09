import os
import shutil
import random

# THIS SCRIPT WILL SPLIT THE DATA INTO 3 FILES
# Paths
all_images_dir = "data/balanced_subset/images"
all_labels_dir = "data/balanced_subset/labels"
output_dir = "data/resplit_data"

splits = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

# Create output folders
for split in splits:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

# Get and shuffle all images
images = [f for f in os.listdir(all_images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(images)

# Split and copy
n = len(images)
train_end = int(n * splits["train"])
val_end = train_end + int(n * splits["val"])

for i, img_file in enumerate(images):
    base_name = os.path.splitext(img_file)[0]
    label_file = f"{base_name}.txt"
    
    if i < train_end:
        split = "train"
    elif i < val_end:
        split = "val"
    else:
        split = "test"

    shutil.copy(f"{all_images_dir}/{img_file}", f"{output_dir}/images/{split}/{img_file}")
    shutil.copy(f"{all_labels_dir}/{label_file}", f"{output_dir}/labels/{split}/{label_file}")
