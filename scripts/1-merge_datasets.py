# THIS WAS USED TO COMBINE ALL FILES DOWNLOADED FORM THE ORIGINAL DATASET

import os
import shutil

# Paths
source_base = "full_dataset"
output_base = "merged_dataset"

image_out = os.path.join(output_base, "images")
label_out = os.path.join(output_base, "labels")

print(f"Creating output directories: {image_out} and {label_out}")
os.makedirs(image_out, exist_ok=True)
os.makedirs(label_out, exist_ok=True)

# Map the splits to their actual directory names
split_map = {
    "train": "train",
    "valid": "valid",
    "test": "test"
}

total_images = 0
total_labels = 0

for split, dir_name in split_map.items():
    split_dir = os.path.join(source_base, dir_name)
    img_dir = os.path.join(split_dir, "images")
    lbl_dir = os.path.join(split_dir, "labels")
    
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory {img_dir} does not exist, skipping...")
        continue
        
    print(f"\nProcessing {split} split...")
    print(f"Looking for images in: {img_dir}")
    print(f"Looking for labels in: {lbl_dir}")
    
    images_processed = 0
    labels_processed = 0
    
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
            
        label_file = os.path.splitext(img_file)[0] + ".txt"

        # Copy image
        src_img = os.path.join(img_dir, img_file)
        dst_img = os.path.join(image_out, img_file)
        shutil.copy(src_img, dst_img)
        images_processed += 1

        # Copy label if it exists
        label_path = os.path.join(lbl_dir, label_file)
        if os.path.exists(label_path):
            dst_label = os.path.join(label_out, label_file)
            shutil.copy(label_path, dst_label)
            labels_processed += 1
    
    total_images += images_processed
    total_labels += labels_processed
    print(f"Processed {images_processed} images and {labels_processed} labels from {split} split")

print(f"\n✅ Merging complete!")
print(f"Total images processed: {total_images}")
print(f"Total labels processed: {total_labels}")
print(f"All data is now in '{output_base}/images' and '{output_base}/labels'")
