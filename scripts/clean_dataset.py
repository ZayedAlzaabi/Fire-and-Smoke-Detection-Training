import os
import shutil
import imagehash
from PIL import Image
import cv2

# THIS SCRIPT WILL CLEAN THE DATA (IT WAS USED ON THE ORIGINAL COMBINED DATA)


# ==== CONFIGURATION ====
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths relative to the script location
image_dir = os.path.join(SCRIPT_DIR, "..", "data", "all_data", "images")
label_dir = os.path.join(SCRIPT_DIR, "..", "data", "all_data", "labels")
resize_to = (640, 640)  # Set to None if you don't want resizing
remove_duplicates = True
# ========================

def remove_orphan_images():
    print("🧹 Removing images without matching labels...")
    for img_file in os.listdir(image_dir):
        base = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, base + ".txt")
        img_path = os.path.join(image_dir, img_file)
        if not os.path.exists(label_path):
            os.remove(img_path)

def remove_orphan_labels():
    print("🧹 Removing labels without matching images...")
    for label_file in os.listdir(label_dir):
        base = os.path.splitext(label_file)[0]
        image_path = os.path.join(image_dir, base + ".jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, base + ".png")
        if not os.path.exists(image_path):
            os.remove(os.path.join(label_dir, label_file))

def remove_empty_labels():
    print("🧹 Removing empty label files...")
    for label_file in os.listdir(label_dir):
        path = os.path.join(label_dir, label_file)
        if os.path.getsize(path) == 0:
            os.remove(path)
            base = os.path.splitext(label_file)[0]
            img_path = os.path.join(image_dir, base + ".jpg")
            if os.path.exists(img_path):
                os.remove(img_path)

def remove_duplicate_images():
    print("🧹 Checking for duplicate images...")
    hashes = {}
    for file in os.listdir(image_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(image_dir, file)
            try:
                img_hash = str(imagehash.average_hash(Image.open(path)))
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
                continue
            if img_hash in hashes:
                print(f"❌ Duplicate found: {file} and {hashes[img_hash]}")
                os.remove(path)
                label = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")
                if os.path.exists(label): os.remove(label)
            else:
                hashes[img_hash] = file

def resize_images():
    if resize_to is None:
        return
    print(f"📐 Resizing all images to {resize_to}...")
    for file in os.listdir(image_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(image_dir, file)
            try:
                img = cv2.imread(path)
                resized = cv2.resize(img, resize_to)
                cv2.imwrite(path, resized)
            except Exception as e:
                print(f"⚠️ Could not resize {file}: {e}")

# Run all cleaning steps
if __name__ == "__main__":
    remove_orphan_images()
    remove_orphan_labels()
    remove_empty_labels()
    if remove_duplicates:
        remove_duplicate_images()
    resize_images()
    print("✅ Dataset cleaning complete.")
