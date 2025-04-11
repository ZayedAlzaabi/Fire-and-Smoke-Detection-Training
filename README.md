# Fire and Smoke Detection Dataset Processing Scripts

This repository contains a series of scripts for processing and preparing a dataset for fire and smoke detection using YOLOv8. Each script performs a specific task in the data preparation pipeline.

## Script Overview

### 1. `1-Merge_dataset.py`
Merges the already split fire & smoke dataset into one file to be cleaned

### 2. `2-clean_dataset.py`
Cleans the merged dataset as it can contain similliar frames by 
- Deletes any image file that does not have a matching .txt label file.
- Deletes any label file that does not have a matching .jpg or .png image file.
- Deletes any label file that is empty.
- Calculates a hash of each image using imagehash. Deletes images that are visually identical (same hash).
- Resizes all images to 640x640 using OpenCV.

### 3. `3-split_and_balance_dataset.py`
Splits the dataset int oa 70/20/10 split with a max number of TOTAL images 


### 4. `4-augment_dataset.py`
- Applies transformations: horizontal flip, brightness/contrast adjustment, Gaussian blur, and rotation.
- Skips images that have no bounding boxes.
- Saves each augmented image in the same training folder with prefix aug_.
- Generates and saves new YOLO-format .txt label files for the augmented images.
- Maintains the original images and labels and adds new, varied samples for better model generalization.


### 5. `5-train_single_model.py`
Trains a selected model on the processed dataset. Configuration options include:
- Model architecture selection
- Training parameters (epochs, batch size, image size)
- GPU/CPU training support
- Automatic requirement checking
- Results saving and logging

## Usage
In order to prepare data from scratch.

1. Download the Fire & smoke dataset from https://universe.roboflow.com/middle-east-tech-university/fire-and-smoke-detection-hiwia/dataset/2
2. Place the files in the "full_dataset" folder
3. Run the scripts 1 to 4 to prepare data for training
3. Run train_single_model.py on the desired model

## Training Configuration
### Yolov8n, Yolov8s, Yolov12n & Yolov12s:
- EPOCHS = 50
- BATCH_SIZE = 16
- Initial learning rate (lr0): 0.01
- Final learning rate (lrf): 0.01 (10% of initial)

### Yolov8m, Yolo12m
- EPOCHS = 50
- BATCH_SIZE = 8
- Initial learning rate (lr0): 0.01
- Final learning rate (lrf): 0.01 (10% of initial)

All trained models available in the /results folder were trained with the above configuration and used the datasplit below:
Total: 3996 Images + 2799 Augmented images only in the training data:

* **Train:**
  * Fire: 933 Images + 933 Augmented Images
  * Smoke: 933 Images + 933 Augmented Images
  * Both (Fire & Smoke): 933 Images + 933 Augmented Images
  * Total: 5598
* **Val:**
  * Fire: 266 Images
  * Smoke: 266 Images
  * Both (Fire & Smoke): 266 Images
  * Total: 798
* **Test:**
  * Fire: 133 Images 
  * Smoke: 133 Images 
  * Both (Fire & Smoke): 133 Images 
  * Total: 399

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- Required Python packages 

