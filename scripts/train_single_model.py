from ultralytics import YOLO
import os
import torch
import yaml
import multiprocessing

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "data.yaml")  # Use absolute path from project root
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0  # Use 0 for GPU, 'cpu' for CPU

# Model configuration
MODEL_NAME = "yolo12s"  # Change this to train a different model
MODEL_WEIGHTS = os.path.join("models", "yolo12s.pt")  # Use model from models folder

def check_requirements():
    # Check if data.yaml exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"data.yaml not found at {DATA_PATH}")
    
    # Check if model weights exist
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS}")
    
    # Check training data directory structure
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")
    
    # Check images directory
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
    
    # Check train/val directories
    train_dir = os.path.join(images_dir, "train")
    val_dir = os.path.join(images_dir, "val")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found at {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found at {val_dir}")
    
    # Check CUDA availability
    if DEVICE == 0 and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        return 'cpu'
    return DEVICE

def train():
    print(f"\n🚀 Starting training for {MODEL_NAME}...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Data path: {DATA_PATH}")
    print(f"Model weights: {MODEL_WEIGHTS}")
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        # Check requirements
        device = check_requirements()
        
        # Load model
        model = YOLO(MODEL_WEIGHTS)
        
        # Train
        model.train(
            data=DATA_PATH,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=device,
            workers=4,
            project="results",
            name=MODEL_NAME,
            exist_ok=True  # to avoid errors if folder exists
        )
        
        print(f"✅ Finished training {MODEL_NAME}. Results saved to /results/{MODEL_NAME}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Required for Windows
    train() 