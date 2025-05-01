import torch
import os

# --- Paths ---
KAGGLE_INPUT_DIR = "/kaggle/input/semantic-drone-dataset"
CSV_PATH = os.path.join(KAGGLE_INPUT_DIR, "class_dict_seg.csv")
IMAGE_DIR = os.path.join(KAGGLE_INPUT_DIR, "dataset/semantic_drone_dataset/original_images")
MASK_DIR = os.path.join(KAGGLE_INPUT_DIR, "dataset/semantic_drone_dataset/label_images_semantic")

OUTPUT_DIR = "/kaggle/working/outputs"
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_SAVE_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create output directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# --- Data Parameters ---
RESIZE_HEIGHT = 512
RESIZE_WIDTH = 512
NUM_CLASSES = 24
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# --- Model Parameters (U-Net Architecture) ---
IN_CHANNELS = 3
INIT_FILTERS = 64
LAYERS_PER_BLOCK = 4
LEVELS = 4
KERNEL_SIZE = 3
MAX_POOL_KERNEL_SIZE = 2

# --- Training Parameters ---
BATCH_SIZE = 4
EPOCHS = 150
LEARNING_RATE = 1e-4
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_CHECKPOINT_EPOCH_INTERVAL = 10

# --- Environment ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1

# --- Model Config Dictionary (Optional, can directly use constants above) ---
# model_config = {
#     "IN_CHANNELS": IN_CHANNELS,
#     "INIT_FILTERS": INIT_FILTERS,
#     "LAYERS_PER_BLOCK": LAYERS_PER_BLOCK,
#     "NUM_CLASSES": NUM_CLASSES, # This will be updated later
#     "LEVELS": LEVELS,
#     "KERNEL_SIZE": KERNEL_SIZE,
#     "MAX_POOL_KERNEL_SIZE": MAX_POOL_KERNEL_SIZE
# }

print(f"Using device: {DEVICE}")
print(f"Number of workers for DataLoader: {NUM_WORKERS}")
print(f"Output directory: {OUTPUT_DIR}")