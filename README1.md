# Drone Image Segmentation Project Guide

This guide provides instructions on setting up, configuring, training, and running predictions for the Drone Image Semantic Segmentation project using a U-Net model.

## Directory Structure

The project follows this structure:

```
drone_segmentation/
│
├── data/                     # (Optional) Placeholder for local dataset storage
│
├── outputs/                  # Directory for all generated outputs
│   ├── models/               # Saved model checkpoints (.pth files)
│   └── plots/                # Saved loss plots and prediction images (.png files)
│
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── config.py             # Central configuration file (paths, hyperparameters)
│   ├── data/                 # Data loading and processing code
│   │   ├── __init__.py
│   │   └── dataset.py        # DroneImageDataset class and transforms
│   ├── models/               # Model definition code
│   │   ├── __init__.py
│   │   └── unet.py           # U-Net model architecture (ConvBlock, Encoder, Decoder, UNet)
│   ├── training/             # Training and evaluation logic
│   │   ├── __init__.py
│   │   └── engine.py         # train_one_epoch, evaluate, training_loop functions
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── helpers.py        # Data splitting, plotting, weight init, etc.
│
├── scripts/                  # Executable Python scripts
│   ├── train.py              # Script to run the training process
│   └── predict.py            # Script to run predictions and visualize results
│
├── requirements.txt          # List of Python package dependencies
└── README.md                 # Project description (this file or similar)
```

## Setup

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd drone_segmentation
    ```

2.  **Create and Activate Virtual Environment (Recommended):**

    ```bash
    # Create environment
    python -m venv venv

    # Activate environment
    # Linux/macOS:
    source venv/bin/activate
    # Windows (cmd/powershell):
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    - The project expects the Semantic Drone Dataset.
    - By default, paths in `src/config.py` point to Kaggle's input directory (`/kaggle/input/semantic-drone-dataset/...`).
    - **If running locally:** Download the dataset and **update** the following variables in `src/config.py` to point to your local dataset location:
      - `CSV_PATH`
      - `IMAGE_DIR`
      - `MASK_DIR`

## Configuration (`src/config.py`)

All major settings are controlled via the `src/config.py` file. Before running training or prediction, review and adjust these parameters as needed:

- **Paths:**
  - `CSV_PATH`: Path to the class dictionary CSV file.
  - `IMAGE_DIR`: Path to the directory containing the original drone images.
  - `MASK_DIR`: Path to the directory containing the semantic mask images.
  - `OUTPUT_DIR`: Base directory where models and plots will be saved.
  - `MODEL_SAVE_DIR`: Specific subdirectory for saved models.
  - `PLOT_SAVE_DIR`: Specific subdirectory for saved plots.
- **Data Parameters:**
  - `RESIZE_HEIGHT`, `RESIZE_WIDTH`: Target dimensions for resizing images and masks.
  - `NUM_CLASSES`: Number of segmentation classes (will be updated automatically if `CSV_PATH` is valid).
  - `IMG_MEAN`, `IMG_STD`: Mean and standard deviation for image normalization (using ImageNet defaults).
- **Model Parameters:**
  - `IN_CHANNELS`: Number of input image channels (usually 3 for RGB).
  - `INIT_FILTERS`: Number of filters in the first convolutional layer of the U-Net.
  - `LAYERS_PER_BLOCK`: Number of convolutional layers within each `ConvBlock`.
  - `LEVELS`: Number of downsampling/upsampling levels in the U-Net (depth).
  - `KERNEL_SIZE`: Size of the convolutional kernels.
- **Training Parameters:**
  - `BATCH_SIZE`: Number of samples per batch during training and evaluation.
  - `EPOCHS`: Total number of training epochs.
  - `LEARNING_RATE`: Initial learning rate for the Adam optimizer.
  - `TEST_SIZE`: Fraction of the data to use for the test/validation set.
  - `RANDOM_STATE`: Seed for the train/test split for reproducibility.
  - `SAVE_CHECKPOINT_EPOCH_INTERVAL`: Frequency (in epochs) for saving model checkpoints during training.
- **Environment:**
  - `DEVICE`: Automatically detects CUDA GPU if available, otherwise uses CPU.
  - `NUM_WORKERS`: Number of worker processes for data loading.

## Training (`scripts/train.py`)

This script handles the entire training process:

1.  Loads configuration from `src/config.py`.
2.  Loads the class dictionary.
3.  Splits the data into training and testing sets.
4.  Creates `Dataset` and `DataLoader` instances for training and testing.
5.  Initializes the U-Net model and applies weight initialization.
6.  Sets up the Adam optimizer and CrossEntropyLoss function.
7.  Runs the training loop (`src/training/engine.py`):
    - Iterates through epochs.
    - Performs training steps on the training data for each epoch.
    - Evaluates the model on the test data after each epoch.
    - Saves model checkpoints periodically (defined by `SAVE_CHECKPOINT_EPOCH_INTERVAL`) to `outputs/models/`.
    - Prints training and evaluation loss.
8.  Saves the final trained model state dictionary as `unet_final.pth` in `outputs/models/`.
9.  Generates and saves a plot of training and test losses over epochs to `outputs/plots/training_loss_plot.png`.

**How to Run:**

Ensure your virtual environment is activated and you are in the `drone_segmentation` directory. Verify paths and hyperparameters in `src/config.py`.

```bash
python scripts/train.py
```

Monitor the console output for training progress and loss values. Check the `outputs/` directory for saved models and the loss plot upon completion.

## Prediction (`scripts/predict.py`)

This script loads a trained model and visualizes its predictions on selected images from the test set.

1.  Loads configuration from `src/config.py`.
2.  Loads the class dictionary to determine the number of classes.
3.  Loads the trained U-Net model state dictionary from a specified path.
4.  Prepares the test dataset (using the same split parameters as training).
5.  Selects images from the test set (either specified indices or random ones).
6.  Performs inference using the loaded model on the selected images.
7.  Generates plots showing the original image, the ground truth mask, and the model's predicted mask side-by-side.
8.  Saves the prediction plots to `outputs/plots/`.

**How to Run:**

Ensure your virtual environment is activated and you are in the `drone_segmentation` directory.

- **Predict using the default final model (`outputs/models/unet_final.pth`) and show 3 random predictions:**

  ```bash
  python scripts/predict.py
  ```

- **Specify a different model checkpoint:**

  ```bash
  python scripts/predict.py --model_path outputs/models/unet_epoch_100.pth
  ```

- **Specify the exact indices (from the test set) to visualize (comma-separated):**

  ```bash
  python scripts/predict.py --indices 5,10,25
  ```

  *(Note: Indices refer to the position within the *test split*, starting from 0).*

- **Specify the number of random predictions to show (if not using `--indices`):**
  ```bash
  python scripts/predict.py --num_predictions 5
  ```

Check the `outputs/plots/` directory for the generated prediction comparison images.
