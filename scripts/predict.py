import torch
from torch.utils.data import DataLoader
import os
import sys
import random
import argparse

# Add src directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src import config
from src.utils import helpers
from src.models.unet import UNet
from src.data.dataset import DroneImageDataset

def predict(args):
    """
    Run U-Net prediction on test images and plot results.
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    print("--- Drone Image Segmentation Prediction ---")

    # 1. Load Class Dictionary and Determine Number of Classes
    print(f"Loading class dictionary from: {config.CSV_PATH}")
    class_dict = helpers.csv_to_dict(config.CSV_PATH)
    if class_dict:
        current_num_classes = len(class_dict)
        print(f"Using {current_num_classes} classes based on CSV.")
    else:
        print(f"Warning: Could not load class dictionary. Using default NUM_CLASSES={config.NUM_CLASSES}")
        current_num_classes = config.NUM_CLASSES

    # 2. Load Model
    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    model = UNet(
        in_channels=config.IN_CHANNELS,
        num_classes=current_num_classes,
        init_filters=config.INIT_FILTERS,
        layers_per_block=config.LAYERS_PER_BLOCK,
        levels=config.LEVELS,
        kernel_size=config.KERNEL_SIZE,
        max_pool_kernel_size=config.MAX_POOL_KERNEL_SIZE
    )

    try:
        # Load state dict - ensure map_location handles CPU/GPU discrepancy if needed
        model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    print("\nPreparing test data...")
    # We need the list of test files, so we run split_data again
    # Alternatively, save the split lists during training and load them here
    _, test_img_names, _, test_mask_names = helpers.split_data(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        test_size=config.TEST_SIZE, # Use the same split parameters as training
        random_state=config.RANDOM_STATE
    )
    if not test_img_names:
        print("Error: Could not get test data file list.")
        return

    test_dataset = DroneImageDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        images=test_img_names,
        masks=test_mask_names,
        resize_height=config.RESIZE_HEIGHT,
        resize_width=config.RESIZE_WIDTH,
        img_mean=config.IMG_MEAN,
        img_std=config.IMG_STD,
        apply_geometric_transforms=False
    )

    # 4. Select Indices and Plot Predictions
    num_samples = len(test_dataset)
    if args.indices:
        indices_to_plot = [int(i) for i in args.indices.split(',')]
        indices_to_plot = [i for i in indices_to_plot if 0 <= i < num_samples]
        if not indices_to_plot:
             print(f"Error: Provided indices are out of range (0-{num_samples-1}).")
             return
    else:
        # Select random indices if none are provided
        num_to_plot = min(args.num_predictions, num_samples)
        indices_to_plot = random.sample(range(num_samples), num_to_plot)
        print(f"No specific indices provided, plotting {num_to_plot} random samples.")

    print(f"\nPlotting predictions for indices: {indices_to_plot}")
    helpers.plot_predictions(
        model=model,
        dataset=test_dataset,
        device=config.DEVICE,
        mean=config.IMG_MEAN,
        std=config.IMG_STD,
        num_classes=current_num_classes,
        indices=indices_to_plot,
        save_dir=config.PLOT_SAVE_DIR
    )

    print("\n--- Prediction Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net prediction on test images.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(config.MODEL_SAVE_DIR, "unet_final.pth"),
        help="Path to the trained model (.pth) file."
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of specific test dataset indices to predict (e.g., '5,10,25'). Overrides --num_predictions."
    )
    parser.add_argument(
        "--num_predictions",
        type=int,
        default=3,
        help="Number of random predictions to show if specific indices are not provided."
    )

    args = parser.parse_args()
    predict(args)