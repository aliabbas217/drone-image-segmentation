import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

# Add src directory to Python path for imports
# learn how to use this in your own code
# https://stackoverflow.com/questions/6760685/creating-a-python-script-that-can-be-run-from-anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src import config
from src.utils import helpers
from src.models.unet import UNet
from src.data.dataset import DroneImageDataset
from src.training import engine

def main():
    print("--- Drone Image Segmentation Training ---")

    # 1. Load Class Dictionary and Update Config
    print(f"Loading class dictionary from: {config.CSV_PATH}")
    class_dict = helpers.csv_to_dict(config.CSV_PATH)
    if class_dict:
        actual_num_classes = len(class_dict)
        if actual_num_classes != config.NUM_CLASSES:
            print(f"Updating NUM_CLASSES from default ({config.NUM_CLASSES}) to {actual_num_classes} based on CSV.")
            # Update the config value if possible (or just use the local variable)
            # Note: Directly modifying config variables imported like this might not be ideal
            #       in complex setups, but works for this structure.
            # setattr(config, 'NUM_CLASSES', actual_num_classes) # Less common way
            current_num_classes = actual_num_classes
        else:
            current_num_classes = config.NUM_CLASSES
        print(f"Using {current_num_classes} classes.")
        # print("Class Dictionary:", class_dict) # Optional: print the dict
    else:
        print(f"Warning: Could not load class dictionary. Using default NUM_CLASSES={config.NUM_CLASSES}")
        current_num_classes = config.NUM_CLASSES

    # 2. Split Data Filenames
    print("\nSplitting data...")
    train_img_names, test_img_names, train_mask_names, test_mask_names = helpers.split_data(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    if not train_img_names or not test_img_names:
        print("Error: Data splitting failed. Exiting.")
        return

    # 3. Create Datasets
    print("\nCreating datasets...")
    train_dataset = DroneImageDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        images=train_img_names,
        masks=train_mask_names,
        resize_height=config.RESIZE_HEIGHT,
        resize_width=config.RESIZE_WIDTH,
        img_mean=config.IMG_MEAN,
        img_std=config.IMG_STD,
        apply_geometric_transforms=True
    )

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

    # 4. Create DataLoaders
    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Optional: Visualize a sample from the train loader
    try:
        print("\nVisualizing one sample from train dataset...")
        sample_img, sample_mask = train_dataset[0]
        print(f"Sample Input Shape: {sample_img.shape}, Dtype: {sample_img.dtype}")
        print(f"Sample Target Shape: {sample_mask.shape}, Dtype: {sample_mask.dtype}")
        assert sample_mask.dtype == torch.long, "Mask dtype should be torch.long!"
        assert len(sample_mask.shape) == 2, "Mask shape should be (H, W)!"
        helpers.visualize_transformed_pair(sample_img, sample_mask, config.IMG_MEAN, config.IMG_STD)
    except Exception as e:
        print(f"Error visualizing sample: {e}")


    # 5. Initialize Model
    print("\nInitializing U-Net model...")
    model = UNet(
        in_channels=config.IN_CHANNELS,
        num_classes=current_num_classes,
        init_filters=config.INIT_FILTERS,
        layers_per_block=config.LAYERS_PER_BLOCK,
        levels=config.LEVELS,
        kernel_size=config.KERNEL_SIZE,
        max_pool_kernel_size=config.MAX_POOL_KERNEL_SIZE
    )

    # Apply weight initialization
    model.apply(helpers.initialize_weights)
    print(f"Model initialized with {current_num_classes} classes.")
    # Optional: Print model summary (requires torchsummary)
    # krna hai to krlo nahi to comment out, Asslaam o Alaikum!
    # try:
    #     from torchsummary import summary
    #     summary(model.to(config.DEVICE), (config.IN_CHANNELS, config.RESIZE_HEIGHT, config.RESIZE_WIDTH))
    # except ImportError:
    #     print("torchsummary not installed, skipping model summary.")
    # except Exception as e:
    #     print(f"Could not print model summary: {e}")


    # 6. Define Optimizer and Loss Function
    print("\nSetting up optimizer and loss function...")
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # CrossEntropyLoss expects raw logits (N, C, H, W) and target indices (N, H, W)
    loss_fn = nn.CrossEntropyLoss()
    print(f"Optimizer: Adam (LR={config.LEARNING_RATE})")
    print(f"Loss Function: CrossEntropyLoss")

    # 7. Start Training Loop
    train_losses, test_losses = engine.training_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=config.DEVICE,
        epochs=config.EPOCHS,
        model_save_dir=config.MODEL_SAVE_DIR,
        save_interval=config.SAVE_CHECKPOINT_EPOCH_INTERVAL
    )

    # 8. Save Final Model
    final_model_path = os.path.join(config.MODEL_SAVE_DIR, "unet_final.pth")
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"\nFinal model saved to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

    # 9. Plot and Save Losses
    if train_losses and test_losses:
        loss_plot_path = os.path.join(config.PLOT_SAVE_DIR, "training_loss_plot.png")
        helpers.save_loss_plot(train_losses, test_losses, loss_plot_path)

    print("\n--- Training Script Finished ---")

if __name__ == "__main__":
    main()