import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
import torch.nn as nn

def csv_to_dict(path: str) -> Optional[Dict[int, str]]:
  """Loads class names from a CSV file into a dictionary."""
  try:
    df = pd.read_csv(path)
    if 'name' not in df.columns:
        print(f"Error: 'name' column not found in {path}")
        return None
    return {index: row['name'] for index, row in df.iterrows()}
  except FileNotFoundError:
    print(f"Error: CSV file not found at {path}")
    return None
  except Exception as e:
    print(f"Error reading CSV file {path}: {e}")
    return None

def initialize_weights(m: nn.Module):
    """ Initializes weights using Kaiming normal for Conv layers. """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def split_data(image_dir: str, mask_dir: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
  """ Splits image and mask filenames into training and testing sets. """
  try:
    images = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and not f.startswith('.')])
    masks = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f)) and not f.startswith('.')])
  except FileNotFoundError as e:
      print(f"Error finding image/mask directory: {e}")
      return [], [], [], []
  except Exception as e:
      print(f"An error occurred listing files: {e}")
      return [], [], [], []


  if not images or not masks:
      print("Error: Image or mask directory is empty or could not be read.")
      return [], [], [], []

  # Basic check: Ensure corresponding files exist (e.g., image '123.jpg' has mask '123.png')
  # More robust checks might be needed depending on exact naming conventions.
  img_basenames = {os.path.splitext(f)[0] for f in images}
  mask_basenames = {os.path.splitext(f)[0] for f in masks}

  if img_basenames != mask_basenames:
      print(f"Warning: Image and mask basenames do not perfectly match.")
      print(f"Images found: {len(images)}, Masks found: {len(masks)}")
      print(f"Images not in masks: {img_basenames - mask_basenames}")
      print(f"Masks not in images: {mask_basenames - img_basenames}")
      # Attempt to proceed with matched files
      common_basenames = list(img_basenames.intersection(mask_basenames))
      if not common_basenames:
          print("Error: No matching image/mask pairs found.")
          return [], [], [], []
      # Find full filenames corresponding to common basenames (assuming consistent extensions)
      # This part needs adjustment based on actual file extensions if they differ
      img_ext = os.path.splitext(images[0])[1]
      mask_ext = os.path.splitext(masks[0])[1]
      images = sorted([f"{bn}{img_ext}" for bn in common_basenames])
      masks = sorted([f"{bn}{mask_ext}" for bn in common_basenames])
      print(f"Proceeding with {len(images)} matched pairs.")


  if len(images) != len(masks):
      print(f"Error: Number of matched images ({len(images)}) and masks ({len(masks)}) differ after filtering.")
      return [], [], [], []

  try:
      train_images, test_images, train_masks, test_masks = train_test_split(
          images, masks, test_size=test_size, random_state=random_state, shuffle=True
      )
      print(f"Data split: {len(train_images)} train, {len(test_images)} test images/masks.")
      return train_images, test_images, train_masks, test_masks
  except Exception as e:
      print(f"Error during train_test_split: {e}")
      return [], [], [], []


def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Reverses the normalization on a tensor image."""
    tensor = tensor.clone()
    mean_t = torch.tensor(mean).view(len(mean), 1, 1).to(tensor.device)
    std_t = torch.tensor(std).view(len(std), 1, 1).to(tensor.device)
    tensor.mul_(std_t).add_(mean_t)
    return tensor

def visualize_transformed_pair(image: torch.Tensor, mask: torch.Tensor, mean: List[float], std: List[float]):
    """
    Visualize the transformed image and mask side by side.
    Assumes image is normalized.
    """
    # Denormalize image for visualization
    image_denorm = denormalize(image.cpu(), mean, std)
    image_denorm = image_denorm.clamp(0, 1)

    # Convert tensors to numpy arrays
    # Permute C, H, W -> H, W, C for matplotlib
    image_np = image_denorm.permute(1, 2, 0).numpy()
    mask_np = mask.cpu().numpy() # (H, W)

    # Plot image and mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Transformed Image")
    axes[0].axis('off')
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Transformed Mask")
    axes[1].axis('off')
    plt.show()


def plot_predictions(model: nn.Module, dataset, device: torch.device,
                     mean: List[float], std: List[float], num_classes: int,
                     indices: List[int], save_dir: Optional[str] = None):
    """
    Plots original image, ground truth mask, and predicted mask for given indices.
    """
    model.eval()
    model.to(device)

    num_plots = len(indices)
    plt.figure(figsize=(15, 5 * num_plots))

    plot_num = 1
    for index in indices:
        if index >= len(dataset):
            print(f"Warning: Index {index} is out of bounds for the dataset (size {len(dataset)}). Skipping.")
            continue

        # Get sample
        img, gt_mask = dataset[index] # img: (C, H, W), gt_mask: (H, W)

        # Prepare image for model input
        input_img = img.unsqueeze(0).to(device) # Add batch dim: (1, C, H, W)

        # Get prediction
        with torch.no_grad():
            pred_logits = model(input_img) # Output: (1, num_classes, H, W)

        # Process prediction: get class indices
        pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0) # (H, W)
        pred_mask_cpu = pred_mask.cpu().numpy() # Move to CPU for plotting

        # Prepare original image for plotting
        img_cpu = img.cpu() # Move original image to CPU
        img_denormalized = denormalize(img_cpu, mean, std)
        # Permute C, H, W -> H, W, C for matplotlib
        img_to_plot = img_denormalized.permute(1, 2, 0).numpy()
        # Clip values to [0, 1]
        img_to_plot = np.clip(img_to_plot, 0, 1)

        # Prepare ground truth mask for plotting
        gt_mask_cpu = gt_mask.cpu().numpy()

        # --- Plotting ---
        plt.subplot(num_plots, 3, plot_num)
        plt.imshow(img_to_plot)
        plt.title(f"Image (Index: {index})")
        plt.axis('off')

        plt.subplot(num_plots, 3, plot_num + 1)
        plt.imshow(gt_mask_cpu, cmap='viridis', vmin=0, vmax=num_classes - 1) # Use consistent colormap
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(num_plots, 3, plot_num + 2)
        plt.imshow(pred_mask_cpu, cmap='viridis', vmin=0, vmax=num_classes - 1) # Use consistent colormap
        plt.title("Predicted Mask")
        plt.axis('off')

        plot_num += 3

    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f"predictions_{'_'.join(map(str, indices))}.png")
        plt.savefig(save_path)
        print(f"Prediction plot saved to {save_path}")
    plt.show()


def save_loss_plot(train_losses: List[float], test_losses: List[float], save_path: str):
    """Saves a plot of training and validation losses."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(save_path)
        print(f"\nLoss plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving loss plot to {save_path}: {e}")
    plt.close() # Close the plot to free memory