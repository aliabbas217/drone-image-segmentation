import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF
from torchvision.io import read_image, ImageReadMode
import os
from typing import List, Tuple, Callable

# --- Transforms Definition ---

def get_image_pre_transform(height: int, width: int) -> Callable:
    """Returns the pre-transform for images (resizing, dtype conversion)."""
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((height, width), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    ])

def get_mask_pre_transform(height: int, width: int) -> Callable:
    """Returns the pre-transform for masks (resizing)."""
    return v2.Compose([
        v2.Resize((height, width), interpolation=v2.InterpolationMode.NEAREST),
    ])

def get_image_final_transform(mean: List[float], std: List[float]) -> Callable:
    """Returns the final transform for images (normalization)."""
    return v2.Compose([
        v2.Normalize(mean=mean, std=std)
    ])

# --- Dataset Class ---

class DroneImageDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, images: List[str], masks: List[str],
                 resize_height: int, resize_width: int, img_mean: List[float], img_std: List[float],
                 apply_geometric_transforms: bool = True):
        """
        Custom Dataset for Drone Images and Semantic Masks.

        Args:
            image_dir (str): Directory containing images.
            mask_dir (str): Directory containing masks.
            images (List[str]): List of image filenames.
            masks (List[str]): List of mask filenames (must correspond to images).
            resize_height (int): Target height for resizing.
            resize_width (int): Target width for resizing.
            img_mean (List[float]): Mean values for image normalization.
            img_std (List[float]): Standard deviation values for image normalization.
            apply_geometric_transforms (bool): Whether to apply random geometric augmentations.
        """
        if len(images) != len(masks):
             raise ValueError(f"Number of images ({len(images)}) and masks ({len(masks)}) must be equal.")

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = images
        self.masks = masks
        self.apply_geometric_transforms = apply_geometric_transforms

        # Initialize transforms
        self.image_pre_transform = get_image_pre_transform(resize_height, resize_width)
        self.mask_pre_transform = get_mask_pre_transform(resize_height, resize_width)
        self.image_final_transform = get_image_final_transform(img_mean, img_std)

        print(f"Dataset initialized with {len(self.images)} samples.")
        print(f"Geometric transforms {'enabled' if apply_geometric_transforms else 'disabled'}.")


    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and transforms an image and its corresponding mask."""
        img_name = self.images[index]
        mask_name = self.masks[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            # Read image as RGB, mask as is (usually single channel)
            image = read_image(img_path, mode=ImageReadMode.RGB)
            mask = read_image(mask_path)
        except Exception as e:
            print(f"Error reading image/mask at index {index}: {img_path} / {mask_path}")
            print(f"Error details: {e}")
            # This prevents crashing the DataLoader loop
            print(f"Returning dummy data for index {index}.")
            # Get dummy tensor shapes from transforms if possible
            h = self.image_pre_transform.transforms[1].size[0]
            w = self.image_pre_transform.transforms[1].size[1]
            dummy_img = torch.zeros((3, h, w), dtype=torch.float32)
            dummy_mask = torch.zeros((h, w), dtype=torch.long)
            return dummy_img, dummy_mask

        # --- Apply Transforms ---
        # 1. Pre-transforms (Resize, ToTensor/Dtype)
        image = self.image_pre_transform(image)
        mask = self.mask_pre_transform(mask)

        # 2. Geometric Transforms (applied to both image and mask)
        # apply this manually because torchvision v2 is not working with both image and mask, will check later
        if self.apply_geometric_transforms:
            # Horizontal Flip
            if torch.rand(1) < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Rotation
            angle = torch.empty(1).uniform_(-15, 15).item()
            image = TF.rotate(image, angle, interpolation=v2.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=v2.InterpolationMode.NEAREST)

        # 3. Final Image Transform (Normalization)
        image = self.image_final_transform(image)

        # 4. Final Mask Processing
        # Ensure mask is (H, W) and Long type for CrossEntropyLoss
        # Remove channel dimension if present and convert to long
        # reasoning: masks are usually single channel and convert to long for loss functions
        # you might ask why single channel? because masks do not overlap in classes
        mask = mask.squeeze(0).long()

        # --- Sanity Checks (Optional but recommended) ---
        if image.shape[1:] != mask.shape:
             print(f"Warning: Mismatch shape between image {image.shape} and mask {mask.shape} at index {index} after transforms.")
             # Potentially resize mask again as a fallback, though this indicates an issue
             mask = TF.resize(mask.unsqueeze(0), image.shape[1:], interpolation=v2.InterpolationMode.NEAREST).squeeze(0).long()
             print(f"Resized mask to {mask.shape}")

        if mask.dtype != torch.long:
            print(f"Warning: Mask dtype is {mask.dtype}, converting to torch.long for index {index}.")
            mask = mask.long()

        return image, mask