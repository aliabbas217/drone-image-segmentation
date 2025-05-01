import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from typing import Tuple, List

def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                    loss_fn: nn.Module, device: torch.device, epoch: int, num_epochs: int,
                    model_save_dir: str, save_interval: int) -> float:
    """ Trains the model for one epoch. """
    model.train()
    total_loss = 0.0
    start_time = time.time()
    num_batches = len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Training...")
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        # Input: (N, C, H, W), Target: (N, H, W) where values are class indices
        loss = loss_fn(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress periodically
        if (batch_idx + 1) % (max(1, num_batches // 4)) == 0:
             print(f"  Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} Train Summary | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    # Save checkpoint periodically
    if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
        checkpoint_path = os.path.join(model_save_dir, f"unet_epoch_{epoch+1}.pth")
        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint at epoch {epoch+1}: {e}")

    return avg_loss

def evaluate(model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    """ Evaluates the model on the test/validation set. """
    model.eval()
    total_loss = 0.0
    start_time = time.time()
    num_batches = len(test_loader)

    print("Evaluating...")
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    eval_time = time.time() - start_time
    print(f"Evaluation Summary | Avg Loss: {avg_loss:.4f} | Time: {eval_time:.2f}s")
    return avg_loss

def training_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                  optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device,
                  epochs: int, model_save_dir: str, save_interval: int) -> Tuple[List[float], List[float]]:
    """ Runs the complete training and evaluation loop. """
    train_losses = []
    test_losses = []
    print("\n" + "="*30)
    print("Starting Training Loop...")
    print(f"Total Epochs: {epochs}")
    print(f"Device: {device}")
    print("="*30 + "\n")

    model.to(device)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, epochs, model_save_dir, save_interval)
        test_loss = evaluate(model, test_loader, loss_fn, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print("-" * (len(f"--- Epoch {epoch+1}/{epochs} ---")))

    print("\n" + "="*30)
    print("Training Finished.")
    print("="*30 + "\n")
    return train_losses, test_losses