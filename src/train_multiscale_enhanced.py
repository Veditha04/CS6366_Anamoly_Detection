import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_dataloaders
from models import MultiScaleAutoencoder

CATEGORIES = ["bottle", "hazelnut", "cable", "tile"]


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch.to(device)
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        recon = model(images)
        loss = criterion(recon, images)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / total_samples


def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]    
            else:
                images = batch
            images = images.to(device)
            batch_size = images.size(0)
            
            recon = model(images)
            loss = criterion(recon, images)
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    return total_loss / total_samples if total_samples > 0 else float('inf')


def plot_training_history(train_losses, val_losses, save_path):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Multi-Scale Autoencoder - Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to: {save_path}")


def main():

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root    = os.path.join(PROJECT_ROOT, "data", "mvtec")
    models_dir   = os.path.join(PROJECT_ROOT, "models")
    results_dir  = os.path.join(PROJECT_ROOT, "results", "multiscale")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data root: {data_root}")
    
    # Hyperparameters
    img_size = 256
    batch_size = 8  # Smaller batch size for multi-scale (more memory)
    num_epochs = 45
    learning_rate = 1e-3
    patience = 15
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, test_loader = get_dataloaders(
        root_dir=data_root,
        categories=CATEGORIES,
        img_size=img_size,
        batch_size=batch_size,
        train_only=False,
    )
    
    # Create model
    model = MultiScaleAutoencoder(in_channels=3, base_channels=32).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: MultiScaleAutoencoder")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Mixed precision training for faster training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    
    # Training variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    best_model_path = os.path.join(models_dir, "multiscale_ae_best_enhanced.pth")
    final_model_path = os.path.join(models_dir, "multiscale_ae_final.pth")
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, best_model_path)
            print(f"✓ Best model saved to: {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
    
    # Save final model
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Plot training history
    plot_path = os.path.join(results_dir, "training_history.png")
    plot_training_history(train_losses, val_losses, plot_path)
    
    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Training completed in {epoch} epochs")
    
    # Save loss values
    loss_file = os.path.join(results_dir, "loss_values.npy")
    np.save(loss_file, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs': epoch
    })
    print(f"Loss values saved to: {loss_file}")


if __name__ == "__main__":
    main()
