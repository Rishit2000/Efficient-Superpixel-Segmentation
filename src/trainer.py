"""
Main Training Script
Implements the trainer, optimizer, and training/validation loops.

Configured with:
- AdamW Optimizer (Weight Decay 0.05)
- Polynomial LR Scheduler (Power 0.9)
- Differential LR (Backbone LR * 0.1)
- Linear Warmup
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import custom modules
from src.models.decoupled_superpixel_vit import DecoupledSuperpixelViT
from src.data.dataset_loader import CityscapesDataset

# ---  Optimizer Setup ---

def create_optimizer(model: nn.Module, lr: float, backbone_lr: float, weight_decay: float = 0.05):
    """
    Creates an AdamW optimizer with differential learning rates
    for the backbone and the rest of the model.

    If backbone is frozen, only optimizes the decoder parameters.
    """
    all_params = set(model.parameters())
    backbone_params = set(model.backbone.parameters())
    decoder_params = all_params - backbone_params

    # Filter out frozen parameters (requires_grad=False)
    trainable_backbone_params = [p for p in backbone_params if p.requires_grad]
    trainable_decoder_params = [p for p in decoder_params if p.requires_grad]

    param_groups = []

    # Only add backbone params if not frozen
    if trainable_backbone_params:
        param_groups.append({"params": trainable_backbone_params, "lr": backbone_lr})

    # Add decoder params
    if trainable_decoder_params:
        param_groups.append({"params": trainable_decoder_params, "lr": lr})

    print(f"Optimizer setup:")
    if trainable_backbone_params:
        print(f"  - Backbone params: {len(trainable_backbone_params)}, LR: {backbone_lr}")
    else:
        print(f"  - Backbone params: 0 (FROZEN)")
    print(f"  - Decoder params: {len(trainable_decoder_params)}, LR: {lr}")

    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    return optimizer

# --- Training & Validation Loops ---

def train_one_epoch(model, loader, optimizer, device, epoch):
    """Runs one full training epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # --- Forward Pass (Loss is now computed inside the model) ---
        dense_logits, loss = model(images, labels)
        
        if loss is None:
            continue
            
        # --- Backward Pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Step the scheduler (per-iteration) ---
        #scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix(
            loss=loss.item(), 
            lr=optimizer.param_groups[1]['lr'] # Show decoder LR
        )
        
    return total_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader, device, num_classes, epoch=None, class_names=None):
    """
    Runs one full validation epoch.
    """
    model.eval()
    total_loss = 0.0
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    desc = f"Epoch {epoch} [Validate]" if epoch is not None else "Validating"

    pbar = tqdm(loader, desc=desc)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # --- Forward Pass ---
        dense_logits, loss = model(images, labels)

        if loss is None:
            continue

        total_loss += loss.item()

        # --- Metrics ---
        preds = torch.argmax(dense_logits, dim=1)

        # Update confusion matrix (vectorized - MUCH faster!)
        preds_np = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        valid_mask = (labels_np != model.ignore_index)

        # Use numpy's bincount for fast confusion matrix update
        valid_preds = preds_np[valid_mask]
        valid_labels = labels_np[valid_mask]

        # Compute indices for confusion matrix
        indices = num_classes * valid_labels + valid_preds
        conf_update = np.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
        conf_matrix += conf_update

    # Calculate metrics
    avg_loss = total_loss / len(loader)

    # IoU per class
    iou = np.diag(conf_matrix) / (
        conf_matrix.sum(axis=1) +
        conf_matrix.sum(axis=0) -
        np.diag(conf_matrix)
    )

    # Mean IoU
    mean_iou = np.nanmean(iou)

    # Pixel accuracy
    pixel_acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

    # Class accuracy (recall)
    class_acc = np.diag(conf_matrix) / conf_matrix.sum(axis=1)

    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"\nOverall Metrics:")
    print(f"  Loss:           {avg_loss:.4f}")
    print(f"  mIoU:           {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"  Pixel Accuracy: {pixel_acc:.4f} ({pixel_acc*100:.2f}%)")

    print(f"\nPer-Class Results:")
    print(f"{'Class':<20} {'IoU':>8} {'Accuracy':>10} {'Support':>10}")
    print("-" * 52)

    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        support = conf_matrix[i].sum()
        if support > 0:
            print(f"{class_name:<20} {iou[i]:>7.4f}  {class_acc[i]:>9.4f}  {support:>10}")
        else:
            print(f"{class_name:<20} {'N/A':>7}  {'N/A':>9}  {support:>10}")

    print("="*80)

    return {
        'loss': avg_loss,
        'mean_iou': mean_iou,
        'pixel_accuracy': pixel_acc,
        'per_class_iou': iou,
        'per_class_accuracy': class_acc,
        'confusion_matrix': conf_matrix
    }

# --- Main Training Function ---

def main():
    # --- Config ---
    NUM_CLASSES = 19
    IGNORE_INDEX = 255
    IMG_SIZE = (256, 512)  # (H, W)
    BATCH_SIZE = 3
    N_SEGMENTS = 2048
    
    EPOCHS = 10
    BASE_LR = 1e-3
    BACKBONE_LR = BASE_LR * 0.1  # Backbone LR multiplier is 0.1
    WEIGHT_DECAY = 0.05
    
    # Scheduler Config
    WARMUP_ITERS = 5000
    POWER = 0.9
    
    # VALIDATION ONLY MODE (set to True to skip training and only run validation)
    VALIDATION_ONLY = False
    
    # RESUME TRAINING FROM CHECKPOINT (set to True to continue from last_checkpoint.pth)
    RESUME_TRAINING = True
    CHECKPOINT_PATH = "last_checkpoint.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataloaders ---
    print("Loading datasets...")
    root_dir = Path(__file__).parent.parent / 'dataset' / 'Cityscapes'
    # NOTE: Random Scaling and Color Jittering must be
    # implemented inside your CityscapesDataset class transforms.
    train_dataset = CityscapesDataset(
        root=root_dir,
        split='train',
        image_size=IMG_SIZE
        # Pass augmentation flags here, e.g., augment=True
    )
    val_dataset = CityscapesDataset(
        root=root_dir,
        split='val',
        image_size=IMG_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"  - Train: {len(train_dataset)} images")
    print(f"  - Val: {len(val_dataset)} images")

    # Cityscapes class names
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]

    # --- Model ---
    print("Initializing model...")
    model = DecoupledSuperpixelViT(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        n_segments=N_SEGMENTS,
    ).to(device)

    # --- Optimizer ---
    optimizer = create_optimizer(model, lr=BASE_LR, backbone_lr=BACKBONE_LR, weight_decay=WEIGHT_DECAY)
    
    # --- Schedulers ---
    #print("Initializing learning rate schedulers...")
    #num_train_steps = EPOCHS * len(train_loader)
    
    #scheduler_warmup = lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=WARMUP_ITERS)
    #scheduler_poly = lr_scheduler.PolynomialLR(optimizer, total_iters=num_train_steps - WARMUP_ITERS, power=POWER)
    #scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_poly], milestones=[WARMUP_ITERS])
    
    # --- Load Checkpoint (if resuming) ---
    start_epoch = 1

    if RESUME_TRAINING and Path(CHECKPOINT_PATH).exists():
        print(f"\n📂 Loading checkpoint from '{CHECKPOINT_PATH}'...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Reinitialize optimizer to match current model state (handles frozen params)
        # NOTE: This resets momentum, but ensures optimizer only tracks trainable params
        print("⚠️  Reinitializing optimizer (momentum will be reset)")
        print("   This is necessary because backbone is now frozen")
        optimizer = create_optimizer(model, lr=BASE_LR, backbone_lr=BACKBONE_LR, weight_decay=WEIGHT_DECAY)
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        print(f"✓ Resumed from epoch {checkpoint['epoch']}")
        print(f"  Starting at epoch {start_epoch}")
    elif RESUME_TRAINING:
        print(f"\n⚠️  Checkpoint '{CHECKPOINT_PATH}' not found. Starting from scratch.")
    
    # --- Training Loop ---
    if VALIDATION_ONLY:
        print("\n🔍 VALIDATION ONLY MODE - Skipping training")
        
        # Load checkpoint if available
        if Path(CHECKPOINT_PATH).exists():
            print(f"Loading checkpoint from '{CHECKPOINT_PATH}'...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
        else:
            print(f"⚠️  No checkpoint found at '{CHECKPOINT_PATH}'. Using randomly initialized model.")
        
        print("Running validation to test speed...")
        import time
        start_time = time.time()
        val_results = validate_one_epoch(model, val_loader, device, NUM_CLASSES, epoch=1, class_names=class_names)
        elapsed = time.time() - start_time
        print(f"\n✓ Validation complete!")
        print(f"  Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
        print(f"  Speed: {len(val_loader)/elapsed:.2f} batches/sec")

        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print("="*80)
        print(f"Test mIoU:            {val_results['mean_iou']:.4f}")
        print(f"Test Loss:            {val_results['loss']:.4f}")
        print(f"Test Pixel Accuracy:  {val_results['pixel_accuracy']:.4f}")
        print("="*80)
        return
    
    print("\nStarting training...")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        print(f"Epoch {epoch}:")
        print(f"  [Train] Loss: {train_loss:.4f}")

        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, "last_checkpoint.pth")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()