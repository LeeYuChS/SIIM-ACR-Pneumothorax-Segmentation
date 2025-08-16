"""
    Claude給的 等train完試試看
    last modify: 2025-0816-2154
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history_files, save_path=None):
    """
    Plot training history from multiple stages and models
    
    Args:
        history_files: List of paths to history JSON files
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Stage Training Progress', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    all_train_loss = []
    all_val_loss = []
    all_train_dice = []
    all_val_dice = []
    
    stage_boundaries = [0]  # Track where each stage starts
    
    for idx, history_file in enumerate(history_files):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Extract metrics
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        train_dice = history.get('train_dice', [])  # Assuming you track dice score
        val_dice = history.get('val_dice', [])
        
        # Accumulate for continuous plotting
        all_train_loss.extend(train_loss)
        all_val_loss.extend(val_loss)
        all_train_dice.extend(train_dice)
        all_val_dice.extend(val_dice)
        
        # Track stage boundary
        if idx > 0:
            stage_boundaries.append(len(all_train_loss))
    
    # Create epoch indices
    epochs = list(range(len(all_train_loss)))
    
    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, all_train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, all_val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add vertical lines for stage boundaries
    for boundary in stage_boundaries[1:]:
        axes[0, 0].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Dice Score
    axes[0, 1].plot(epochs, all_train_dice, 'b-', label='Training Dice', linewidth=2)
    axes[0, 1].plot(epochs, all_val_dice, 'r-', label='Validation Dice', linewidth=2)
    axes[0, 1].set_title('Training & Validation Dice Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add vertical lines for stage boundaries
    for boundary in stage_boundaries[1:]:
        axes[0, 1].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Loss Comparison by Stage
    start_idx = 0
    for idx, boundary in enumerate(stage_boundaries[1:] + [len(all_train_loss)]):
        stage_epochs = list(range(start_idx, boundary))
        stage_train_loss = all_train_loss[start_idx:boundary]
        stage_val_loss = all_val_loss[start_idx:boundary]
        
        axes[1, 0].plot(stage_epochs, stage_train_loss, color=colors[idx % len(colors)], 
                       label=f'Stage {idx} Train', linewidth=2)
        axes[1, 0].plot(stage_epochs, stage_val_loss, color=colors[idx % len(colors)], 
                       linestyle='--', label=f'Stage {idx} Val', linewidth=2)
        start_idx = boundary
    
    axes[1, 0].set_title('Loss by Training Stage')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate Schedule (if available)
    # You need to modify your train_model function to save learning rates
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add stage annotations
    for i, boundary in enumerate(stage_boundaries[1:]):
        axes[1, 1].axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        if i == 0:
            axes[1, 1].text(boundary/2, axes[1, 1].get_ylim()[1]*0.9, 'Stage 0\n(lr=1e-3)', 
                           ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        elif i == 1:
            mid_point = (stage_boundaries[1] + boundary) / 2
            axes[1, 1].text(mid_point, axes[1, 1].get_ylim()[1]*0.9, 'Stage 1\n(lr=1e-5)', 
                           ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

# Usage example:
def plot_multi_model_comparison(deeplabv3_histories, unet_histories, save_path=None):
    """
    Compare DeepLabV3Plus and U-Net training across all stages
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DeepLabV3Plus vs U-Net Training Comparison', fontsize=16)
    
    # Load and combine histories for DeepLabV3Plus
    deeplabv3_train_loss, deeplabv3_val_loss = [], []
    deeplabv3_train_dice, deeplabv3_val_dice = [], []
    
    for hist_file in deeplabv3_histories:
        with open(hist_file, 'r') as f:
            hist = json.load(f)
        deeplabv3_train_loss.extend(hist.get('train_loss', []))
        deeplabv3_val_loss.extend(hist.get('val_loss', []))
        deeplabv3_train_dice.extend(hist.get('train_dice', []))
        deeplabv3_val_dice.extend(hist.get('val_dice', []))
    
    # Load and combine histories for U-Net
    unet_train_loss, unet_val_loss = [], []
    unet_train_dice, unet_val_dice = [], []
    
    for hist_file in unet_histories:
        with open(hist_file, 'r') as f:
            hist = json.load(f)
        unet_train_loss.extend(hist.get('train_loss', []))
        unet_val_loss.extend(hist.get('val_loss', []))
        unet_train_dice.extend(hist.get('train_dice', []))
        unet_val_dice.extend(hist.get('val_dice', []))
    
    epochs_deeplab = list(range(len(deeplabv3_train_loss)))
    epochs_unet = list(range(len(unet_train_loss)))
    
    # Training Loss Comparison
    axes[0, 0].plot(epochs_deeplab, deeplabv3_train_loss, 'b-', label='DeepLabV3Plus', linewidth=2)
    axes[0, 0].plot(epochs_unet, unet_train_loss, 'r-', label='U-Net', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation Loss Comparison
    axes[0, 1].plot(epochs_deeplab, deeplabv3_val_loss, 'b-', label='DeepLabV3Plus', linewidth=2)
    axes[0, 1].plot(epochs_unet, unet_val_loss, 'r-', label='U-Net', linewidth=2)
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Dice Comparison
    axes[1, 0].plot(epochs_deeplab, deeplabv3_train_dice, 'b-', label='DeepLabV3Plus', linewidth=2)
    axes[1, 0].plot(epochs_unet, unet_train_dice, 'r-', label='U-Net', linewidth=2)
    axes[1, 0].set_title('Training Dice Score Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation Dice Comparison
    axes[1, 1].plot(epochs_deeplab, deeplabv3_val_dice, 'b-', label='DeepLabV3Plus', linewidth=2)
    axes[1, 1].plot(epochs_unet, unet_val_dice, 'r-', label='U-Net', linewidth=2)
    axes[1, 1].set_title('Validation Dice Score Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Dice')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

# Example usage:

# After training
history_files = [
    r"G:\XrayPnxSegment-main\checkpoints\2508162150\history_deeplabv3plus_stage0.json",
    r"G:\XrayPnxSegment-main\checkpoints\2508162150\history_deeplabv3plus_stage1.json", 
    r"G:\XrayPnxSegment-main\checkpoints\2508162150\history_deeplabv3plus_stage2.json"
]

plot_training_history(history_files, 'training_progress.png')
"""
# For model comparison
deeplabv3_files = ['deeplabv3plus_stage0_history.json', 'deeplabv3plus_stage1_history.json', 'deeplabv3plus_stage2_history.json']
unet_files = ['unet_stage0_history.json', 'unet_stage1_history.json', 'unet_stage2_history.json']

plot_multi_model_comparison(deeplabv3_files, unet_files, 'model_comparison.png')
"""