"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
import cv2

def load_json_data(
    file_path: str,
):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: Data file not found {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed - {e}")
        return []
    
def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def merge_histories(stage_histories):
    """
    將多個 stage 的 history 串接在一起
    輸出一個單一 dict
    """
    merged = {"train_loss": [], "val_loss": [], "val_score": []}
    for h in stage_histories:
        for k in merged.keys():
            if k in h:
                merged[k].extend(h[k])
    return merged


def plot_training_comparison(deeplabv3_history, unet_history, save_dir):
    """
    支援多階段 history 的比較繪圖
    deeplabv3_history, unet_history: list of history dicts
    """

    # merge 多個 stage
    deeplabv3_merged = merge_histories(deeplabv3_history)
    unet_merged = merge_histories(unet_history)

    epochs_deep = range(1, len(deeplabv3_merged["train_loss"]) + 1)
    epochs_unet = range(1, len(unet_merged["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # --- Loss curve ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_deep, deeplabv3_merged["train_loss"], label="DeepLabV3 Train Loss")
    plt.plot(epochs_deep, deeplabv3_merged["val_loss"], label="DeepLabV3 Val Loss")
    plt.plot(epochs_unet, unet_merged["train_loss"], label="U-Net Train Loss")
    plt.plot(epochs_unet, unet_merged["val_loss"], label="U-Net Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    # --- Score curve ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_deep, deeplabv3_merged["val_score"], label="DeepLabV3 Val Score")
    plt.plot(epochs_unet, unet_merged["val_score"], label="U-Net Val Score")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Score")
    plt.legend()
    plt.title("Validation Score Comparison")

    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Training comparison plot saved to {save_path}")
    

def predict_and_visualize(
    model, 
    image_path, 
    mask_path,
    device, 
    transform=None
):
    model.eval()
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (768, 768), interpolation=cv2.INTER_NEAREST)
    
    if transform:
        augmented = transform(image=image)
        image_tensor = augmented['image']
    else:
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        if output.dim() == 4 and output.shape[1] == 1:
            output = output.squeeze(1)
        
        pred_mask = torch.sigmoid(output) > 0.5
        pred_mask = pred_mask.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('image')
    
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('predicted Mask')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('groundtruth')
    
    plt.tight_layout()
    plt.show()
    














