"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import json
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from ..common.utils import convert


class SegmentationMetrics:
    def __init__(self, 
        threshold: float = 0.5, 
        eps: float = 1e-8
    ):
        self.threshold = threshold
        self.eps = eps
        self.reset()
    
    def reset(self
    ):
        self.iou_scores = []
        self.dice_scores = []
        self.pixel_accuracies = []
        self.valid_samples = 0
    
    def update(self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ):
        if preds.dim() == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        probs = torch.sigmoid(preds)
        pred_bin = (probs > self.threshold).float()
        target = targets.float()
        
        B = pred_bin.shape[0]
        pred_flat   = pred_bin.view(B, -1)
        target_flat = target.view(B, -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum    = pred_flat.sum(dim=1)
        target_sum  = target_flat.sum(dim=1)
        union       = pred_sum + target_sum - intersection
        
        non_empty = target_sum > 0
        if non_empty.sum() == 0:
            return
        
        inter_kept = intersection[non_empty]
        pred_kept  = pred_sum[non_empty]
        targ_kept  = target_sum[non_empty]
        union_kept = union[non_empty]
        
        # IoU / Dice
        iou_kept  = inter_kept / (union_kept + self.eps)
        dice_kept = (2 * inter_kept) / (pred_kept + targ_kept + self.eps)
        
        # Pixel accuracy
        correct = (pred_flat == target_flat).sum(dim=1)
        total   = pred_flat.shape[1]
        acc_kept = correct[non_empty].float() / float(total)
        
        self.iou_scores.extend( iou_kept.detach().cpu().tolist() )
        self.dice_scores.extend(dice_kept.detach().cpu().tolist())
        self.pixel_accuracies.extend(acc_kept.detach().cpu().tolist())
        self.valid_samples += non_empty.sum().item()
    
    def get_metrics(self
    ):
        if self.valid_samples == 0:
            return {
                'mean_iou': 0, 
                'mean_dice': 0, 
                'mean_pixel_accuracy': 0
            }
        else:
            return {
                'mean_iou': float(np.mean(self.iou_scores)),
                'mean_dice': float(np.mean(self.dice_scores)),
                'mean_pixel_accuracy': float(np.mean(self.pixel_accuracies)),
            }

def get_optim(
    model,
    lr,
    scheduler_type
):
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': lr * 0.5},
        {'params': model.decoder.parameters(), 'lr': lr},
    ], weight_decay=1e-4)
    if scheduler_type=="ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=3, 
            verbose=True
        )
    elif scheduler_type=="CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10
        )
    return optimizer, scheduler


def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device, 
    model_name, 
    save_dir
):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    best_val_iou = 0.0
    best_model_path = os.path.join(save_dir, f'best_{model_name}.pth')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # 記錄當前學習率
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        print(f'Current Learning Rate: {current_lr:.2e}')

        model.train()
        train_loss = 0.0
        train_metrics = SegmentationMetrics()
        
        progress_bar = tqdm(train_loader, desc='Training')
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            if outputs.dim() == 4 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_metrics.update(outputs, masks)
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        model.eval()
        val_loss = 0.0
        val_metrics = SegmentationMetrics()
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            for images, masks in progress_bar:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                
                if outputs.dim() == 4 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                
                loss = criterion(outputs, masks.float())
                val_loss += loss.item()
                val_metrics.update(outputs, masks)
                
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_metrics_dict = train_metrics.get_metrics()
        val_metrics_dict = val_metrics.get_metrics()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_iou'].append(train_metrics_dict['mean_iou'])
        history['val_iou'].append(val_metrics_dict['mean_iou'])
        history['train_dice'].append(train_metrics_dict['mean_dice'])
        history['val_dice'].append(val_metrics_dict['mean_dice'])
        history['train_accuracy'].append(train_metrics_dict['mean_pixel_accuracy'])
        history['val_accuracy'].append(val_metrics_dict['mean_pixel_accuracy'])
        
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Train IoU: {train_metrics_dict["mean_iou"]:.4f}, Val IoU: {val_metrics_dict["mean_iou"]:.4f}')
        print(f'Train Dice: {train_metrics_dict["mean_dice"]:.4f}, Val Dice: {val_metrics_dict["mean_dice"]:.4f}')
        print(f'Train Acc: {train_metrics_dict["mean_pixel_accuracy"]:.4f}, Val Acc: {val_metrics_dict["mean_pixel_accuracy"]:.4f}')
        # print(f'Empty samples - Train: {train_metrics_dict["empty_ratio"]:.2%}, Val: {val_metrics_dict["empty_ratio"]:.2%}')
        
        if val_metrics_dict['mean_iou'] > best_val_iou:
            best_val_iou = val_metrics_dict['mean_iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou,
                'history': history
            }, best_model_path)
            print(f'New best model saved with IoU: {best_val_iou:.4f}')
        
        scheduler.step(avg_val_loss)
    
    history_path = os.path.join(save_dir, f'history_{model_name}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4, default=convert)
    
    print(f'\nTraining completed! Best validation IoU: {best_val_iou:.4f}')
    return history

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.class_weights is not None:
            weight = self.class_weights.to(inputs.device)
            weight = weight[1] * targets + weight[0] * (1 - targets)
            focal_loss *= weight
        
        return focal_loss.mean()

class EnhancedCombinedLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal_loss = WeightedFocalLoss(class_weights=class_weights)
        self.lovasz_loss = smp.losses.LovaszLoss(mode='binary', from_logits=True)
    
    def forward(self, inputs, targets):
        return (0.3 * self.dice_loss(inputs, targets) +
                0.4 * self.focal_loss(inputs, targets) +
                0.3 * self.lovasz_loss(inputs, targets))

def get_lossFunc(
    lossFunc, 
    class_weights=None
):
    if lossFunc == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif lossFunc == 'focal':
        criterion = WeightedFocalLoss(
            class_weights=class_weights
        )
    elif lossFunc == 'dice':
        criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    elif lossFunc == 'lovasz':
        criterion = smp.losses.LovaszLoss(mode='binary', from_logits=True)
    elif lossFunc == 'combined':
        criterion = EnhancedCombinedLoss(
            class_weights=class_weights
        )
    return criterion



