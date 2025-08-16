"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import torch
import os
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from XrayPnxSegment.common.utils import plot_training_comparison
from XrayPnxSegment.datasets.pnxImgSegSet import pnxImgSegSet, validate_dataset
from XrayPnxSegment.models.modeling_segModels import get_DeepLabV3Plus, get_Unet
from XrayPnxSegment.processors.img_processor import get_transform
from XrayPnxSegment.trainer.building_SegModelTrainer import get_lossFunc, get_optim, train_model


# save_time_stamp = os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime("%y%m%d%H%M"))
# config = {
#         'bsz': 12,
#         # 'lr': 1e-4,
#         # 'num_epoch': 150,
#         # 'img_size': (768, 768),
#         'mask_key': 'cropped_mask_path',  # 'mask_path', 'cropped_mask_path'
#         'skip_has_pnx': False,
#         'calc_class_weights': True,
#         'criterion': 'combined',          # 'BCE', 'combined'
#         'root_path': os.getcwd(),
#         'meta_path': 'subset_data_2508132253_small.json',
#         'save_path': os.path.join(os.getcwd(), 'checkpoints', save_time_stamp),
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#     }
 

# ---------- Sampler ----------
def create_ratio_based_sampler(dataset, target_pos_ratio, total_samples):
    """
    Creates a sampler that maintains a specific ratio of positive (pneumothorax) 
    to negative (non-pneumothorax) samples in each batch/epoch.
    
    Args:
        dataset: The dataset containing images with meta_list attribute
        target_pos_ratio: Desired ratio of positive samples (0.0 to 1.0)
        total_samples: Total number of samples to draw per epoch
    
    Returns:
        Either SubsetRandomSampler or WeightedRandomSampler
    """
    # Step 1: Separate indices based on pneumothorax presence
    positive_indices, negative_indices = [], []
    for i in range(len(dataset)):
        if dataset.meta_list[i]['has_pnx']:  # has pneumothorax
            positive_indices.append(i)
        else:  # no pneumothorax
            negative_indices.append(i)

    # Step 2: Calculate desired sample counts for each class
    pos_samples = int(total_samples * target_pos_ratio)     # e.g., 0.8 * 1000 = 800
    neg_samples = total_samples - pos_samples               # e.g., 1000 - 800 = 200

    # Step 3: Choose sampling strategy based on data availability
    if pos_samples <= len(positive_indices) and neg_samples <= len(negative_indices):
        # CASE 1: We have enough samples in both classes - use subset sampling
        # Randomly select exact number of samples from each class WITHOUT replacement
        selected_pos = np.random.choice(positive_indices, pos_samples, replace=False)
        selected_neg = np.random.choice(negative_indices, neg_samples, replace=False)
        
        # Combine and shuffle the selected indices
        selected_indices = np.concatenate([selected_pos, selected_neg])
        np.random.shuffle(selected_indices)
        
        return SubsetRandomSampler(selected_indices)
    else:
        # CASE 2: Not enough samples in one/both classes - use weighted sampling
        # This allows sampling WITH replacement to achieve desired ratios
        all_weights = np.zeros(len(dataset))
        
        # Calculate weights: higher weight = more likely to be sampled
        # If we need more pos samples than available, increase weight proportionally
        pos_weight = pos_samples / len(positive_indices) if pos_samples > len(positive_indices) else 1.0
        neg_weight = neg_samples / len(negative_indices) if neg_samples > len(negative_indices) else 1.0
        
        # Assign weights to corresponding indices
        all_weights[positive_indices] = pos_weight
        all_weights[negative_indices] = neg_weight
        
        return WeightedRandomSampler(all_weights, total_samples, replacement=True)

def run_pipeline(stages, config):
    best_deeplabv3_path = None
    # best_unet_path = None

    for idx, stage in enumerate(stages):
        print(f"\n=== Stage {idx} ===")
        print(f"lr={stage['lr']}, sample_rate={stage['sample_rate']}, image_size={stage['image_size']}")

        train_transform, val_transform = get_transform(image_size=stage['image_size'])

        train_dataset = pnxImgSegSet(
            datapath=config['root_path'], 
            meta_list_path=config['meta_path'], 
            mask_key=config['mask_key'],
            skip_has_pnx=config['skip_has_pnx'],
            split='train', 
            transform=train_transform, 
            image_size=stage['image_size'],
            calc_class_weights=config['calc_class_weights'],
        )
        print("Checking training dataset...")
        _ = validate_dataset(train_dataset)
        print(f'Training samples: {len(train_dataset)}')

        val_dataset = pnxImgSegSet(
            datapath=config['root_path'], 
            meta_list_path=config['meta_path'], 
            mask_key=config['mask_key'],
            split='test', 
            transform=val_transform, 
            image_size=stage['image_size'],
        )
        print("Checking validation dataset...")
        _ = validate_dataset(val_dataset)
        print(f'Validation samples: {len(val_dataset)}')

        # 動態建立 DataLoader
        sampler = create_ratio_based_sampler(
            dataset=train_dataset,
            target_pos_ratio=stage["sample_rate"],
            total_samples=len(train_dataset),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['bsz'],
            sampler=sampler,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['bsz'],
            shuffle=False,
            num_workers=4,
        )
        
        criterion = get_lossFunc(
            lossFunc=config['criterion'],
            class_weights=train_dataset.weights
        )


        # TRAIN DEEPLABV3PLUS
        print("\n" + "="*60)
        print("Training DeepLabV3Plus")
        print("="*60)
        deeplabv3_model = get_DeepLabV3Plus(
            device=config['device'],
        )
        
        # CRITICAL FIX: Load best weights from previous stage
        if idx > 0 and best_deeplabv3_path:
            if os.path.exists(best_deeplabv3_path):
                checkpoint = torch.load(best_deeplabv3_path, weights_only=True)
                deeplabv3_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise FileNotFoundError(f"{best_deeplabv3_path} does not exist!")
            print(f"Loading best DeepLabV3Plus weights from: {best_deeplabv3_path}")
            

        deeplabv3_optimizer, deeplabv3_scheduler = get_optim(
            model=deeplabv3_model,
            lr=stage["lr"],
            scheduler_type=stage["scheduler"],
        )
        deeplabv3_history = train_model(
            model=deeplabv3_model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=deeplabv3_optimizer, 
            scheduler=deeplabv3_scheduler, 
            num_epochs=stage['epochs'], 
            device=config['device'], 
            model_name=f'deeplabv3plus_stage{idx}', 
            save_dir=config['save_path'],
        )
        # Update best model path for DeepLabV3Plus
        best_deeplabv3_path = os.path.join(config['save_path'], f'best_deeplabv3plus_stage{idx}.pth')

        ## TRAIN UNET
        # print("\n" + "="*60)
        # print("Training U-Net")
        # print("="*60)

        # unet_model = get_Unet(
        #     device=config['device'],
        # )

        # # CRITICAL FIX: Load best weights from previous stage
        # if idx > 0 and best_unet_path:
        #     print(f"Loading best U-Net weights from: {best_unet_path}")
        #     unet_model.load_state_dict(torch.load(best_unet_path))

        # unet_optimizer, unet_scheduler = get_optim(
        #     model=unet_model,
        #     lr=config['lr'],
        #     scheduler_type=stage["scheduler"],
        # )
        # unet_history = train_model(
        #     model=unet_model, 
        #     train_loader=train_loader, 
        #     val_loader=val_loader, 
        #     criterion=criterion, 
        #     optimizer=unet_optimizer, 
        #     scheduler=unet_scheduler, 
        #     num_epochs=stages['epochs'], 
        #     device=config['device'], 
        #     model_name=f'unet_stage{idx}', 
        #     save_dir=config['save_path'],
        # )

        # # Update best model path for U-Net
        # best_unet_path = os.path.join(config['save_path'], f'best_unet_stage{idx}.pth')


    return {
        'best_deeplabv3_path': best_deeplabv3_path,
        # 'best_unet_path': best_unet_path
    }


# ---------- Main ----------
def main():

    config = {
        'bsz': 12,
        # 'lr': 1e-4,
        # 'num_epoch': 150,
        # 'img_size': (768, 768),
        'mask_key': 'cropped_mask_path',  # 'mask_path', 'cropped_mask_path'
        'skip_has_pnx': False,
        'calc_class_weights': True,
        'criterion': 'combined',          # 'BCE', 'combined'
        'root_path': os.getcwd(),
        'meta_path': 'subset_data_2508132253.json',
        'save_path': os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime("%y%m%d%H%M")),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    print(os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime("%y%m%d%H%M")))
    os.makedirs(config['save_path'], exist_ok=True)
    stages = [
        {"epochs": 12, "lr": 1e-3, "sample_rate": 0.8, "image_size": (512, 512), "scheduler": "ReduceLROnPlateau"},
        {"epochs": 30, "lr": 1e-5, "sample_rate": 0.6, "image_size": (768, 768), "scheduler": "CosineAnnealingLR"},
        {"epochs": 40, "lr": 1e-5, "sample_rate": 0.4, "image_size": (1024, 1024), "scheduler": "CosineAnnealingLR"},
    ]

    run_pipeline(stages, config)











    # print(f'Using device: {CONFIG["device"]}')
    # os.makedirs(CONFIG['save_path'], exist_ok=True)
    


    # # criterion = get_lossFunc(
    # #     lossFunc=CONFIG['criterion'],
    # #     class_weights=train_dataset.weights
    # # )

    # print("\n" + "="*60)
    # print("Training DeepLabV3Plus with staged pipeline")
    # print("="*60)
    # deeplabv3_history = run_pipeline(
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     model_builder=get_DeepLabV3Plus,
    #     criterion=criterion,
    #     save_dir=CONFIG['save_path'],
    #     model_name="deeplabv3plus",
    # )

    # print("\n" + "="*60)
    # print("Training U-Net with staged pipeline")
    # print("="*60)
    # unet_history = run_pipeline(
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     model_builder=get_Unet,
    #     criterion=criterion,
    #     save_dir=CONFIG['save_path'],
    #     model_name="unet",
    # )

    # # 你可以改寫 plot_training_comparison 來支援多個階段 history
    # plot_training_comparison(
    #     deeplabv3_history, 
    #     unet_history, 
    #     save_dir=CONFIG['save_path']
    # )


if __name__ == "__main__":
    main()
