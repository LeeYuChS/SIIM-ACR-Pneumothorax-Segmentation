"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import torch
import segmentation_models_pytorch as smp

from ..common.utils import get_trainable_params


IMG_ENCODER = "resnet50"
IMG_ENCODER_WEIGHT = "imagenet"

def get_DeepLabV3Plus(
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    deeplabv3_model = smp.DeepLabV3Plus(
        encoder_name=IMG_ENCODER,
        encoder_weights=IMG_ENCODER_WEIGHT,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    print(get_trainable_params(deeplabv3_model))
    return deeplabv3_model

def get_Unet(
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    unet_model = smp.Unet(
        encoder_name=IMG_ENCODER,
        encoder_weights=IMG_ENCODER_WEIGHT,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    print(get_trainable_params(unet_model))
    return unet_model
