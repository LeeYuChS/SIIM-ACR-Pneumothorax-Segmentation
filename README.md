# XrayPnxSegment
Perform image segmentation on chest X-ray to obtain the scope of pneumothorax lesions

XrayPnxSegment provides a training pipeline for segmenting pneumothorax lesions from chest X-ray images. It contains scripts for preparing the dataset, defining neural network models, and training either **DeepLabV3+** or **U-Net** architectures using PyTorch.

## Project structure

```
XrayPnxSegment/
├── common/         # Utility functions
├── datasets/       # Dataset classes and helpers
├── models/         # Model definitions
├── processors/     # Data augmentation pipeline
└── trainer/        # Training loops and loss functions
```

## Installation
1. Create a Python environment (tested with Python 3.10):
   ```bash
   conda create --name classMAI python=3.10
   conda activate classMAI
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/yasaisen/XrayPnxSegment.git
   cd XrayPnxSegment
   ```
3. Install the dependencies:
   ```bash
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
   pip install matplotlib scikit-learn pandas tqdm opencv-python segmentation-models-pytorch albumentations
   ```

## Dataset preparation

The `preparing_data.py` script processes chest X-ray images and masks. It crops the masks to squares, creates subsets for training and testing, and saves the metadata as a JSON file.

```bash
python preparing_data.py
```

Modify the constants at the top of the script to point to your dataset directory. The resulting JSON file is used by the training pipeline.

## Training

`train.py` trains both DeepLabV3+ and U-Net models. Configuration settings such as batch size, learning rate and paths are stored in the `CONFIG` dictionary.

```bash
python train.py
```

Model checkpoints and training history will be saved under `checkpoints/<timestamp>/`.

## Inference example

After training, you can visualize predictions using the `predict_and_visualize` function:

```python
import torch
from XrayPnxSegment.common.utils import predict_and_visualize
from XrayPnxSegment.processors.img_processor import get_transform
from XrayPnxSegment.models.modeling_segModels import get_DeepLabV3Plus

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = get_DeepLabV3Plus()
model.load_state_dict(torch.load('path/to/model.pth')['model_state_dict'])

predict_and_visualize(
    model=model, 
    image_path='path/to/example.png', 
    mask_path='path/to/groundtruth.png', 
    device=device, 
    transform=get_transform()[1], 
)
```

## Data Source

This project relies on the **Chest X-Ray Images with Pneumothorax Masks** dataset  
(<https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks>),  
published on Kaggle by user **vbookshelf**.  
The dataset is a PNG-converted and folder-reorganised version of the original
**SIIM-ACR Pneumothorax Segmentation Challenge** data  
(<https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation>) released in 2019.

> **Size & contents** 12 047 CXR images (PNG) and 12 047 binary masks  
> **Pre-processing** DICOM → PNG conversion, filename normalisation, no extra annotation  
> **Download** Please obtain the data directly from Kaggle and place it under the `root_path` in your `CONFIG` dictionary from `train.py`.  
> **Redistribution** This repository does *not* bundle the raw images or masks.

### Citation

If you use the dataset in academic work, please cite *both* sources:

```bibtex
@misc{siim-acr-pneumothorax-segmentation,
    author = {Anna Zawacki and Carol Wu and George Shih and Julia Elliott and Mikhail Fomitchev and Mohannad Hussain and ParasLakhani and Phil Culliton and Shunxing Bao},
    title = {SIIM-ACR Pneumothorax Segmentation},
    year = {2019},
    howpublished = {\url{https://kaggle.com/competitions/siim-acr-pneumothorax-segmentation}},
    note = {Kaggle}
}

@dataset{vbookshelf_pneumothorax_masks_2019,
  title  = {Chest X-Ray Images with Pneumothorax Masks},
  author = {vbookshelf},
  year   = {2019},
  url    = {https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks},
  note   = {Processed PNG version of the SIIM-ACR challenge dataset}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
