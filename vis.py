import torch
from XrayPnxSegment.common.utils import predict_and_visualize
from XrayPnxSegment.processors.img_processor import get_transform
from XrayPnxSegment.models.modeling_segModels import get_DeepLabV3Plus

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_weight = r"G:\checkpoint_backup\multi-stage-blance_training\2508241734_same_resolution\best_deeplabv3plus_stage2.pth"
img_path = r"G:\XrayPnxSegment-main\siim-acr-pneumothorax\png_images\3_train_1_.png"
mask_path = r"G:\XrayPnxSegment-main\siim-acr-pneumothorax\png_masks\3_train_1_.png"

model = get_DeepLabV3Plus()
model.load_state_dict(torch.load(model_weight, weights_only=True)['model_state_dict'])
# state_dict = torch.load(model_weight, weights_only=True)
# model.load_state_dict(state_dict, strict=False)


predict_and_visualize(
    model=model, 
    image_path=img_path, 
    mask_path=mask_path, 
    device=device, 
    transform=get_transform()[1], 
)