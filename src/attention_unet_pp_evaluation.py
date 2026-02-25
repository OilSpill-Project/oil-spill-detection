import torch
from torch.utils.data import DataLoader

from dataset import OilSpillDataset
from deeplabv3_plus import DeepLabV3Plus
from utils import dice_score, iou_score

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# DATASET (VALIDATION)
# ----------------------------
val_dataset = OilSpillDataset(
    image_dir="/content/oil-spill-data/images/train",
    mask_dir="/content/oil-spill-data/masks/train",
    augment=False
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ----------------------------
# MODEL
# ----------------------------
model = DeepLabV3Plus().to(device)
model.load_state_dict(
    torch.load("/content/drive/MyDrive/oil-spill-detection/models/deeplabv3_plus.pth", map_location=device)
)

model.eval()

dice_total = 0.0
iou_total = 0.0

# ----------------------------
# EVALUATION
# ----------------------------
with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        dice_total += dice_score(preds, masks)
        iou_total += iou_score(preds, masks)


dice_avg = dice_total / len(val_loader)
iou_avg = iou_total / len(val_loader)

print(f"Average Dice Score: {dice_avg:.4f}")
print(f"Average IoU Score : {iou_avg:.4f}")

