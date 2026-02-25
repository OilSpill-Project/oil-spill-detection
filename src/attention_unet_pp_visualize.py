import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from attention_unet_pp import AttentionUNetPlusPlus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
val_dataset = OilSpillDataset(
    image_dir="/content/data/images/val",
    mask_dir="/content/data/masks/val",
    augment=False
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Model
model = AttentionUNetPlusPlus().to(device)
model.load_state_dict(
    torch.load("results/models/attention_unet_pp_30.pth", map_location=device)
)
model.eval()

# Get one batch
images, masks = next(iter(val_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    preds = torch.sigmoid(outputs)
    preds = (preds > 0.5).float()

# Plot
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(images[0][0].cpu(), cmap="gray")
axs[0].set_title("Input Image")

axs[1].imshow(masks[0][0], cmap="gray")
axs[1].set_title("Ground Truth")

axs[2].imshow(preds[0][0].cpu(), cmap="gray")
axs[2].set_title("Prediction")

for ax in axs:
    ax.axis("off")

plt.show()
show()
