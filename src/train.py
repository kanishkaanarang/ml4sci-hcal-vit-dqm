import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Resize
import numpy as np
import timm

from dataset import HCALDataset

# ---------------------------
# Safety for Windows CPU
# ---------------------------
torch.set_num_threads(1)

# ---------------------------
# Device
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Load data
# ---------------------------
d1 = np.load("data/Run355456_Dataset.npy")
d2 = np.load("data/Run357479_Dataset.npy")

# Use more data for better calibration
ds1 = HCALDataset(d1[:1000], 0)
ds2 = HCALDataset(d2[:1000], 1)

dataset = ConcatDataset([ds1, ds2])

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

# ---------------------------
# Model
# ---------------------------
model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=False,
    num_classes=2
)
model = model.to(device)

# ---------------------------
# Optimizer
# ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# Resize HCAL → ViT input
# ---------------------------
resize = Resize((224, 224))

# ---------------------------
# Training
# ---------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for i, (x, y) in enumerate(loader):
        x = resize(x).to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Batch {i}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}\n")

# ---------------------------
# Save model
# ---------------------------
torch.save(model.state_dict(), "vit_hcal.pth")
print("Model saved as vit_hcal.pth")
print("Training complete.")
# ---------------------------