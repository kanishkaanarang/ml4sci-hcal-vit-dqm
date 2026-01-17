import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Resize
import numpy as np
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from dataset import HCALDataset

# ---------------------------
# Device
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Load data (held-out slice)
# ---------------------------
d1 = np.load("data/Run355456_Dataset.npy")
d2 = np.load("data/Run357479_Dataset.npy")

ds1 = HCALDataset(d1[1000:1200], 0)
ds2 = HCALDataset(d2[1000:1200], 1)

dataset = ConcatDataset([ds1, ds2])

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
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
model.load_state_dict(torch.load("vit_hcal.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------------------
# Resize
# ---------------------------
resize = Resize((224, 224))

# ---------------------------
# Evaluation
# ---------------------------
all_labels = []
all_probs = []

with torch.no_grad():
    for x, y in loader:
        x = resize(x).to(device)
        y = y.to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]

        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ---------------------------
# Metrics
# ---------------------------
preds = [1 if p > 0.5 else 0 for p in all_probs]

acc = accuracy_score(all_labels, preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")

print("Min prob:", min(all_probs))
print("Max prob:", max(all_probs))
print("Mean prob:", sum(all_probs) / len(all_probs))

# ---------------------------
# ROC Curve
# ---------------------------
fpr, tpr, _ = roc_curve(all_labels, all_probs)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – ViT HCAL Run Classification")
plt.legend()
plt.tight_layout()

plt.savefig("roc_curve_vit_hcal.png", dpi=300)
plt.show()

print("ROC curve saved as roc_curve_vit_hcal.png")
print("Evaluation complete.")