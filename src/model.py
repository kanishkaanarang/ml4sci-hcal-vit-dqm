import timm
import torch.nn as nn

def create_vit():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=2,
        in_chans=1
    )
    return model
