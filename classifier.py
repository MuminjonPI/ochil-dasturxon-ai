"""Model code: load the fine-tuned ResNet18 and predict one image."""

import numpy as np
import timm
import torch

CLASSES = ["lagman", "manti", "plov", "somsa"]  # alphabetical = label order used in training
IMG = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(path="uzbek_dishes_fp16.pt"):
    """Build ResNet18 with 4 outputs and load the fine-tuned weights."""
    model = timm.create_model("resnet18", pretrained=False, num_classes=len(CLASSES))
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)  # float16 weights are cast back to float32 automatically
    model.eval()
    return model


def predict(model, img):
    """Return a numpy array of class probabilities for one PIL image."""
    x = np.asarray(img.convert("RGB").resize((IMG, IMG)))
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float() / 255
    x = (x - MEAN) / STD
    with torch.no_grad():
        return model(x).softmax(1)[0].numpy()
