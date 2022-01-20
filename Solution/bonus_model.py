"""Define your architecture here."""
import torch
from torch import nn
import torchvision.models as models


def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = models.mobilenet_v3_small(pretrained=True)

    model.classifier = nn.Sequential(nn.Linear(576, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 2))

    model.load_state_dict(torch.load('checkpoints/bonus.pt')['model'])

    return model



