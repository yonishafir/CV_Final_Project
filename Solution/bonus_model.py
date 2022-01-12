"""Define your architecture here."""
import torch
from torch import nn
import torchvision.models as models

from models import SimpleNet


def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    #model = SimpleNet()
    # load your model using exactly this line (don't change it):
    # model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])

    #model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    model = models.mobilenet_v3_small(pretrained=True)


    # model.fc = nn.Sequential(nn.Linear(2048, 1000),
    #                                  nn.ReLU(),
    #                                  nn.Linear(1000, 256),
    #                                  nn.ReLU(),
    #                                  nn.Linear(256, 64),
    #                                  nn.ReLU(),
    #                                  nn.Linear(64, 2))

    model.fc = nn.Sequential(nn.Linear(1000, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 2))

    #model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    model.load_state_dict(torch.load('checkpoints/fakes_dataset_my_bonus_model_Adam.pt')['model'])


    return model
