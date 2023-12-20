import numpy as np
import timm

import torch.nn as nn
import torch

from torchvision.models.segmentation.fcn import FCNHead


NUM_CLASSES = 2


class XcpFcn(nn.Module):
    def __init__(self, pretrained=False):
        super(XcpFcn, self).__init__()
        self.xcp_backbone = timm.create_model("xception", pretrained=pretrained)
        self.xcp_backbone.global_pool = nn.Identity()
        self.xcp_backbone.fc = nn.Identity()
        self.head = FCNHead(728, NUM_CLASSES)

    def feat(self, x):
        x = self.xcp_backbone.conv1(x)
        x = self.xcp_backbone.bn1(x)
        x = self.xcp_backbone.act1(x)

        x = self.xcp_backbone.conv2(x)
        x = self.xcp_backbone.bn2(x)
        x = self.xcp_backbone.act2(x)

        x = self.xcp_backbone.block1(x)
        x = self.xcp_backbone.block2(x)
        x = self.xcp_backbone.block3(x)
        x = self.xcp_backbone.block4(x)
        x = self.xcp_backbone.block5(x)
        x = self.xcp_backbone.block6(x)
        x = self.xcp_backbone.block7(x)
        x = self.xcp_backbone.block8(x)
        x = self.xcp_backbone.block9(x)
        x = self.xcp_backbone.block10(x)
        x = self.xcp_backbone.block11(x)

        return x

    def forward(self, x):
        feat = self.feat(x)
        patches = self.head(feat)

        return patches


def xceptionnet(device, filename=None):
    model = timm.create_model("xception", pretrained=True)
    model.fc = nn.Linear(2048, NUM_CLASSES)
    if filename:
        state_dict = torch.load(filename, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def xceptionnetfcn(device, filename=None):
    model = XcpFcn(True)
    if filename:
        state_dict = torch.load(filename, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model
