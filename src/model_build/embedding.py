##in this module, I build the embedding model with residual connection
##this includes two layers of convolution for a small size model and acts as
#a  feature extractor
#it is known that residual connections allow the smooth flow of gradients during model training

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any

from src.logger import get_logger

logger = get_logger(__name__)

class ResidualBlock(nn.Module):
    """
    A simple residual block with two convolutional layers and a skip connection.
    If input and output channels differ, the skip connection uses a 1x1 conv.
    """
    def __init__(self, in_channels: int, out_channels:int, stride:int=1):
        super(ResidualBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                             kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(num_features=out_channels)
        #self.relu2=nn.ReLU(inplace=True)

        #skip connection : if dimensions change (stride > 1 or channel change), adjust

        self.skip = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu1(out)
        return out


class DeepCNN(nn.Module):
    """
    Custom deep embedding network with residual blocks.
    Uses OrderedDict to keep track of all layers for clarity and debugging.
    """
    def __init__(self, embedding_dim:int=128):
        super(DeepCNN,self).__init__()
        self.embedding_dim = embedding_dim

        #building the networkk with OrderedDict function
        #gives a clear way to debug and track layers (block_name, module)

        layers = OrderedDict()

        #initial stem: 3 ->32 channels, 7x7 conv, stride=2
        layers["stem_conv"] = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        layers["stem_bn"] = nn.BatchNorm2d(32)
        layers["stem_relu"] = nn.ReLU(True)

        #residual blocks: progressively increase channels and reduce spatial size
        #block1: 32 -> 64 stride=2 (downsample)
        layers["resblock1"] = ResidualBlock(32,64,2)
        #block2: 64 -> 128 stride=2
        layers["resblock2"] = ResidualBlock(64,128,2)
        #block3: 128 -> 256 stride =2
        layers["resblock3"] = ResidualBlock(128,256,2)
        #block4: 256 -> 512 stride 2
        layers["resblock4"] = ResidualBlock(256,512,2)

        #global average pool to reduce 512x1x1
        layers["avgpool"] = nn.AdaptiveAvgPool2d((1,1))

        #store all blocks as a sequential (preserves order)
        self.features = nn.Sequential(layers)

        #embedding projection
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.embedding_dim)
        )
        logger.info(
            "DeepCNN initialized with %d parameters, embedding dim %d",
            sum(p.numel() for p in self.parameters()),
            self.embedding_dim
        )
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.features(x)
        x = self.embedding(x)
        #l2 normaliztion
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
    

def build_embedding_network(config:Dict[str,Any])->nn.Module:
    """
    Factory function that returns an embedding network based on the config.
    Supported backbones: "custom_cnn" (DeepCNN), "resnet18" (torchvision).
    """
    backbone_type = config["model"]["backbone"]
    embedding_dim = config["model"]["embedding_dim"]

    if backbone_type == "custom_cnn":
        logger.info("using custom deepcnn backbone")
        return DeepCNN(embedding_dim=embedding_dim)
    elif backbone_type == "resnet18":
        try:
            import torchvision.models as tv_models
        except ImportError:
            raise ImportError("torchvision is required for resnet backbone")
        logger.info("using resnet18 backbone (pretrained=False by default, modify here to enable)")

        #for current models of we load the model as follows
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

        #now we replace the final fully connected layer to output embeddings
        in_features = model.fc.in_features

        #create a new fc with the model's in_features and our custom embedding dimension for the images
        model.fc = nn.Linear(in_features,embedding_dim)

        #wrap model to incliude normalization
        class ResNetEmbedding(nn.Module):
            def __init__(self, resnet,embed_dim):
                super().__init__()
                self.resnet = resnet
                self.embed_dim = embed_dim

            def forward(self,x):
                x = self.resnet(x)
                return nn.functional.normalize(x,p=2,dim=1)
        return ResNetEmbedding(model, embed_dim=embedding_dim)
    else:
        raise ValueError(f"unknown backbone: {backbone_type}. choose either 'custom_cnn' or 'resnet18'")