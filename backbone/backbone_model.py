
import torch.nn as nn
import torchvision.models as models
from load_config import *

config = load_config()
class ModifiedResNet50(nn.Module):
    def __init__(self,current_config):
        super(ModifiedResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # Extract all layers except the final fully connected layer (avgpool + fc)
        self.features = nn.Sequential(*list(resnet50.children())[:-2])
        # Add custom classification heads based on the dataset
        self.classification_heads = nn.ModuleList()
        # add classification heads
        for i in range(config['general']['heads_num']):
            self.classification_heads.append(self.generate_classification_head(config['general']['num_classes'], resnet50,current_config))
        # Freeze feature extraction layers
        for param in self.features.parameters():
            param.requires_grad = False
        # Enable training for classification heads
        for head in self.classification_heads:
            for param in head.parameters():
                param.requires_grad = False

    def generate_classification_head(self, num_classes, resnet50,current_config):
        """
        Generates a custom classification head with multiple feed-forward layers.

        Parameters:
        - num_classes: The number of output classes.
        - resnet50: The backbone ResNet50 model.
        - num_layers: The number of hidden layers in the classification head.

        Returns:
        - A sequential model with adaptive pooling, flattening, and multiple linear layers.
        """
        layers = []

        # Adaptive Pooling and Flattening at the start
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        layers.append(nn.Flatten())
        init_channels = resnet50.layer4[-1].conv3.out_channels

        if init_channels<=num_classes:
            step_channel =  2*(num_classes - init_channels) // (current_config['current_step']['num_layers']//2)
            for _ in range(current_config['current_step']['num_layers']//2):
                layers.append(nn.Linear(init_channels, init_channels + step_channel))
                layers.append(nn.ReLU())
                init_channels += step_channel
        else:
            step_channel = (init_channels - num_classes) // (current_config['current_step']['num_layers'] + 1)  # Ensure step size >= 1

        # Hidden layers with linear and ReLU
        for _ in range(current_config['current_step']['num_layers']):
            layers.append(nn.Linear(init_channels, init_channels - step_channel))
            layers.append(nn.BatchNorm1d(init_channels - step_channel))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=config['general']['dropout_prob']))
            init_channels -= step_channel

        # Final output layer
        layers.append(nn.Linear(init_channels, num_classes))

        # Return as a sequential model
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        x=self.features(x)
        for head in self.classification_heads:
            score = head(x)
            outputs.append(score)

        return outputs