import torch
from torch import nn
import torchvision
from torchvision.transforms import Resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available

class CNN_Encoder(nn.Module):
    """
    CNN Encoder module that supports ResNet, VGG, and ViT backbones.
    It extracts visual features from input images and resizes them
    into a fixed spatial resolution (encoded_image_size x encoded_image_size).
    """

    def __init__(self, NetType, method, encoded_image_size=14):
        super(CNN_Encoder, self).__init__()
        self.NetType = NetType
        self.enc_image_size = encoded_image_size

        # If backbone is a ResNet
        if 'resnet' in NetType:
            # Load a pretrained ResNet variant from torchvision
            cnn = getattr(torchvision.models, NetType)(pretrained=True)
            layers = [
                cnn.conv1,
                cnn.bn1,
                cnn.relu,
                cnn.maxpool,
            ]

            # Use the first N residual stages (here: 3 stages)
            model_stage = 3
            for i in range(model_stage):
                name = 'layer%d' % (i + 1)
                layers.append(getattr(cnn, name))
            self.net = nn.Sequential(*layers)

        # If backbone is VGG
        if 'vgg' in NetType:
            net = torchvision.models.vgg16(pretrained=True)
            # Drop the final classification head, keep convolutional layers
            modules = list(net.children())[:-1]
            self.net = nn.Sequential(*modules)

        # If backbone is Vision Transformer
        if 'vit' in NetType:  # e.g., "vit_b_16"
            net = getattr(torchvision.models, NetType)(pretrained=True)
            self.net = net

        # Resize features to fixed size (e.g., 14x14) regardless of input image size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Freeze most layers, enable fine-tuning selectively
        self.fine_tune()

    def forward(self, images):
        """
        Forward pass through encoder.

        :param images: input tensor (batch_size, 3, H, W)
        :return: encoded feature maps, shape depends on backbone
        """

        # Hardcoded override: force ResNet forward path
        self.NetType = 'resnet'

        if 'resnet' in self.NetType:
            # Extract convolutional feature maps
            out = self.net(images)  # shape: (batch_size, C, H/32, W/32)
            out = self.adaptive_pool(out)  # resized to (batch_size, C, 14, 14)

        if 'vgg' in self.NetType:
            out = self.net(images)
            out = self.adaptive_pool(out)  # resized to (batch_size, C, 14, 14)

        if 'vit' in self.NetType:
            # Resize input images to 224x224 before passing to ViT
            torch_resize = Resize([224, 224])
            images = torch_resize(images)

            # Extract patches and project them
            x = self.net._process_input(images)
            n = x.shape[0]

            # Expand class token for the whole batch
            batch_class_token = self.net.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Pass through Transformer encoder
            x = self.net.encoder(x)

            # Drop the class token and keep patch embeddings
            x = x[:, 1:, :]

            # Reshape to 2D feature map (n, C, 14, 14)
            out = x.permute(0, 2, 1).view(n, -1, 14, 14)

        return out

    def fine_tune(self, fine_tune=True):
        """
        Control which layers require gradients (for fine-tuning).

        - By default, freeze all layers.
        - If fine_tune=True, unfreeze later convolutional blocks
          so they can be updated during training.
        """
        # Freeze all parameters
        for p in self.net.parameters():
            p.requires_grad = False

        # If fine-tuning, unfreeze higher-level blocks (beyond index 5)
        for c in list(self.net.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
