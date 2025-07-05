import torch
import torchvision.models as models
import torch.nn as nn

def load_pretrained_encoder(checkpoint_path, backbone='resnet50'):
    """Load pretrained encoder for downstream tasks"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize encoder
    if backbone == 'resnet50':
        encoder = models.resnet50(weights=None)
        encoder.fc = nn.Identity()
    elif backbone == 'resnet18':
        encoder = models.resnet18(weights=None)
        encoder.fc = nn.Identity()
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    return encoder

def create_linear_classifier(encoder, num_classes, freeze_encoder=True):
    """Create a linear classifier on top of the pretrained encoder"""
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
    
    # Get encoder output dimension
    if 'resnet50' in encoder:
        encoder_dim = 2048
    elif 'resnet18' in encoder:
        encoder_dim = 512
        print("ResNet 18 initialized")
    else:
        encoder_dim = 512  # default
    
    classifier = nn.Sequential(
        encoder,
        nn.Linear(encoder_dim, num_classes)
    )
    
    return classifier