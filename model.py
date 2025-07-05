# SimSiam Training on SSL4EO-S12 Dataset
import torch.nn as nn
import torchvision.models as models

class ProjectionMLP(nn.Module):
    """Projection MLP for SimSiam"""
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # No bias/scale in final BN
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class PredictionMLP(nn.Module):
    """Prediction MLP for SimSiam"""
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class SimSiam(nn.Module):
    """
    SimSiam model implementation
    """
    def __init__(self, backbone='resnet50', proj_dim=2048, pred_dim=512):
        super().__init__()
        
        # Backbone encoder
        if backbone == 'resnet50':
            self.encoder = models.resnet50(weights=None)
            self.encoder.fc = nn.Identity()  # Remove classification head
            encoder_dim = 2048
        elif backbone == 'resnet18':
            self.encoder = models.resnet18(weights=None)
            self.encoder.fc = nn.Identity()
            encoder_dim = 512
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        # Projection head
        self.projector = ProjectionMLP(encoder_dim, proj_dim, proj_dim)
        
        # Prediction head
        self.predictor = PredictionMLP(proj_dim, pred_dim, proj_dim)
    
    def forward(self, x1, x2):
        # Encode both views
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        # Predict
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        return p1, p2, z1.detach(), z2.detach()