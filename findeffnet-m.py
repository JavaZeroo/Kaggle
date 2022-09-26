import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torchvision as tv
from torchvision.models.feature_extraction import create_feature_extractor

WEIGHTS = tv.models.efficientnet.EfficientNet_V2_M_Weights.DEFAULT

writer = SummaryWriter(log_dir='./log', comment='effnet')

class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_m(weights=WEIGHTS)
        self.model = create_feature_extractor(effnet, ['flatten'])
        with writer:
            writer.add_graph(effnet, (Variable(torch.rand(32, 3, 384, 384)),))

        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )

    def forward(self, x):
        # returns logits
        x = self.model(x)['flatten']
        return self.nn_fracture(x), self.nn_vertebrae(x)

    def predict(self, x):
        frac, vert = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)
print('model')
model = EffnetModel()
print('model')
with writer:
    writer.add_graph(model, (Variable(torch.rand(1, 3, 384, 384)),))