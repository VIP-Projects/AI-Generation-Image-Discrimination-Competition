from torch import nn
import timm
import torchvision.models as models
# from efficientnet_pytorch import EfficientNet

class EffNet(nn.Module):
    def __init__(self, backbone, n_out, is_sigmoid):
        super(EffNet, self).__init__()
        self.model = timm.create_model(model_name=backbone, pretrained=True)
        # self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.LazyLinear(n_out)
        self.is_sigmoid = is_sigmoid

    def forward(self, x):
        x = self.model(x)
        if self.is_sigmoid:
            x = nn.Sigmoid()(x)
        return x




