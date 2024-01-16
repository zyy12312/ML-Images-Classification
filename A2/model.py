import torchvision
from torch import nn

# 搭建神经网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        vgg16_true = torchvision.models.vgg16(pretrained=True)
        vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 2))
        # print(vgg16_true)

    def forward(self, x):
        x = self.model(x)
        return x