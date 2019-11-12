import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.models.resnet import ResNet, BasicBlock


class MyResnet(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        penultimate = torch.flatten(x, 1)
        x = self.fc(penultimate)

        return x, penultimate


def give_me_resnet(pretrained):
    model = MyResnet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':
    pass
