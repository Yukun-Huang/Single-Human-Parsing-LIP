from collections import OrderedDict
from torch.utils import model_zoo
from .resnet import BasicBlock, Bottleneck, ResNet, model_urls
from .densenet import DenseNet
from .squeezenet import SqueezeNet


def load_weights_sequential(target, source_state):
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
        new_dict[k1] = v2
    target.load_state_dict(new_dict)


def load_weights_matched(target, source_state):
    new_dict = OrderedDict()
    for (k1, _) in target.state_dict().items():
        if k1 in source_state.keys():
            new_dict[k1] = source_state[k1]
    target.load_state_dict(new_dict)


def squeezenet(pretrained=True):
    model = SqueezeNet()
    if pretrained:
        from torchvision.models.squeezenet import squeezenet1_1
        source_state = squeezenet1_1(pretrained=True).features.state_dict()
        load_weights_sequential(model, source_state)
    return model


def densenet(pretrained=True):
    return DenseNet(pretrained=pretrained)


def resnet18(pretrained=True):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=True):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        load_weights_matched(model, model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        load_weights_matched(model, model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=True):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        load_weights_matched(model, model_zoo.load_url(model_urls['resnet152']))
    return model
