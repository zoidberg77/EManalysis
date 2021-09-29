import os
import torch
import torch.nn as nn
from analyzer.vae.model.block import *

__all__ = ['ResNet3D', 'ResNet2D', 'resnet18', 'resnet34', 'resnet50',
           'resnet101', 'resnet152']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet3D(nn.Module):
    """ResNet backbone for 3D semantic/instance segmentation.
       The global average pooling and fully-connected layer are removed.
    """
    block_dict = {
        'residual': BasicBlock3d,
        'residual_se': BasicBlock3dSE,
    }
    num_stages = 5

    def __init__(self,
                 block_type: str = 'residual',
                 num_classes: int = 10,
                 in_channel: int = 1,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 blocks: List[int] = [2, 2, 2, 2],
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **_):
        super().__init__()
        assert len(filters) == self.num_stages
        self.block = self.block_dict[block_type]
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}
        self.filters = filters

        if isotropy[0]:
            kernel_size, padding = 5, 2
        else:
            kernel_size, padding = (1, 5, 5), (0, 2, 2)
        self.layer0 = conv3d_norm_act(in_channel,
                                      filters[0],
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      **self.shared_kwargs)

        self.layer1 = self._make_layer(
            filters[0], filters[1], blocks[0], 2, isotropy[1])
        self.layer2 = self._make_layer(
            filters[1], filters[2], blocks[1], 2, isotropy[2])
        self.layer3 = self._make_layer(
            filters[2], filters[3], blocks[2], 2, isotropy[3])
        self.layer4 = self._make_layer(
            filters[3], filters[4], blocks[3], 2, isotropy[4])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.fc = nn.Linear(filters[4], num_classes)

    def _make_layer(self, in_planes: int, planes: int, blocks: int,
                    stride: int = 1, isotropic: bool = False):
        if stride == 2 and not isotropic:
            stride = (1, 2, 2)
        layers = []
        layers.append(self.block(in_planes, planes, stride=stride,
                                 isotropic=isotropic, **self.shared_kwargs))
        for _ in range(1, blocks):
            layers.append(self.block(planes, planes, stride=1,
                                     isotropic=isotropic, **self.shared_kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class ResNet2D(nn.Module):
    '''ResNet backbone for 2D semantic/instance segmentation.
    '''
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet3DMM(nn.Module):
    '''ResNet backbone for 3D semantic/instance segmentation.
       Performs up-sampling using transposed 3D convolutions.
    '''
    block_dict = {
        'residual': BasicBlock3d,
        'residual_se': BasicBlock3dSE,
    }
    num_stages = 5

    def __init__(self,
                 block_type: str = 'residual',
                 num_classes: int = 10,
                 in_channel: int = 1,
                 filters: List[int] = [80, 64, 48, 36, 28],
                 blocks: List[int] = [2, 2, 2, 2],
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **_):
        super().__init__()
        assert len(filters) == self.num_stages
        self.block = self.block_dict[block_type]
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}
        self.filters = filters

        self.layer0 = trans_conv3d_norm_act(filters[0],
                                            filters[1],
                                            kernel_size=(3,3,3),
                                            **self.shared_kwargs)
        self.layer1 = trans_conv3d_norm_act(filters[1],
                                            filters[2],
                                            kernel_size=(3,3,3),
                                            **self.shared_kwargs)
        self.layer2 = trans_conv3d_norm_act(filters[2],
                                            filters[3],
                                            kernel_size=(1,1,1),
                                            **self.shared_kwargs)
        self.layer3 = trans_conv3d_norm_act(filters[3],
                                            filters[4],
                                            kernel_size=(1,1,1),
                                            **self.shared_kwargs)

    def _make_layer(self, in_planes: int, planes: int, blocks: int,
                    stride: int = 1, isotropic: bool = False):
        if stride == 2 and not isotropic:
            stride = (1, 2, 2)
        layers = []
        layers.append(self.block(in_planes, planes, stride=stride,
                                 isotropic=isotropic, **self.shared_kwargs, transpose=True))
        for _ in range(1, blocks):
            layers.append(self.block(planes, planes, stride=1,
                                     isotropic=isotropic, **self.shared_kwargs, transpose=True))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        #x = self.avgpool(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(script_dir + '/state_dicts/'+arch+'.pt', map_location=device)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, device,
                   **kwargs)


def resnet34(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, device,
                   **kwargs)


def resnet50(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, device,
                   **kwargs)


def resnet101(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, device,
                   **kwargs)


def resnet152(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, device,
                   **kwargs)
