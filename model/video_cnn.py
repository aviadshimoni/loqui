# coding: utf-8
import math
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    TODO add documentation
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride: int = 1, downsample=None, se: bool = False) -> None:
        print(f"type of given inplanes inside BasicBlock init is: {type(inplanes)}")
        print(f"type of given planes inside BasicBlock init is: {type(planes)}")
        print(f"type of given downsample inside BasicBlock init is: {type(downsample)}")
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if self.se:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = nn.Conv2d(planes, planes // 16, kernel_size=1)
            self.conv4 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        print(f"type of given x inside BasicBlock forward is: {type(x)}")
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se:
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()

            out = out * w

        out = out + residual
        out = self.relu(out)
        print(f"type of returned out inside BasicBlock forward is: {type(out)}")
        return out


class ResNet(nn.Module):
    """
    TODO add documentation
    """

    def __init__(self, block, layers, se: bool = False) -> None:
        print(f"type of given block inside ResNet init is: {type(block)}")
        print(f"type of given layers inside ResNet init is: {type(layers)}")
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.se = se

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(512)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride: int = 1) -> nn.Sequential:
        print(f"type of given block inside ResNet _make_layer is: {type(block)}")
        print(f"type of given planes inside ResNet _make_layer is: {type(planes)}")
        print(f"type of given blocks inside ResNet _make_layer is: {type(blocks)}")
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample, se=self.se)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x):
        print(f"type of given x inside ResNet forward is: {type(x)}")
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        print(f"type of returned x inside ResNet forward is: {type(x)}")
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VideoCNN(nn.Module):
    """
    TODO add documentation
    """

    def __init__(self, se: bool = False) -> None:
        super(VideoCNN, self).__init__()

        # frontend3D
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # resnet
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        self.dropout = nn.Dropout(p=0.5)

        # backend_gru
        # initialize
        self._initialize_weights()

    def visual_frontend_forward(self, x):
        print(f"type of given x inside video_cnn visual is: {type(x)}")
        x = x.transpose(1, 2)
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet18.forward(x)
        print(f"type of returned x inside video_cnn visual is: {type(x)}")
        return x

    def forward(self, x):
        print(f"type of given x inside video_cnn is: {type(x)}")
        b, t = x.size()[:2]
        x = self.visual_frontend_forward(x)
        x = x.view(b, -1, 512)
        print(f"type of returned x inside video_cnn is: {type(x)}")
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
