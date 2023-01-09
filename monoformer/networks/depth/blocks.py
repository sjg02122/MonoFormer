import torch
from torch.autograd.grad_mode import F
import torch.nn as nn
import sys
from monoformer.networks.depth import attention_module
from .vit import (
    _make_pretrained_vitb_rn50_384,
    _make_pretrained_vitl16_384,
    _make_pretrained_vitb16_384,
    _make_pretrained_regionvit,
    forward_vit,
    forward_twins,
    forward_region,
    forward_slak,
    forward_convnext,
)
from . import attention_module
from .SLaK import SLaK_base
from .convnext import convnext_base

import timm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def _make_fusion_block(features, use_bn,expand,downsampling):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=expand,
        align_corners=True,
        downsampling=downsampling,
    )

def _make_conv(in_features,out_features):
    conv = nn.Module()
    conv.layer1 = nn.Conv2d(in_features[0],out_features[0],kernel_size=1,bias=True)
    conv.layer2 = nn.Conv2d(in_features[1],out_features[1],kernel_size=1,bias=True)
    conv.layer3 = nn.Conv2d(in_features[2],out_features[2],kernel_size=1,bias=True)
    conv.layer4 = nn.Conv2d(in_features[3],out_features[3],kernel_size=1,bias=True)
    
    return conv

def _make_DA(in_features):
    DA = nn.Module()
    DA.layer1 = attention_module._DAHead(in_features[0])
    DA.layer2 = attention_module._DAHead(in_features[1])
    DA.layer3 = attention_module._DAHead(in_features[2])
    DA.layer4 = attention_module._DAHead(in_features[3])
    return DA

def _make_refinenet(in_features,use_bn,expand,downsampling):
    refinenet = nn.Module()

    refinenet.layer1 = _make_fusion_block(in_features[0],use_bn,expand,downsampling)
    refinenet.layer2 = _make_fusion_block(in_features[1],use_bn,expand,downsampling)
    refinenet.layer3 = _make_fusion_block(in_features[2],use_bn,expand,downsampling)
    refinenet.layer4 = _make_fusion_block(in_features[3],use_bn,expand,downsampling)

    return refinenet
    


def _make_encoder(
    backbone,
    features,
    use_pretrained,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="project",
    enable_attention_hooks=False
):
    if backbone == "vitl16_384":
        pretrained = _make_pretrained_vitl16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # ViT-L/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb_rn50_384":
        pretrained = _make_pretrained_vitb_rn50_384(
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )  # ViT-H/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb16_384":
        pretrained = _make_pretrained_vitb16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # ViT-B/16 - 84.6% Top1 (backbone)
    elif backbone =='twins_base':
        pretrained = timm.create_model('twins_svt_base',pretrained=True)
        pretrained.head = Identity()
        scratch=_make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )
    elif backbone == "resnet50":
        pretrained = _make_pretrained_resnet50(use_pretrained)
        scratch = _make_scratch(
            [256, 512, 1024, 2048], features, groups=groups, expand=expand
        )  # efficientnet_lite3
    elif backbone=='regionvit':
        pretrained = _make_pretrained_regionvit(use_pretrained)
        scratch = _make_scratch(
            [128, 256, 512, 1024], features, groups=groups, expand=expand
        )
    elif backbone=='slak':
        pretrained = SLaK_base(True)
        scratch = _make_scratch([128,256,512,1024],features, groups=1, expand=False)
    elif backbone=='convnext':
        pretrained = convnext_base(True)
        scratch = _make_scratch(
            [128,256,512,1024],features, groups=1, expand=False)
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = in_shape[0]
    out_shape2 = in_shape[1]
    out_shape3 = in_shape[2]
    out_shape4 = in_shape[3]

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnet50(use_pretrained):
    # resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    return _make_resnet_backbone(resnet)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class SE_Block(nn.Module):
    """Residual convolution module."""

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output

class SE_Block_custom(nn.Module):
    def __init__(self,features,r,activation,bn):
        super().__init__()
        self.bn = bn
        self.group = 1
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(features,features//r,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(features//r,features,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        # print('y1',y.size())
        y = self.excitation(y)
        # print('y2',y.size())
        y = y.view(bs, c, 1, 1)
        # print('y3',y.size())
        res = x * y.expand_as(x)
        # print('final',res.size())
        return res

class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        # print(x.shape)
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        # print((x*gate).shape)        
        return x * gate

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self,features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        downsampling = False,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        
        self.groups = 1
        self.expand = expand
        if(downsampling=='region'):
            self.downsampling = False
            switch = True
        else:
            self.downsampling = downsampling
            switch =False

        out_features = features
        if self.expand == True:
            if(not switch):
                if(features>256):
                    out_features = features // 2
            else:
                if(features>128):
                    out_features = features // 2

        
        if(self.downsampling==True):
            if(features<256):
                out_features = features * 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )


        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.gct =GCT(features,after_relu='True')
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.gct(output)
        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)

        return output
