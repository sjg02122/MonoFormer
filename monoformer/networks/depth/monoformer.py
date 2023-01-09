import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
    forward_region,
    forward_twins,
    forward_slak,
    forward_convnext,
    _make_refinenet,
    _make_conv,
    _make_DA,
)

import numpy as np
from monoformer.networks.depth.regionvit import regionvit_base_224


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


class monoformer_base(BaseModel):
    def __init__(
        self,
        head,
        backbone,
        features=256,
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):
        super(monoformer_base, self).__init__()
        self.channels_last = channels_last
        self.features = features
        self.backbone = backbone
        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
            "resnet50" :[0,0,0,0],
            'twins_base' : [0,1,2,3],
            "slak":[0,0,0,0],
            'convnext':[0,0,0,0],
        }
        self.expand = True
        self.downsampling = False
        

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=self.expand,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.relu = nn.ReLU()
        self.scratch.output_conv = head
        print("Current Backbone : ",backbone)

        if(backbone=='vitb_rn50_384'):
            refine_features=[256,512,1024,2048]
            in_features = [256,512,768,768]
            out_features = [256,512,1024,2048]
        elif(backbone=='vitb16_384'):
            refine_features=[256,512,1024,2048]
            in_features = [96,192,384,768]
            out_features = [256,512,1024,2048]
        elif(backbone=='vitl16_384'):
            refine_features=[256,512,1024,2048]
            in_features = [256,512,768,768]
            out_features = [256,512,1024,2048]
        elif(backbone=='resnet50'):
            refine_features=[256,512,1024,2048]
            in_features = [256,512,1024,2048]
            out_features = [256,512,1024,2048]
        elif(backbone=='twins_base'):
            refine_features=[256,512,1024,2048]
            in_features = [96,192,384,768]
            out_features = [256,512,1024,2048] 
        elif(backbone=='slak'):
            refine_features=[256,512,1024,2048]
            in_features = [128,256,512,1024]
            out_features = [256,512,1024,2048]
        elif(backbone=='convnext'):
            refine_features=[256,512,1024,2048]
            in_features = [128,256,512,1024]
            out_features = [256,512,1024,2048]


        self.refinenet = _make_refinenet(refine_features,use_bn,self.expand,self.downsampling) if backbone!='twins_base' else  _make_refinenet(refine_features,use_bn,self.expand,'region')
        self.conv_layer = _make_conv(in_features,out_features)
        self.DA = _make_DA(in_features)





    def forward(self, x):

        if(self.backbone=='regionvit'):
            layer_1 , layer_2 , layer_3 , layer_4 = forward_region(self.pretrained,x)
        elif(self.backbone=='twins_base'):
            layer_1 , layer_2 , layer_3 , layer_4 = forward_twins(self.pretrained,x)
        elif(self.backbone =='resnet50'):
            layer_1 = self.pretrained.layer1(x)
            layer_2 = self.pretrained.layer2(layer_1)
            layer_3 = self.pretrained.layer3(layer_2)
            layer_4 = self.pretrained.layer4(layer_3)
        elif(self.backbone == 'slak'):
            layer_1 , layer_2 , layer_3 , layer_4 = forward_slak(self.pretrained,x)
        elif(self.backbone=='convnext'):
            layer_1 , layer_2 , layer_3 , layer_4 = forward_convnext(self.pretrained,x)
        else:
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)


        DA1 = self.DA.layer1(layer_1)
        DA2 = self.DA.layer2(layer_2)
        DA3 = self.DA.layer3(layer_3)
        DA4 = self.DA.layer4(layer_4)
        
        layer_1_rn = layer_1_rn + layer_1 + DA1
        layer_1_rn = self.relu(layer_1_rn)
 
        layer_2_rn = layer_2_rn + layer_2 +DA2
        layer_2_rn = self.relu(layer_2_rn)

        layer_3_rn = layer_3_rn + layer_3 +DA3
        layer_3_rn = self.relu(layer_3_rn)
        
        layer_4_rn = layer_4_rn + layer_4 +DA4
        layer_4_rn = self.relu(layer_4_rn)

        layer_1_rn = self.conv_layer.layer1(layer_1_rn)
        layer_2_rn = self.conv_layer.layer2(layer_2_rn)
        layer_3_rn = self.conv_layer.layer3(layer_3_rn)
        layer_4_rn = self.conv_layer.layer4(layer_4_rn)     
        
        path_4 = self.refinenet.layer4(layer_4_rn)
        path_3 = self.refinenet.layer3(path_4,layer_3_rn)
        path_2 = self.refinenet.layer2(path_3,layer_2_rn)
        path_1 = self.refinenet.layer1(path_2,layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class monoformer(monoformer_base):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256
        self.scale = scale
        self.shift = shift
        self.invert = invert
        path = None
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        inv_depth = super().forward(x).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


