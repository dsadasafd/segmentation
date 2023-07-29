import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from backbone import resnet50




class get_Intermediatelayer_out(nn.ModuleDict):
    def __init__(self, model, return_layer):
        if not set(return_layer).issubset([name for name,_ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layer = return_layer  # {'layer4:" out, 'layer3': 'aux'} 字典类型
        return_layer = {str(name): str(module) for name, module in return_layer.items()}
        self.return_layer = orig_return_layer

        #  重构backbone
        new_backbone = OrderedDict()
        for name, module in model.named_children():
            new_backbone[name] = module
            if name in return_layer:
                del return_layer[name]
            if not return_layer:
                break

        super().__init__(new_backbone)
        pass
    # 这部分是得到输入新backbone的每一层子模块的中间输出
    def forward(self, x):
        out_features = OrderedDict()
        for name, module in self.items(): # 记得将self.items改成model.named_children()
            x = module(x)
            if name in self.return_layer:
                out_name = self.return_layer[name]
                out_features[out_name] = x
        return out_features # 是一个字典类型的，包含model每个中间层输出的特征图

class FCN(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        result = OrderedDict()
        input_shape = x.shape[-2:]
        features = self.backbone(x) # 这里的features应该指代的是上面模块输出的out_features字典，涵盖了两个值，layer3 和layer4
        x = features['out']  # 这个得到的是（3，4，6，3）后的没进入全连接层的最后输出out,然后将它输出进全卷积网络的卷积层最后
        x = self.classifier(x)  # 这个就是最后的一层卷积层，命名为分类器
        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False) # 插值扩大特征图变回原图大小
        result["out"] = x

        #  这里还要再用上aux层的分类器，所以应该从out_features获取中间层的特征图进入aux分类器
        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x
            pass
        return result # 这个result是一个有序字典，查询最终输出的


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        inter_channels = in_channels // 4
        layer = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1)
        ]

        super().__init__(*layer)
        pass

def fcn_resnet50_model(aux, num_classes=21):
    model = resnet50(replace_conv=[False, True, True])

    return_layer = {'layer4': 'out'}
    if aux:
        return_layer['layer3'] = 'aux'
    backbone = get_Intermediatelayer_out(model, return_layer)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(1024, num_classes)
    classifier = FCNHead(2048, num_classes)
    FCN_model = FCN(backbone, classifier, aux_classifier)

    return FCN_model

if __name__ == '__main__':
    FCN_model = fcn_resnet50_model(True, num_classes=21)
    FCN_model = FCN_model.to('cuda')
    x = torch.rand((4, 3, 224, 224))
    x = x.to("cuda")
    predict = FCN_model(x)
    print(type(predict['out']))
    print(predict['out'].shape)
    print(predict['out'])
    print(predict['out'].argmax(1))

""""""""""""""""""''''''''''''''''''''''''''''''''''''''''''








