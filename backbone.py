import torch
import torch.nn as nn
from torchsummary import summary


if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    #print(device_name)
    pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)


class Bottleneck(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation_rate=1):
    # 继承模块接口
    super().__init__()
    # 残差块内容
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation_rate, dilation=dilation_rate)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1)
    self.bn3 = nn.BatchNorm2d(out_channels*4)
    self.relu = nn.ReLU(inplace=True)
    # 判断是否需要下采样
    self.downsample = downsample
    pass

  def forward(self, inputs):
    identity = inputs
    # 进入残差块的第一个卷积层
    out = self.conv1(inputs)
    out = self.bn1(out)
    out = self.relu(out)
    # 进入残差块的第二个卷积层
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    # 进入残差块的第三个卷积层
    out = self.conv3(out)
    out = self.bn3(out)
    # 判断是否需要边缘下采样
    if self.downsample is not None:
        identity = self.downsample(identity)
    # 最后输出
    out = out + identity
    final_out = self.relu(out)

    return final_out
  pass

class Resnet50(nn.Module):
    def __init__(self, bottleneck_nums=(3,4,6,3), class_num=21, replace_conv=None):
        super().__init__()

        if replace_conv is None:
            replace_conv = [False, False, False]
            pass
        self.in_channels = 64
        self.dilation_rate = 1

        if len(replace_conv) != 3:
            raise ValueError("replace_stride_with_dilation should be None " "or a 3-element tuple, got {}".format(replace_conv))

        # 网络主体初始的卷积层，池化层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 网络主力结构， （3，4，6，3）个数的残差主体
        self.layer1 = self._make_block(64, bottleneck_nums[0])
        self.layer2 = self._make_block(128, bottleneck_nums[1], stride=2, replace_conv=replace_conv[0])
        self.layer3 = self._make_block(256, bottleneck_nums[2], stride=2, replace_conv=replace_conv[1])
        self.layer4 = self._make_block(512, bottleneck_nums[3], stride=2, replace_conv=replace_conv[2])

        # 最后的全连接层和均值池化
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, class_num)



    def _make_block(self, out_channels, block_num, stride=1, replace_conv=False):

        pre_dilation_rate = self.dilation_rate
        if replace_conv:
            self.dilation_rate = stride * self.dilation_rate
            stride = 1
            pass

        downsample = None
        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )


        layer = []
        layer.append(Bottleneck(self.in_channels, out_channels, stride, downsample=downsample, dilation_rate=pre_dilation_rate))
        self.in_channels = out_channels*4
        for i in range(1, block_num):
            layer.append(Bottleneck(self.in_channels, out_channels, dilation_rate=self.dilation_rate))
            pass
        return nn.Sequential(*layer)

    def forward(self, inputs):
        # 预进入层
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # 残差1
        out = self.layer1(out)
        # 残差2
        out = self.layer2(out)
        # 残差3
        out = self.layer3(out)
        # 残差4
        out = self.layer4(out)
        # 预输出层
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        final_out = self.fc(out)

        return final_out
########################################

def resnet50(**kwargs):
    model = Resnet50(**kwargs)
    model.to(device)
    return model

if __name__ == '__main__':
    resnet50_model = resnet50(replace_conv=[False, True, True])
    print(resnet50_model)
    print(summary(resnet50_model, (3,244,244)))
    print(device)
