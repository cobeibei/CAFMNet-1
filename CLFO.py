import torch
import torch.nn as nn

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CLFO(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=8):
        super(CLFO, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0),
            nn.GroupNorm(groups, out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x += self.residual_connection(identity)
        x = self.relu(x)
        return x

# 测试CLFO模块
if __name__ == "__main__":
    input_tensor = torch.randn(2, 64, 56, 56)  # 假设输入为2个样本，64个通道，56x56的特征图
    clfo = CLFO(in_channels=64, out_channels=64)
    output = clfo(input_tensor)
    print(output.shape)  # 应输出 (2, 64, 56, 56)