import torch
import torch.nn as nn

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CLFR(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CLFR, self).__init__()

        # 通道重校准
        self.channel_recalibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),  # 减少通道数
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力机制
        self.spatial_attention = nn.Sequential(
            DepthwiseSeparableConv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 可学习的残差连接权重
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 通道重校准
        ca = self.channel_recalibration(x)
        x_ca = x * ca

        # 空间注意力
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([max_out, avg_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x_sa = x_ca * sa

        # 特征融合
        refined_feature = x_sa + self.alpha * x  # 残差连接，带有可学习的权重

        return refined_feature

# 测试CLFR模块
if __name__ == "__main__":
    input_tensor = torch.randn(2, 1024, 56, 56)  # 假设输入为2个样本，1024个通道，56x56的特征图
    clfr = CLFR(in_channels=1024)
    output = clfr(input_tensor)
    print(output.shape)  # 应输出 (2, 1024, 56, 56)