import torch
import torch.nn as nn
from torch.nn import init
from CBAM import CBAM 

class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0, kernel_size=1, stride=1, padding=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=kernel_size, stride=stride, padding=padding)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                                   nn.BatchNorm2d(high_dim), )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=kernel_size, stride=2, padding=padding)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=kernel_size, stride=2, padding=padding),
                                   nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    # 保持forward方法不变

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h

        return z

class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2, kernel_size=1, stride=1, padding=0):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio
        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=kernel_size, stride=stride, padding=padding)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=kernel_size, stride=stride, padding=padding)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=kernel_size, stride=stride, padding=padding)
        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).reshape(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).reshape(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).reshape(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MDFO(nn.Module):
    def __init__(self, high_dim, low_dim, flag, dropout_rate=0.1):
        super(MDFO, self).__init__()
        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)
        self.cbam = CBAM(high_dim)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 可学习的权重
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, x0):
        z_cnl = self.CNL(x, x0)
        z_pnl = self.PNL(z_cnl, x0)
        z_cbam = self.cbam(z_pnl)  # 应用CBAM模块

        # 加权融合
        z = self.fusion_weight * z_cbam + (1 - self.fusion_weight) * x
        z = self.dropout(z)  # 应用Dropout

        return z