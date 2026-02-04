import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2


class RGB_Prompt_Block(nn.Module):
    """
    结合FSFE频域增强与多尺度卷积的特征处理模块
    先通过FSFE模块进行频域特征增强，再通过多尺度卷积提取细粒度特征
    """

    def __init__(self, inplanes, hide_channel, drop=0.1):
        super(RGB_Prompt_Block, self).__init__()
        # 1. FSFE频域特征增强模块（基于TIR_Prompt_Block修改）
        self.fsfe = nn.Sequential(
            nn.BatchNorm2d(inplanes),  # 输入标准化
            # 频域处理分支
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1),  # 实部处理卷积
            nn.BatchNorm2d(inplanes),
            nn.GELU()
        )

        # 频域虚部专用处理层
        self.fsfe_imag = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1),  # 虚部处理卷积
            nn.BatchNorm2d(inplanes),
            nn.GELU()
        )

        # 频域动态权重系统（增强注意力机制）
        self.freq_weight = nn.Parameter(torch.randn(1, hide_channel, 1, 1))
        self.weight_adapter = nn.Conv2d(hide_channel, inplanes, kernel_size=1)

        # 频域后处理
        self.fsfe_post = nn.Sequential(
            nn.Dropout2d(drop),
            nn.BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, inplanes, kernel_size=1)
        )

        # 2. 多尺度卷积特征提取模块（基于Prompt_Block修改）
        self.multi_scale = nn.Sequential(
            # 通道压缩
            nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0),

            # 多尺度卷积分支
            nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1, padding=1, groups=hide_channel),  # 3x3分组卷积
            nn.Conv2d(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0),  # 1x1逐点卷积
            nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1, dilation=2, padding=2),  # 膨胀卷积（感受野扩大）

            # 特征激活与归一化
            nn.Dropout(drop),
            nn.BatchNorm2d(hide_channel),
            nn.GELU(),

            # 通道恢复
            nn.Conv2d(hide_channel, inplanes, kernel_size=1, stride=1, padding=0)
        )

        # 最终残差调整层
        self.residual_adjust = nn.Conv2d(inplanes, inplanes, kernel_size=1)

    def forward(self, x):
        # 原始输入保留（用于残差连接）
        identity = x

        # --------------------------
        # 第一步：FSFE频域特征增强
        # --------------------------
        x_norm = self.fsfe[0](x)  # 输入标准化

        # 傅里叶变换：转换至频域
        x_fft = fft2(x_norm)
        real = x_fft.real  # 实部
        imag = x_fft.imag  # 虚部

        # 实部和虚部分支处理
        real_processed = self.fsfe[1:](real)  # 共享前序卷积层
        imag_processed = self.fsfe_imag(imag)  # 虚部专用处理

        # 重组为复数特征
        x_complex = torch.complex(real_processed, imag_processed)

        # 动态权重生成（频域注意力）
        B, C, H, W = x_complex.shape
        weight = F.interpolate(self.freq_weight, size=(H, W), mode='bilinear', align_corners=False)
        weight = self.weight_adapter(weight)  # 适配输入通道数

        # 频域门控（复数乘法）
        x_gated = x_complex * weight

        # 逆傅里叶变换回空域 + 后处理
        x_fsfe = ifft2(x_gated).real  # 取实部
        x_fsfe = self.fsfe_post(x_fsfe)

        # 频域增强残差
        x_enhanced = F.relu(identity + x_fsfe)

        # --------------------------
        # 第二步：多尺度卷积特征提取
        # --------------------------
        # 多尺度分支处理（注：此处将三个卷积的输出相加，增强多尺度融合）
        x_multi = self.multi_scale[0](x_enhanced)  # 通道压缩

        # 多尺度卷积融合
        conv1 = self.multi_scale[1](x_multi)  # 3x3分组卷积
        conv2 = self.multi_scale[2](x_multi)  # 1x1卷积
        conv3 = self.multi_scale[3](x_multi)  # 膨胀卷积
        x_multi = conv1 + conv2 + conv3  # 多尺度特征融合

        # 激活与通道恢复
        x_multi = self.multi_scale[4:](x_multi)  # 包含dropout、归一化和通道恢复

        # --------------------------
        # 最终残差融合
        # --------------------------
        out = identity + self.residual_adjust(x_multi)
        return F.relu(out)