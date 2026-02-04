import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2


class RGB_Prompt_Block(nn.Module):
    """
    仅保留FSFE频域特征增强模块
    通过频域处理增强特征判别性，移除多尺度卷积部分
    """

    def __init__(self, inplanes, hide_channel, drop=0.1):
        super(RGB_Prompt_Block, self).__init__()
        # FSFE频域特征增强模块
        self.fsfe = nn.Sequential(
            nn.BatchNorm2d(inplanes),  # 输入标准化
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

        # 最终残差调整层（保持维度一致性）
        self.residual_adjust = nn.Conv2d(inplanes, inplanes, kernel_size=1)

    def forward(self, x):
        # 原始输入保留（用于残差连接）
        identity = x

        # --------------------------
        # FSFE频域特征增强
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

        # 最终残差融合
        out = identity + self.residual_adjust(x_fsfe)
        return F.relu(out)
