import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2


class RGB_Prompt_Block(nn.Module):
    """
    改进版RGB频域特征增强模块（移除频域门控机制）
    保留：实部/虚部分支的自适应多尺度卷积权重学习
    移除：动态权重系统（频域门控）
    """

    def __init__(self, inplanes, hide_channel, drop):
        super(RGB_Prompt_Block, self).__init__()
        self.inplanes = inplanes
        self.hide_channel = hide_channel

        # 输入标准化层
        self.bn_in = nn.BatchNorm2d(inplanes)

        # --------------------------
        # 多尺度卷积分支（实部处理）
        # --------------------------
        self.real_conv0 = nn.Conv2d(inplanes, hide_channel, kernel_size=1, stride=1, padding=0)  # 通道压缩
        self.real_conv1_0 = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                      padding=1, groups=hide_channel)  # 分组卷积（局部特征）
        self.real_conv1_1 = nn.Conv2d(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0)  # 1x1卷积（特征融合）
        self.real_conv1_2 = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                      dilation=2, padding=2)  # 膨胀卷积（扩大感受野）
        self.real_act = nn.Sequential(nn.BatchNorm2d(hide_channel), nn.GELU())
        self.real_conv2 = nn.Conv2d(hide_channel, inplanes, kernel_size=1, stride=1, padding=0)  # 通道恢复

        # 实部多尺度卷积的自适应融合注意力层
        self.real_fusion_attn = nn.Sequential(
            nn.Conv2d(hide_channel * 3, hide_channel, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hide_channel, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # --------------------------
        # 多尺度卷积分支（虚部处理）
        # --------------------------
        self.imag_conv0 = nn.Conv2d(inplanes, hide_channel, kernel_size=1, stride=1, padding=0)  # 通道压缩
        self.imag_conv1_0 = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                      padding=1, groups=hide_channel)  # 分组卷积（局部细节）
        self.imag_conv1_1 = nn.Conv2d(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0)  # 1x1卷积（特征融合）
        self.imag_conv1_2 = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                      dilation=2, padding=2)  # 膨胀卷积（扩大感受野）
        self.imag_act = nn.Sequential(nn.BatchNorm2d(hide_channel), nn.GELU())
        self.imag_conv2 = nn.Conv2d(hide_channel, inplanes, kernel_size=1, stride=1, padding=0)  # 通道恢复

        # 虚部多尺度卷积的自适应融合注意力层
        self.imag_fusion_attn = nn.Sequential(
            nn.Conv2d(hide_channel * 3, hide_channel, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hide_channel, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 移除：动态权重系统（频域门控相关参数）

        # 后处理层
        self.dropout = nn.Dropout2d(drop)
        self.bn_post = nn.BatchNorm2d(inplanes)
        self.conv_out = nn.Conv2d(inplanes, inplanes, kernel_size=1)

    def forward(self, x):
        identity = x  # 残差连接的原始输入

        # 1. 输入标准化
        x_norm = self.bn_in(x)

        # 2. 快速傅里叶变换（转换至频域）
        x_fft = fft2(x_norm)
        real = x_fft.real  # 实部特征 (B, C, H, W)：侧重低频全局轮廓
        imag = x_fft.imag  # 虚部特征 (B, C, H, W)：侧重高频细节噪声

        # --------------------------
        # 3. 多尺度卷积处理实部特征
        # --------------------------
        real = self.real_conv0(real)  # 通道压缩至hide_channel

        # 3.1 计算3种卷积的输出
        real_conv0_out = self.real_conv1_0(real)  # 分组卷积输出（局部特征）
        real_conv1_out = self.real_conv1_1(real)  # 1x1卷积输出（通道融合）
        real_conv2_out = self.real_conv1_2(real)  # 膨胀卷积输出（全局结构）

        # 3.2 计算实部的自适应融合权重
        real_concat = torch.cat([real_conv0_out, real_conv1_out, real_conv2_out], dim=1)
        real_weights = self.real_fusion_attn(real_concat)

        # 3.3 按权重加权融合
        real_multi = (real_conv0_out * real_weights[:, 0:1, :, :] +
                      real_conv1_out * real_weights[:, 1:2, :, :] +
                      real_conv2_out * real_weights[:, 2:3, :, :])

        # 3.4 后续处理
        real_multi = self.real_act(real_multi)
        real_processed = self.real_conv2(real_multi)

        # --------------------------
        # 4. 多尺度卷积处理虚部特征
        # --------------------------
        imag = self.imag_conv0(imag)  # 通道压缩至hide_channel

        # 4.1 计算3种卷积的输出
        imag_conv0_out = self.imag_conv1_0(imag)
        imag_conv1_out = self.imag_conv1_1(imag)
        imag_conv2_out = self.imag_conv1_2(imag)

        # 4.2 计算虚部的自适应融合权重
        imag_concat = torch.cat([imag_conv0_out, imag_conv1_out, imag_conv2_out], dim=1)
        imag_weights = self.imag_fusion_attn(imag_concat)

        # 4.3 按权重加权融合
        imag_multi = (imag_conv0_out * imag_weights[:, 0:1, :, :] +
                      imag_conv1_out * imag_weights[:, 1:2, :, :] +
                      imag_conv2_out * imag_weights[:, 2:3, :, :])

        # 4.4 后续处理
        imag_multi = self.imag_act(imag_multi)
        imag_processed = self.imag_conv2(imag_multi)

        # 5. 重组为复数特征
        x_complex = torch.complex(real_processed, imag_processed)

        # 移除：频域门控机制相关代码（动态权重生成和复数乘法）

        # 6. 逆傅里叶变换回空域（直接使用处理后的复数特征）
        x_ifft = ifft2(x_complex).real  # 取实部

        # 7. 后处理
        x_out = self.dropout(x_ifft)
        x_out = self.bn_post(x_out)
        x_out = self.conv_out(x_out)

        # 8. 残差连接
        return F.relu(identity + x_out)
