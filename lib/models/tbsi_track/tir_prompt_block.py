import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2


class TIR_Prompt_Block(nn.Module):
    """
    简化版TIR专用特征增强模块（类名已改为Tir_Prompt_Block）
    核心功能：热对比度增强 + 热显著性检测 + 显著性引导频域滤波
    """

    def __init__(self, inplanes, hide_channel, drop):
        super(TIR_Prompt_Block, self).__init__()
        self.inplanes = inplanes
        self.hide_channel = hide_channel

        # 1. 热对比度预增强模块
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(inplanes, hide_channel, 3, padding=1),
            nn.BatchNorm2d(hide_channel),
            nn.GELU(),
            nn.Conv2d(hide_channel, inplanes, 3, padding=1),
            nn.Sigmoid()  # 输出热对比度权重 [0,1]
        )

        # 2. 热显著性检测模块
        self.saliency_detector = nn.Sequential(
            nn.Conv2d(inplanes, hide_channel, 3, padding=1),
            nn.BatchNorm2d(hide_channel),
            nn.GELU(),
            nn.Conv2d(hide_channel, hide_channel // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hide_channel // 2, 1, 1),  # 输出单通道显著图
            nn.Sigmoid()
        )

        # 3. 显著性引导的频域滤波
        self.saliency_guided_filter = nn.Sequential(
            nn.Conv2d(1, hide_channel // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hide_channel // 4, 2, 1),  # 输出实部和虚部的滤波权重
            nn.Softmax(dim=1)
        )

        # 基础后处理
        self.dropout = nn.Dropout2d(drop)
        self.bn_post = nn.BatchNorm2d(inplanes)
        self.conv_out = nn.Conv2d(inplanes, inplanes, 1)

        # 在定义完所有层后立即进行Xavier初始化
        self._init_weights()

    def _init_weights(self):
        """使用Xavier/Glorot初始化所有可学习参数"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Xavier均匀分布初始化卷积层
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm的标准初始化
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        # 特别处理输出层，使用更小的初始化以确保残差连接的稳定性
        # 这里我们仍然使用Xavier但调整gain值
        nn.init.xavier_uniform_(self.conv_out.weight, gain=0.1)

    def forward(self, x):
        identity = x  # 残差连接

        # --------------------------
        # 1. 热对比度预增强
        # --------------------------
        contrast_weights = self.contrast_enhancer(x)  # [B, inplanes, H, W]
        x_enhanced = x * (1 + contrast_weights)  # 增强热对比度

        # 输入标准化
        x_norm = F.layer_norm(x_enhanced, x_enhanced.shape[1:])

        # --------------------------
        # 2. 热显著性检测
        # --------------------------
        saliency_map = self.saliency_detector(x_norm)  # [B, 1, H, W]

        # --------------------------
        # 3. 显著性引导的频域滤波
        # --------------------------
        # 生成频域滤波权重
        filter_weights = self.saliency_guided_filter(saliency_map)  # [B, 2, H, W]

        # 频域变换
        x_fft = fft2(x_norm)
        real = x_fft.real  # 实部：热辐射分布
        imag = x_fft.imag  # 虚部：热边缘信息

        # 显著性引导滤波
        real_filtered = real * filter_weights[:, 0:1, :, :]  # 热分布区域增强
        imag_filtered = imag * filter_weights[:, 1:2, :, :]  # 热边缘区域增强

        # 重组复数并逆变换
        x_complex = torch.complex(real_filtered, imag_filtered)
        x_ifft = ifft2(x_complex).real

        # 显著性后调制
        x_saliency_modulated = x_ifft * (1 + saliency_map)

        # 基础后处理
        x_out = self.dropout(x_saliency_modulated)
        x_out = self.bn_post(x_out)
        x_out = self.conv_out(x_out)

        # 残差连接
        return F.relu(identity + x_out)