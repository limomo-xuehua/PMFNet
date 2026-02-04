import torch
import torch.nn as nn
import torch.nn.functional as F


class TIR_Prompt_Block(nn.Module):
    """
    优化版TIR专用特征增强模块
    核心功能：轻量化热对比度增强 + 简化热显著性检测 + 空域多尺度增强 + 动态强度控制
    """

    def __init__(self, inplanes, hide_channel, drop):
        super(TIR_Prompt_Block, self).__init__()
        self.inplanes = inplanes
        self.hide_channel = hide_channel

        # 1. 热对比度预增强模块（保留核心，简化结构）
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(inplanes, hide_channel, kernel_size=3, padding=1, groups=1),  # 修正：添加逗号，改为普通卷积
            nn.BatchNorm2d(hide_channel),
            nn.GELU(),
            nn.Conv2d(hide_channel, inplanes, kernel_size=1),  # 1x1恢复通道
            nn.Sigmoid()  # 输出热对比度权重 [0,1]
        )

        # 2. 简化版热显著性检测 + 轻量级空间注意力
        self.saliency_detector = nn.Sequential(
            nn.Conv2d(inplanes, hide_channel, kernel_size=3, padding=1),  # 缩减为2层卷积
            nn.GELU(),
            nn.Conv2d(hide_channel, 1, kernel_size=1)  # 直接输出单通道显著图
        )
        # 新增轻量级空间注意力（3x3平均池化增强局部一致性）
        self.spatial_attn = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),  # 平滑空间权重
            nn.Sigmoid()  # 空间注意力权重 [0,1]
        )

        # 3. 空域多尺度增强（替代频域滤波，实现全局-局部特征分离）
        # 1x1卷积：全局通道交互（对应原频域实部的全局分布）
        self.global_conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        # 3x3深度卷积：局部热边缘增强（对应原频域虚部的边缘信息）
        self.local_conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, groups=inplanes)
        # 5x5膨胀卷积：扩大感受野，捕捉全局热分布（替代频域全局特性）
        self.dilated_conv = nn.Conv2d(inplanes, inplanes, kernel_size=5, padding=4, dilation=2, groups=inplanes)
        # 多尺度特征融合
        self.fusion_conv = nn.Conv2d(inplanes * 3, inplanes, kernel_size=1)  # 3路特征融合为原通道

        # 基础后处理
        self.dropout = nn.Dropout2d(drop)
        self.bn_post = nn.BatchNorm2d(inplanes)
        self.conv_out = nn.Conv2d(inplanes, inplanes, kernel_size=1)

        # 动态增强因子（初始0.1，避免增强过度）
        self.beta = nn.Parameter(torch.tensor(0.1))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化，确保轻量化模块的稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.in_channels:  # 深度卷积特殊处理
                    nn.init.xavier_uniform_(m.weight, gain=0.5)  # 深度卷积权重缩放
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
        # 输出层弱初始化，保证残差稳定
        nn.init.xavier_uniform_(self.conv_out.weight, gain=0.1)

    def forward(self, x):
        identity = x  # 残差连接

        # --------------------------
        # 1. 热对比度预增强
        # --------------------------
        contrast_weights = self.contrast_enhancer(x)  # [B, inplanes, H, W]
        x_enhanced = x * (1 + contrast_weights)  # 增强热区与背景差异

        # 输入标准化
        x_norm = F.layer_norm(x_enhanced, x_enhanced.shape[1:])

        # --------------------------
        # 2. 热显著性检测 + 空间注意力
        # --------------------------
        saliency_raw = self.saliency_detector(x_norm)  # [B, 1, H, W]
        saliency_map = self.spatial_attn(saliency_raw)  # 空间平滑后的显著图

        # --------------------------
        # 3. 空域多尺度增强（替代频域操作）
        # --------------------------
        # 多尺度特征提取
        global_feat = self.global_conv(x_norm)  # 全局通道交互
        local_feat = self.local_conv(x_norm)    # 局部热边缘
        dilated_feat = self.dilated_conv(x_norm)  # 全局热分布

        # 特征融合（拼接后1x1卷积压缩）
        multi_scale_feat = torch.cat([global_feat, local_feat, dilated_feat], dim=1)
        multi_scale_feat = self.fusion_conv(multi_scale_feat)

        # 显著性引导调制（聚焦显著区域的多尺度特征）
        x_modulated = multi_scale_feat * (1 + saliency_map)  # 显著区域增强

        # 基础后处理
        x_out = self.dropout(x_modulated)
        x_out = self.bn_post(x_out)
        x_out = self.conv_out(x_out)

        # 动态增强强度控制（残差连接）
        return F.relu(identity + self.beta * x_out)