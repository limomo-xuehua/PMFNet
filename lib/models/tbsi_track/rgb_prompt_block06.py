import torch
import torch.nn as nn
import torch.nn.functional as F

"""
模块参数说明（依据文章3.4节与4.7节消融实验配置）：
- inplanes: 输入特征图的通道数（需与ViT骨干网络输出通道一致，如ViT-Base设为768，ViT-Tiny设为192）
- hide_channel: 空间增强阶段的中间通道数（文章隐含设置为输入通道的1/6~1/4，推荐768输入对应128，192输入对应48）
- drop: Dropout概率（文章4.7节验证0.1为最优，平衡过拟合抑制与特征表达能力）
功能定位：实现空间-通道协同的特征增强，为两阶段单向融合（TUF）提供高质量单模态特征（对应文章图3的SCIP结构）
"""

class RGB_Prompt_Block(nn.Module):
    def __init__(self, inplanes, hide_channel, drop=0.1):
        super(RGB_Prompt_Block, self).__init__()
        # -------------------------- 1. 空间维度增强（对应文章公式5） --------------------------
        # 输入特征低维映射：降低计算复杂度，聚焦关键空间特征
        self.spatial_proj = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        # 多感受野卷积分支：全局特征（1×1）、局部通道私有特征（3×3深度可分离）、扩大感受野特征（3×3空洞）
        self.conv_global = nn.Conv2d(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0)  # 全局语义特征提取
        self.conv_local = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                    padding=1, groups=hide_channel)  # 局部通道私有特征（深度可分离卷积）
        self.conv_dilated = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                      dilation=2, padding=2)  # 扩大感受野（空洞率2，保持输出尺寸）
        # 空间特征融合后处理：归一化+激活+正则化（文章隐含BatchNorm与GELU）
        self.spatial_post = nn.Sequential(
            nn.BatchNorm2d(hide_channel),
            nn.GELU(),
            nn.Dropout(drop)
        )
        # 空间增强特征恢复：将中间通道映射回输入通道数，适配后续残差连接
        self.spatial_recover = nn.Conv2d(hide_channel, inplanes, kernel_size=1, stride=1, padding=0)

        # -------------------------- 2. 通道维度增强（对应文章公式6） --------------------------
        # 通道统计信息提取：全局平均池化（GAP）+全局最大池化（GMP），捕捉通道重要性
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        # 挤压-激励（Squeeze-and-Excitation）层：生成通道注意力权重
        self.channel_squeeze = nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, padding=0)  # 通道压缩至1/4
        self.channel_excite = nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, padding=0)  # 通道恢复
        self.channel_post = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(drop)
        )

    def forward(self, x):
        """
        输入x: 单模态特征图，维度为[batch_size, inplanes, height, width]（如RGB或TIR的Transformer输出特征）
        输出x_final: 增强后的单模态特征图，维度与输入一致
        """
        x_ori = x  # 保存原始特征，用于最终残差连接（避免特征分布偏移，对应文章残差设计）
        batch, _, h, w = x.shape

        # 阶段1：空间维度特征增强（多尺度融合）
        x_spatial = self.spatial_proj(x)  # 低维映射：[B, C, H, W] → [B, hide_C, H, W]
        # 多感受野特征融合：全局+局部+扩大感受野特征逐元素相加
        x_spatial = self.conv_global(x_spatial) + self.conv_local(x_spatial) + self.conv_dilated(x_spatial)
        x_spatial = self.spatial_post(x_spatial)  # 归一化+激活+正则化
        x_spatial = self.spatial_recover(x_spatial)  # 恢复通道数：[B, hide_C, H, W] → [B, C, H, W]
        x_spatial_enhanced = x_ori + x_spatial  # 空间增强特征与原始特征残差融合

        # 阶段2：通道维度特征增强（注意力加权）
        # 提取通道统计信息并融合
        x_gap = self.gap(x_spatial_enhanced)  # [B, C, 1, 1]
        x_gmp = self.gmp(x_spatial_enhanced)  # [B, C, 1, 1]
        x_channel = self.channel_squeeze(x_gap + x_gmp)  # 融合统计信息并压缩通道
        x_channel = self.channel_post(x_channel)
        # 生成通道注意力权重（Sigmoid确保权重在[0,1]）
        x_channel_weight = torch.sigmoid(self.channel_excite(x_channel))  # [B, C, 1, 1]
        # 通道加权：权重广播至特征图维度，强化重要通道特征
        x_channel_enhanced = x_spatial_enhanced * x_channel_weight  # [B, C, H, W]

        # 最终残差连接：原始特征与增强特征融合，保留基础语义信息
        x_final = x_ori + x_channel_enhanced
        return F.relu(x_final)
