import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2


class RGB_Prompt_Block(nn.Module):
    """
    改进版RGB频域特征增强模块（实部/虚部差异化卷积策略）
    核心优化：针对实部/虚部特征差异设计专属卷积策略，增强针对性
    - 实部：聚焦低频全局轮廓，强化一致性，避免背景干扰
    - 虚部：聚焦高频细节+噪声，精准捕捉细节并抑制噪声
    """

    def __init__(self, inplanes, hide_channel, drop):
        super(RGB_Prompt_Block, self).__init__()
        self.inplanes = inplanes
        self.hide_channel = hide_channel
        # 确保虚部分组卷积的分组数可整除（避免报错）
        self.imag_group = hide_channel // 4 if hide_channel >= 4 else 1

        # 输入标准化层
        self.bn_in = nn.BatchNorm2d(inplanes)

        # --------------------------
        # 实部分支（低频全局轮廓）：差异化卷积策略
        # 核心：小感受野+无膨胀，强化轮廓一致性，避免背景干扰
        # --------------------------
        self.real_conv0 = nn.Conv2d(inplanes, hide_channel, kernel_size=1, stride=1, padding=0)  # 通道压缩
        self.real_conv1_0 = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                      padding=1, groups=hide_channel)  # 分组卷积：保持局部轮廓精细度
        self.real_conv1_1 = nn.Conv2d(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0)  # 1x1卷积：融合通道一致性
        # 替换为小感受野普通卷积（dilation=1），避免膨胀卷积引入背景干扰
        self.real_conv1_2 = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                      dilation=1, padding=1, groups=1)  # 普通卷积：强化全局轮廓连贯性
        self.real_act = nn.Sequential(nn.BatchNorm2d(hide_channel), nn.GELU())
        self.real_conv2 = nn.Conv2d(hide_channel, inplanes, kernel_size=1, stride=1, padding=0)  # 通道恢复

        # 实部自适应融合注意力层（保持原有结构，适配轮廓权重学习）
        self.real_fusion_attn = nn.Sequential(
            nn.Conv2d(hide_channel * 3, hide_channel, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hide_channel, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # --------------------------
        # 虚部分支（高频细节噪声）：差异化卷积策略
        # 核心：灵活感受野+噪声抑制，精准捕捉有效细节
        # --------------------------
        self.imag_conv0 = nn.Conv2d(inplanes, hide_channel, kernel_size=1, stride=1, padding=0)  # 通道压缩
        # 调整分组数（减少分组，避免细节割裂），聚焦局部细节捕捉
        self.imag_conv1_0 = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                      padding=1, groups=self.imag_group)  # 分组卷积：精细捕捉细节单元
        self.imag_conv1_1 = nn.Conv2d(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0)  # 1x1卷积：筛选有效细节
        self.imag_conv1_2 = nn.Conv2d(hide_channel, hide_channel, kernel_size=3, stride=1,
                                      dilation=2, padding=2)  # 膨胀卷积：扩大感受野，关联分散细节
        # 新增：轻量噪声抑制层（基于像素方差的细节注意力）
        self.imag_noise_suppress = nn.Sequential(
            nn.Conv2d(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()  # 生成噪声抑制权重（0~1）
        )
        self.imag_act = nn.Sequential(nn.BatchNorm2d(hide_channel), nn.GELU())
        self.imag_conv2 = nn.Conv2d(hide_channel, inplanes, kernel_size=1, stride=1, padding=0)  # 通道恢复

        # 虚部自适应融合注意力层（保持原有结构，适配细节-噪声区分权重）
        self.imag_fusion_attn = nn.Sequential(
            nn.Conv2d(hide_channel * 3, hide_channel, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hide_channel, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 后处理层（保持原有）
        self.dropout = nn.Dropout2d(drop)
        self.bn_post = nn.BatchNorm2d(inplanes)
        self.conv_out = nn.Conv2d(inplanes, inplanes, kernel_size=1)

        # 初始化（确保新增层参数合理）
        self._init_weights()

    def _init_weights(self):
        """补充新增层的初始化，保证训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
        # 输出层轻量化初始化，避免残差融合失真
        nn.init.xavier_uniform_(self.conv_out.weight, gain=0.1)

    def forward(self, x):
        identity = x  # 残差连接的原始输入

        # 1. 输入标准化
        x_norm = self.bn_in(x)

        # 2. 快速傅里叶变换（转换至频域）
        x_fft = fft2(x_norm)
        real = x_fft.real  # 实部特征：侧重低频全局轮廓
        imag = x_fft.imag  # 虚部特征：侧重高频细节噪声

        # --------------------------
        # 3. 实部分支处理（全局轮廓增强）
        # --------------------------
        real = self.real_conv0(real)  # 通道压缩

        # 3.1 小感受野三路径卷积（无膨胀，避免背景干扰）
        real_conv0_out = self.real_conv1_0(real)  # 分组卷积：局部轮廓
        real_conv1_out = self.real_conv1_1(real)  # 1x1卷积：通道融合
        real_conv2_out = self.real_conv1_2(real)  # 普通卷积：全局连贯性

        # 3.2 自适应权重融合（聚焦轮廓重要性）
        real_concat = torch.cat([real_conv0_out, real_conv1_out, real_conv2_out], dim=1)
        real_weights = self.real_fusion_attn(real_concat)
        real_multi = (real_conv0_out * real_weights[:, 0:1, :, :] +
                      real_conv1_out * real_weights[:, 1:2, :, :] +
                      real_conv2_out * real_weights[:, 2:3, :, :])

        # 3.3 激活与通道恢复
        real_multi = self.real_act(real_multi)
        real_processed = self.real_conv2(real_multi)

        # --------------------------
        # 4. 虚部分支处理（细节捕捉+噪声抑制）
        # --------------------------
        imag = self.imag_conv0(imag)  # 通道压缩

        # 4.1 灵活感受野三路径卷积（捕捉细节+关联全局）
        imag_conv0_out = self.imag_conv1_0(imag)  # 分组卷积：局部细节
        imag_conv1_out = self.imag_conv1_1(imag)  # 1x1卷积：细节筛选
        imag_conv2_out = self.imag_conv1_2(imag)  # 膨胀卷积：细节关联

        # 新增：噪声抑制（对每个卷积输出施加抑制权重，过滤孤立噪声）
        noise_weight = self.imag_noise_suppress(imag)  # 生成噪声抑制权重
        imag_conv0_out = imag_conv0_out * noise_weight
        imag_conv1_out = imag_conv1_out * noise_weight
        imag_conv2_out = imag_conv2_out * noise_weight

        # 4.2 自适应权重融合（区分有效细节与噪声）
        imag_concat = torch.cat([imag_conv0_out, imag_conv1_out, imag_conv2_out], dim=1)
        imag_weights = self.imag_fusion_attn(imag_concat)
        imag_multi = (imag_conv0_out * imag_weights[:, 0:1, :, :] +
                      imag_conv1_out * imag_weights[:, 1:2, :, :] +
                      imag_conv2_out * imag_weights[:, 2:3, :, :])

        # 4.3 激活与通道恢复
        imag_multi = self.imag_act(imag_multi)
        imag_processed = self.imag_conv2(imag_multi)

        # 5. 复数重组
        x_complex = torch.complex(real_processed, imag_processed)

        # 6. 空域还原
        x_ifft = ifft2(x_complex).real

        # 7. 后处理
        x_out = self.dropout(x_ifft)
        x_out = self.bn_post(x_out)
        x_out = self.conv_out(x_out)

        # 8. 残差融合与输出
        return F.relu(identity + x_out)