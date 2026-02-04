import torch.nn as nn
import torch.nn.functional as F


class RGB_Prompt_Block(nn.Module):
    def __init__(self, inplanes, hide_channel, drop, init_method="conservative"):
        super(RGB_Prompt_Block, self).__init__()
        self.inplanes = inplanes
        self.hide_channel = hide_channel
        self.init_method = init_method

        # 网络层定义（新增通道注意力模块）
        self.conv0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)

        # 并行卷积分支
        self.conv1_0 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=3, stride=1,
                                 padding=1, groups=hide_channel)  # 深度卷积：局部纹理
        self.conv1_1 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=1, stride=1,
                                 padding=0)  # 1x1卷积：通道交互
        # 通道注意力模块（在conv1_1后添加）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化：(H,W)->1x1
            nn.Conv2d(hide_channel, hide_channel // 4, kernel_size=1, stride=1),  # 通道压缩
            nn.GELU(),
            nn.Conv2d(hide_channel // 4, hide_channel, kernel_size=1, stride=1),  # 通道恢复
            nn.Sigmoid()  # 生成注意力权重（0-1）
        )

        self.conv1_2 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=3, stride=1,
                                 dilation=2, padding=2)  # 膨胀卷积：扩大感受野
        self.dropout = nn.Dropout(drop)
        self.act_layer = nn.Sequential(nn.BatchNorm2d(hide_channel), nn.GELU())
        self.conv2 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)

        # 动态增强强度：可学习缩放因子（初始化为0.1，避免增强过度）
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        if self.init_method == "conservative":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.kernel_size == (1, 1):
                        nn.init.normal_(m.weight, mean=0, std=0.01)
                    else:
                        nn.init.normal_(m.weight, mean=0, std=0.03)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)
            # 输出层特殊处理
            nn.init.normal_(self.conv2.weight, mean=0, std=0.005)

        elif self.init_method == "xavier":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_ori = x  # 保留原始特征
        x = self.conv0(x)  # 通道压缩

        # 并行分支计算（conv1_1后添加通道注意力）
        conv1_0_out = self.conv1_0(x)
        conv1_1_out = self.conv1_1(x)
        # 通道注意力作用于conv1_1的输出（聚焦有效通道）
        attn_weight = self.channel_attn(conv1_1_out)  # 生成通道权重
        conv1_1_out = conv1_1_out * attn_weight  # 通道加权
        conv1_2_out = self.conv1_2(x)

        # 多分支特征融合
        x_enhanced = conv1_0_out + conv1_1_out + conv1_2_out
        x_enhanced = self.act_layer(self.dropout(x_enhanced))  # 激活与 dropout
        x_enhanced = self.conv2(x_enhanced)  # 通道恢复

        # 动态调整增强强度（通过可学习的alpha缩放增强特征）
        x = x_ori + self.alpha * x_enhanced

        return F.relu(x)