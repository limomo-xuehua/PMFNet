import torch.nn as nn
import torch.nn.functional as F


class RGB_Prompt_Block(nn.Module):
    def __init__(self, inplanes, hide_channel, drop, init_method="conservative"):
        super(RGB_Prompt_Block, self).__init__()
        self.inplanes = inplanes
        self.hide_channel = hide_channel
        self.init_method = init_method

        # 网络层定义
        self.conv0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1_0 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=3, stride=1,
                                 padding=1, groups=hide_channel)
        self.conv1_1 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=1, stride=1,
                                 padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=hide_channel, out_channels=hide_channel, kernel_size=3, stride=1,
                                 dilation=2, padding=2)
        self.dropout = nn.Dropout(drop)
        self.act_layer = nn.Sequential(nn.BatchNorm2d(hide_channel), nn.GELU())
        self.conv2 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)

        # 在定义完所有层后立即初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """直接在模块内部集成初始化方法"""
        if self.init_method == "conservative":
            # 保守初始化策略
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

            # 对输出层特别处理
            nn.init.normal_(self.conv2.weight, mean=0, std=0.005)

        elif self.init_method == "xavier":
            # Xavier初始化
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_ori = x
        x = self.conv0(x)
        x_enhanced = self.conv1_1(x) + self.conv1_0(x) + self.conv1_2(x)
        x_enhanced = self.act_layer(self.dropout(x_enhanced))
        x_enhanced = self.conv2(x_enhanced)
        x = x_ori + x_enhanced
        return F.relu(x)