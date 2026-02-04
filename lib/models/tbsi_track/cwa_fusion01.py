import torch
from torch import nn


# 自定义维度转换层：替代 nn.Lambda，封装 permute 操作
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


# 自定义特征拼接层：适配SGFA多尺度卷积输出的拼接
class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, feats):
        return torch.cat(feats, dim=self.dim)


# 自定义多尺度卷积层：替代 nn.Lambda，封装多尺度空洞卷积逻辑
class MultiScaleConv(nn.Module):
    def __init__(self, dilated_convs):
        super().__init__()
        self.dilated_convs = dilated_convs

    def forward(self, x):
        return [conv(x) for conv in self.dilated_convs]


class CWA_Fusion(nn.Module):
    """
    红外适配版CWA融合模块（参数与原始版完全对齐）
    输出：4维特征图（B, C, H, W），直接适配跟踪头的Conv2d输入
    """
    # __init__ 参数与原始版一致（无feat_sz_s）
    def __init__(self, dim: int, dilation_rates: list = [1, 6, 12, 18]):
        super().__init__()
        self.dim = dim
        self.dilation_rates = dilation_rates

        self.mpwa = self._build_mpwa()
        self.spatial_weight = self._build_spatial_weight()
        self.sgfa = self._build_sgfa()

    def _build_mpwa(self):
        return nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=True),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(self.dim, eps=1e-5),
            Permute(0, 3, 1, 2),
            nn.Sigmoid()
        )

    def _build_spatial_weight(self):
        return nn.Sequential(
            nn.ModuleList([
                nn.AdaptiveAvgPool2d(1),
                nn.AdaptiveMaxPool2d(1)
            ]),
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def _build_sgfa(self):
        dilated_convs = nn.ModuleList([
            nn.Conv2d(
                self.dim, self.dim,
                kernel_size=3,
                padding=d,
                dilation=d,
                bias=False
            ) for d in self.dilation_rates
        ])

        return nn.Sequential(
            MultiScaleConv(dilated_convs),
            Concat(dim=1),
            nn.Conv2d(self.dim * len(self.dilation_rates), self.dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )

    # 核心修复：添加 feat_sz_s 参数到 forward 方法
    def forward(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor, feat_sz_s: int) -> torch.Tensor:
        """
        前向传播：输入Token格式→输出4维特征图（适配跟踪头Conv2d）
        Args:
            rgb_feat: RGB Token特征 (B, HW, C)
            tir_feat: TIR Token特征 (B, HW, C)
            feat_sz_s: 搜索区特征图尺寸（如16，对应H=W=16）
        Returns:
            final_fused_map: 融合后的4维特征图 (B, C, H, W)
        """
        B, HW, C = rgb_feat.shape

        # 1. Token→特征图转换（核心：从3维Token→4维特征图）
        def token2feat(token):
            return token.permute(0, 2, 1).view(B, C, feat_sz_s, feat_sz_s)  # (B,HW,C)→(B,C,H,W)

        rgb_map = token2feat(rgb_feat)  # (B, 768, 16, 16)
        tir_map = token2feat(tir_feat)  # (B, 768, 16, 16)

        # 2. MPWA像素级权重分配
        cat_feat = torch.cat([rgb_map, tir_map], dim=1)  # (B, 1536, 16, 16)
        mpwa_weight = self.mpwa(cat_feat)  # (B, 768, 16, 16)
        rgb_weighted = rgb_map * mpwa_weight
        tir_weighted = tir_map * (mpwa_weight.new_ones(mpwa_weight.shape) - mpwa_weight)

        # 3. 空间权重（GAP+GMP）
        temp_fused = rgb_weighted + tir_weighted  # (B, 768, 16, 16)
        gap_layer, gmp_layer = self.spatial_weight[0]
        gap_feat = gap_layer(temp_fused)  # (B, 768, 1, 1)
        gmp_feat = gmp_layer(temp_fused)  # (B, 768, 1, 1)
        spatial_weight = self.spatial_weight[1](torch.cat([gap_feat, gmp_feat], dim=1))  # (B,768,1,1)

        rgb_spatial = rgb_weighted * spatial_weight
        tir_spatial = tir_weighted * spatial_weight

        # 4. 融合+SGFA多尺度聚合
        init_fused = rgb_spatial + tir_spatial  # (B, 768, 16, 16)
        sgfa_fused = self.sgfa(init_fused)  # (B, 768, 16, 16)
        final_fused_map = sgfa_fused + init_fused  # (B, 768, 16, 16) → 4维特征图

        return final_fused_map