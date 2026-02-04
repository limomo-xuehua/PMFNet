import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 先定义所有依赖的自定义层（顺序关键） =====================
# 1. 自定义维度转换层：替代 nn.Lambda，封装 permute 操作
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


# 2. 自定义特征拼接层：适配SGFA多尺度卷积输出的拼接
class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, feats):
        return torch.cat(feats, dim=self.dim)


# 3. 自定义多尺度卷积层：替代 nn.Lambda，封装多尺度空洞卷积逻辑（必须在CWA_Fusion前定义）
class MultiScaleConv(nn.Module):
    def __init__(self, dilated_convs):
        super().__init__()
        self.dilated_convs = dilated_convs  # 接收nn.ModuleList类型的空洞卷积列表

    def forward(self, x):
        # 对输入x应用每个空洞卷积，返回多尺度特征列表
        return [conv(x) for conv in self.dilated_convs]


# ===================== MPWA前向函数 =====================
def mpwa_rgbt_forward(R_Res, T_Res, conv_cat, conv_global, linear_compress, conv_fusion, layer_norm):
    """
    MPWA函数式前向逻辑：基于压缩+全局协作双分支的像素级权重分配
    Args:
        R_Res: RGB特征图 (B, C, H, W)
        T_Res: TIR特征图 (B, C, H, W)
        conv_cat: 拼接后降维卷积层 (2C→C)
        conv_global: 全局分支1×1卷积层 (C→C/2)
        linear_compress: 压缩分支线性变换层 (C→1)
        conv_fusion: 融合后恢复维度卷积层 (C/2→C)
        layer_norm: 通道级LayerNorm层
    Returns:
        R_MPWA: 加权后的RGB特征 (B, C, H, W)
        T_MPWA: 加权后的TIR特征 (B, C, H, W)
    """
    B, C, H, W = R_Res.shape

    # Step1: 拼接RGB和TIR特征 + 1×1卷积降维（2C → C）
    F_cat = torch.cat([R_Res, T_Res], dim=1)  # (B, 2C, H, W)
    Fn_Cat = conv_cat(F_cat)  # (B, C, H, W)

    # Step2: 全局分支处理（保留高分辨率特征）
    Fn_Global = conv_global(Fn_Cat)  # (B, C/2, H, W)

    # Step3: 压缩分支处理（通道级压缩+Softmax强化关键信息）
    Fn_Cat_reshaped = Fn_Cat.permute(0, 2, 3, 1)  # (B, H, W, C)
    linear_out = linear_compress(Fn_Cat_reshaped)  # (B, H, W, 1)

    # 修复softmax多维度问题
    B_hw, H_hw, W_hw, C_hw = linear_out.shape
    linear_out_reshaped = linear_out.reshape(B_hw, H_hw * W_hw, C_hw)  # (B, H*W, 1)
    Fn_Compress_reshaped = F.softmax(linear_out_reshaped, dim=1)
    Fn_Compress = Fn_Compress_reshaped.reshape(B_hw, H_hw, W_hw, C_hw)  # (B, H, W, 1)

    Fn_Compress = Fn_Compress.permute(0, 3, 1, 2)  # (B, 1, H, W)

    # Step4: 全局分支 × 压缩分支（像素级乘法）+ 恢复维度
    F_fusion = Fn_Global * Fn_Compress  # (B, C/2, H, W)
    F_fusion = conv_fusion(F_fusion)  # (B, C, H, W)

    # 修复LayerNorm维度不匹配
    F_fusion_reshaped = F_fusion.permute(0, 2, 3, 1)  # (B, H, W, C)
    F_fusion_normed = layer_norm(F_fusion_reshaped)  # 对通道维度C做LayerNorm
    F_fusion = F_fusion_normed.permute(0, 3, 1, 2)  # 还原为(B, C, H, W)

    # Step5: Sigmoid生成像素级权重W_MPWA（0~1）
    W_MPWA = torch.sigmoid(F_fusion)  # (B, C, H, W)

    # Step6: 分配RGB/TIR权重并加权
    alpha_MPWA = W_MPWA  # RGB权重
    beta_MPWA = 1 - W_MPWA  # TIR权重
    R_MPWA = R_Res * alpha_MPWA  # RGB特征加权
    T_MPWA = T_Res * beta_MPWA  # TIR特征加权

    return R_MPWA, T_MPWA


# ===================== CWA融合模块（依赖上述所有定义） =====================
class CWA_Fusion(nn.Module):
    """
    红外适配版CWA融合模块（MPWA改为函数式调用）
    输出：4维特征图（B, C, H, W），直接适配跟踪头的Conv2d输入
    """

    def __init__(self, dim: int, dilation_rates: list = [1, 6, 12, 18]):
        super().__init__()
        self.dim = dim
        self.dilation_rates = dilation_rates

        # ---------------------- 初始化MPWA所需的层 ----------------------
        self.mpwa_conv_cat = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=True)  # 拼接降维
        self.mpwa_conv_global = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=True)  # 全局分支
        self.mpwa_linear_compress = nn.Linear(dim, 1)  # 压缩分支线性层
        self.mpwa_conv_fusion = nn.Conv2d(dim // 2, dim, kernel_size=1, bias=True)  # 融合恢复维度
        self.mpwa_layer_norm = nn.LayerNorm(dim)  # 仅指定通道维度dim

        # ---------------------- 原有层初始化 ----------------------
        self.spatial_weight = self._build_spatial_weight()
        self.sgfa = self._build_sgfa()  # 此处调用MultiScaleConv，需确保该类已定义

    def _build_spatial_weight(self):
        """构建空间权重分支（GAP+GMP）"""
        return nn.Sequential(
            nn.ModuleList([
                nn.AdaptiveAvgPool2d(1),
                nn.AdaptiveMaxPool2d(1)
            ]),
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def _build_sgfa(self):
        """构建SGFA多尺度聚合分支（依赖MultiScaleConv）"""
        # 构建多尺度空洞卷积列表
        dilated_convs = nn.ModuleList([
            nn.Conv2d(
                self.dim, self.dim,
                kernel_size=3,
                padding=d,
                dilation=d,
                bias=False
            ) for d in self.dilation_rates
        ])

        # 组装SGFA模块（顺序：多尺度卷积→拼接→降维→激活）
        return nn.Sequential(
            MultiScaleConv(dilated_convs),  # 此处使用MultiScaleConv，需提前定义
            Concat(dim=1),  # 拼接多尺度特征
            nn.Conv2d(self.dim * len(self.dilation_rates), self.dim, kernel_size=1, bias=True),  # 降维
            nn.ReLU(inplace=True)  # 激活
        )

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

        # 1. Token→特征图转换
        def token2feat(token):
            return token.permute(0, 2, 1).view(B, C, feat_sz_s, feat_sz_s)

        rgb_map = token2feat(rgb_feat)  # (B, dim, 16, 16)
        tir_map = token2feat(tir_feat)  # (B, dim, 16, 16)

        # 2. 调用MPWA函数完成像素级权重分配
        rgb_weighted, tir_weighted = mpwa_rgbt_forward(
            R_Res=rgb_map,
            T_Res=tir_map,
            conv_cat=self.mpwa_conv_cat,
            conv_global=self.mpwa_conv_global,
            linear_compress=self.mpwa_linear_compress,
            conv_fusion=self.mpwa_conv_fusion,
            layer_norm=self.mpwa_layer_norm
        )

        # 3. 空间权重（GAP+GMP）—— 逻辑不变
        temp_fused = rgb_weighted + tir_weighted  # (B, dim, 16, 16)
        gap_layer, gmp_layer = self.spatial_weight[0]
        gap_feat = gap_layer(temp_fused)  # (B, dim, 1, 1)
        gmp_feat = gmp_layer(temp_fused)  # (B, dim, 1, 1)
        spatial_weight = self.spatial_weight[1](torch.cat([gap_feat, gmp_feat], dim=1))  # (B,dim,1,1)

        rgb_spatial = rgb_weighted * spatial_weight
        tir_spatial = tir_weighted * spatial_weight

        # 4. 融合+SGFA多尺度聚合——逻辑不变
        init_fused = rgb_spatial + tir_spatial  # (B, dim, 16, 16)
        sgfa_fused = self.sgfa(init_fused)  # (B, dim, 16, 16)
        final_fused_map = sgfa_fused + init_fused  # (B, dim, 16, 16)

        return final_fused_map