import math

import torch
import torch.nn.functional as F


def combine_tokens(template_tokens, search_tokens, mode='direct', return_res=False):
    # [B, HW, C]
    len_t = template_tokens.shape[1]
    len_s = search_tokens.shape[1]

    if mode == 'direct':
        merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
    elif mode == 'template_central':
        central_pivot = len_s // 2
        first_half = search_tokens[:, :central_pivot, :]
        second_half = search_tokens[:, central_pivot:, :]
        merged_feature = torch.cat((first_half, template_tokens, second_half), dim=1)
    elif mode == 'partition':
        feat_size_s = int(math.sqrt(len_s))
        feat_size_t = int(math.sqrt(len_t))
        window_size = math.ceil(feat_size_t / 2.)
        # pad feature maps to multiples of window size
        B, _, C = template_tokens.shape
        H = W = feat_size_t
        template_tokens = template_tokens.view(B, H, W, C)
        pad_l = pad_b = pad_r = 0
        # pad_r = (window_size - W % window_size) % window_size
        pad_t = (window_size - H % window_size) % window_size
        template_tokens = F.pad(template_tokens, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, Hp // window_size, window_size, W, C)
        template_tokens = torch.cat([template_tokens[:, 0, ...], template_tokens[:, 1, ...]], dim=2)
        _, Hc, Wc, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, -1, C)
        merged_feature = torch.cat([template_tokens, search_tokens], dim=1)

        # calculate new h and w, which may be useful for SwinT or others
        merged_h, merged_w = feat_size_s + Hc, feat_size_s
        if return_res:
            return merged_feature, merged_h, merged_w

    else:
        raise NotImplementedError

    return merged_feature


def recover_tokens(merged_tokens, len_template_token, len_search_token, mode='direct'):
    if mode == 'direct':
        recovered_tokens = merged_tokens
    elif mode == 'template_central':
        central_pivot = len_search_token // 2
        len_remain = len_search_token - central_pivot
        len_half_and_t = central_pivot + len_template_token

        first_half = merged_tokens[:, :central_pivot, :]
        second_half = merged_tokens[:, -len_remain:, :]
        template_tokens = merged_tokens[:, central_pivot:len_half_and_t, :]

        recovered_tokens = torch.cat((template_tokens, first_half, second_half), dim=1)
    elif mode == 'partition':
        recovered_tokens = merged_tokens
    else:
        raise NotImplementedError

    return recovered_tokens


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
#为提示词模块新加入的函数

def token2feature(tokens): # 将tokens转换为feature
    B, L, D = tokens.shape
    H = W = int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()  # (B,D,L) --> (B,C,H,W)
    return x


'''
feature2token
'''
def feature2token(x):  # 将feartue转换为tokens
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens


def separate_features(merged_features, template_length=64):
    """
    从融合特征中分离模板和搜索区域特征

    参数:
    merged_features: 融合特征张量，形状为 [B, N, C]
    template_length: 模板区域token数量，默认为64

    返回:
    template_features: 模板特征，形状为 [B, template_length, C]
    search_features: 搜索区域特征，形状为 [B, N-template_length, C]
    """
    # 验证输入形状
    if merged_features.dim() != 3:
        raise ValueError(f"输入张量应为3D，当前维度: {merged_features.dim()}")

    batch_size, total_length, feature_dim = merged_features.shape

    # 计算搜索区域长度
    search_length = total_length - template_length

    # 验证总长度匹配
    if total_length != template_length + search_length:
        raise ValueError(f"总长度({total_length})与模板+搜索长度({template_length + search_length})不匹配")

    # 分离特征
    template_features = merged_features[:, :template_length, :]
    search_features = merged_features[:, template_length:, :]

    return template_features,search_features


def combine_features(template_features, search_features):
    """
    将模板特征和搜索区域特征合并为融合特征（separate_features的反向操作）

    参数:
    template_features: 模板特征张量，形状为 [B, L1, C]
    search_features: 搜索区域特征张量，形状为 [B, L2, C]

    返回:
    merged_features: 融合特征张量，形状为 [B, L1+L2, C]
    """
    # 验证输入形状
    if template_features.dim() != 3 or search_features.dim() != 3:
        raise ValueError("输入张量必须为3D")

    batch_size_t, template_length, feature_dim_t = template_features.shape
    batch_size_s, search_length, feature_dim_s = search_features.shape

    # 验证批量大小和特征维度一致
    if batch_size_t != batch_size_s:
        raise ValueError(f"批量大小不匹配: {batch_size_t} vs {batch_size_s}")

    if feature_dim_t != feature_dim_s:
        raise ValueError(f"特征维度不匹配: {feature_dim_t} vs {feature_dim_s}")

    # 在序列长度维度上合并特征
    merged_features = torch.cat(
        [template_features, search_features],
        dim=1
    )

    return merged_features
#添加结束