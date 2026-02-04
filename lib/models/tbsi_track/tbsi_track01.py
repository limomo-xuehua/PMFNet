"""
TBSI_Track model. Developed on OSTrack.
"""
import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.tbsi_track.vit_tbsi_care import vit_base_patch16_224_tbsi
from lib.utils.box_ops import box_xyxy_to_cxcywh
# 在TBSI_Track代码头部添加导入
from lib.models.tbsi_track.cwa_fusion import CWA_Fusion  # 路径需根据实际项目调整

class TBSITrack(nn.Module):
    """ This is the base class for TBSITrack developed on OSTrack (Ye et al. ECCV 2022) """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        # -------------------------- 核心修改1：CWA_Fusion实例化（匹配方案1的__init__参数） --------------------------
        self.cwa_fusion = CWA_Fusion(
            dim=hidden_dim,  # 与TBSI backbone的特征维度一致（如ViT-Base为768）
            # 方案1的__init__无feat_sz_s参数，删除该参数；dilation_rates用默认值或自定义
            dilation_rates=[1, 6, 12, 18]  # 可根据场景调整，默认适配多尺度
        )
        # -----------------------------------------------------------------------------
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)  # 特征图尺寸（如16），方案1的forward需要该参数
            self.feat_len_s = int(box_head.feat_sz ** 2)  # 256，对应enc_opt1的HW维度

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # 1. 提取RGB/TIR搜索区Token特征（原逻辑不变，维度完全匹配方案1输入）
        # enc_opt1: RGB搜索区Token → (B, 256, C)，匹配方案1的(B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        # enc_opt2: TIR搜索区Token → (B, 256, C)，匹配方案1的(B, HW, C)
        enc_opt2 = cat_feature[:, -num_search_token:, :]
        bs = enc_opt1.shape[0]  # 批次大小，用于后续输出维度调整

        # 2. 核心修改2：调用CWA_Fusion（方案1的forward需传入feat_sz_s，原代码已正确传参）
        # 输入：RGB Token + TIR Token + 搜索区特征图尺寸 → 输出：融合特征图 (B, C, feat_sz_s, feat_sz_s)
        opt_feat = self.cwa_fusion(enc_opt1, enc_opt2, self.feat_sz_s)

        # 3. 跟踪头预测（原逻辑不变，维度已适配）
        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            # 修正：Nq=1（跟踪任务仅需预测1个目标框），避免原代码中Nq未定义的错误
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new, 'score_map': score_map}
            return out
        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)  # 同样修正Nq=1
            out = {'pred_boxes': outputs_coord_new, 'score_map': score_map_ctr, 'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_tbsi_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('TBSITrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_tbsi':
        backbone = vit_base_patch16_224_tbsi(pretrained=pretrained,
                                             cfg=cfg,
                                             drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                             tbsi_loc=cfg.MODEL.BACKBONE.TBSI_LOC,
                                             tbsi_drop_path=cfg.TRAIN.TBSI_DROP_PATH
                                            )
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = TBSITrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'TBSITrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        # 应用冻结
        freeze_non_moe_tbsi_parameters(model)
        print(
            "##########################################Trainable parameters:#########################################")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    return model


def freeze_non_moe_tbsi_parameters(model):
    for name, param in model.named_parameters():
        # 条件：名字中包含这些关键词则保留为可训练
        if any(keyword in name for keyword in [
            "prompt","cwa" # 保留CWA_Fusion的可训练性，原逻辑不变
        ]):
            param.requires_grad = True
        else:
            param.requires_grad = False