from easydict import EasyDict as edict
import yaml

"""
Add default config for TBSITrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
# 优化修改：使用SOT预训练权重（原MAE预训练更偏向图像分类，SOT预训练更适配跟踪任务）
cfg.MODEL.PRETRAIN_FILE = "/home/hanpenghui/fx/TBSI-08/pretrained_models/TBSITrack_SOT.pth.tar"
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
# 优化修改：调整返回阶段，与Prompt插入层（2,5,8）匹配，避免深层冗余
cfg.MODEL.RETURN_STAGES = [2, 5, 8]

# 添加 PROMPT 配置块（同步YAML优化：轻量化+Dropout正则化）
# ========== 新增开始 ==========
cfg.MODEL.PROMPT_RGB = edict()
cfg.MODEL.PROMPT_RGB.ENABLED = True  # 是否启用 Prompt 模块
# 优化修改：插入ViT第2、5、8层（浅层+中层，不破坏深层语义特征）
cfg.MODEL.PROMPT_RGB.LOCATIONS = [2, 5, 8]   # Prompt 模块插入的层索引（从0开始）
cfg.MODEL.PROMPT_RGB.INPLANES = 768   # 输入特征通道数（与ViT-Base的embed_dim一致）
# 优化修改：通道压缩至128（原256参数冗余，128平衡轻量化与特征保留）
cfg.MODEL.PROMPT_RGB.HIDE_CHANNEL = 128  # Prompt 模块隐藏层通道数
# 优化修改：Dropout率提升至0.15（原0.1正则化不足，0.15更适配跨模态增强）
cfg.MODEL.PROMPT_RGB.DROP_RATE = 0.15  # Prompt 模块中的 Dropout 率


# ========== 新增结束 ==========

# MODEL.BACKBONE（同步YAML的TBSI配置）
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224_tbsi"  # 优化修改：启用TBSI专用ViT骨干
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# RGBT.BACKBONE（优化TBSI模块配置）
# 优化修改：TBSI模块插入第3、6、9层（与Prompt层错开，形成"增强→交互"递进）
cfg.MODEL.BACKBONE.TBSI_LOC = [3, 6, 9]
cfg.MODEL.BACKBONE.RGB_ONLY = False
cfg.MODEL.BACKBONE.RGBT_UNSHARE = False  # 共享骨干参数，减少冗余

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN（同步YAML抗过拟合配置）
cfg.TRAIN = edict()
# 1. 基础训练参数
cfg.TRAIN.LR = 0.0003
cfg.TRAIN.WEIGHT_DECAY = 0.0003
cfg.TRAIN.EPOCH = 20
cfg.TRAIN.LR_DROP_EPOCH = 10
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 3
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

# 2. 显式初始化早停相关键（关键：解决“not exist”错误）
cfg.TRAIN.EARLY_STOP_PATIENCE = 2  # 与YAML中的键名完全一致
cfg.TRAIN.EARLY_STOP_METRIC = "Success"  # 与YAML中的键名完全一致

# 3. 其他参数
cfg.TRAIN.CE_START_EPOCH = 20
cfg.TRAIN.CE_WARM_EPOCH = 80
cfg.TRAIN.DROP_PATH_RATE = 0.25
cfg.TRAIN.TBSI_DROP_RATE = 0.
cfg.TRAIN.TBSI_DROP_PATH = [0.15, 0.15, 0.15]
cfg.TRAIN.SOT_PRETRAIN = True

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1  # 与YAML一致，学习率衰减10倍

# DATA（同步YAML的双模态归一化与增强配置）
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
# 优化修改：补充TIR模态的均值（后3个0.449），适配RGB-T双模态归一化
cfg.DATA.MEAN = [0.485, 0.456, 0.406, 0.449, 0.449, 0.449]
# 优化修改：补充TIR模态的标准差（后3个0.226），双模态归一化更准确
cfg.DATA.STD = [0.229, 0.224, 0.225, 0.226, 0.226, 0.226]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200

# DATA.TRAIN（同步YAML的训练数据集配置）
cfg.DATA.TRAIN = edict()
# 优化修改：数据集改为LasHeR_train（与YAML一致，RGB-T专用数据集）
cfg.DATA.TRAIN.DATASETS_NAME = ["LasHeR_train"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000  # 与YAML一致，保证训练样本量

# DATA.VAL（同步YAML的验证数据集配置）
cfg.DATA.VAL = edict()
# 优化修改：验证集改为LasHeR_test（与YAML一致，独立测试集）
cfg.DATA.VAL.DATASETS_NAME = ["LasHeR_test"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000  # 与YAML一致，覆盖足够场景

# DATA.SEARCH（同步YAML的搜索区增强配置）
cfg.DATA.SEARCH = edict()
# 优化修改：搜索区尺寸从320降至256（与YAML一致）
cfg.DATA.SEARCH.SIZE = 256
# 优化修改：缩放因子从5.0降至4.0（与YAML一致）
cfg.DATA.SEARCH.FACTOR = 4.0
# 优化修改：中心抖动从4.5降至3（与YAML一致）
cfg.DATA.SEARCH.CENTER_JITTER = 3
# 优化修改：尺度抖动从0.5降至0.25（与YAML一致）
cfg.DATA.SEARCH.SCALE_JITTER = 0.25
cfg.DATA.SEARCH.NUMBER = 1
# 优化修改：新增水平翻转（与YAML一致）
cfg.DATA.SEARCH.HORIZONTAL_FLIP = True
# 优化修改：新增亮度抖动（与YAML一致）
cfg.DATA.SEARCH.BRIGHTNESS_JITTER = 0.1

# DATA.TEMPLATE（同步YAML的模板增强配置）
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128  # 与YAML一致
cfg.DATA.TEMPLATE.FACTOR = 2.0  # 与YAML一致
cfg.DATA.TEMPLATE.CENTER_JITTER = 0  # 与YAML一致
cfg.DATA.TEMPLATE.SCALE_JITTER = 0  # 与YAML一致
# 优化修改：新增水平翻转（与YAML一致）
cfg.DATA.TEMPLATE.HORIZONTAL_FLIP = True
# 优化修改：新增亮度抖动（与YAML一致）
cfg.DATA.TEMPLATE.BRIGHTNESS_JITTER = 0.05

# TEST（同步YAML的测试配置）
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
# 优化修改：搜索区因子从5.0降至4.0（与YAML一致）
cfg.TEST.SEARCH_FACTOR = 4.0
# 优化修改：搜索区尺寸从320降至256（与YAML一致）
cfg.TEST.SEARCH_SIZE = 256
# 优化修改：测试epoch从500降至20（与训练epoch一致）
cfg.TEST.EPOCH = 20


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    """
    核心修改：允许从exp_cfg新增键到base_cfg，不再抛出"键不存在"错误
    逻辑：
    1. 若k在base_cfg中：正常更新（覆盖值或递归更新子字典）
    2. 若k不在base_cfg中：
       - 若exp_cfg的v是edict：在base_cfg中创建空edict后递归更新
       - 若v是普通值：直接赋值到base_cfg[k]
    """
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            # 关键修改：键不存在时，先在base_cfg中创建对应键
            if k not in base_cfg:
                # 若v是子配置（edict），则创建空edict；否则直接赋值初始值
                base_cfg[k] = edict() if isinstance(v, edict) else v
            # 递归更新（无论键是否新增，都需同步exp_cfg的数值）
            if not isinstance(v, dict):
                base_cfg[k] = v
            else:
                _update_config(base_cfg[k], v)
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        # 加载YAML为edict格式（与cfg结构一致）
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            # 无指定base_cfg时，更新全局cfg
            _update_config(cfg, exp_config)