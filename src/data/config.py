import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.PRETRAINED = ''
_C.MODEL.SSL_PRETRAINED = ''  # New key based on your YAML
_C.MODEL.CHECKPOINT_DIR = ''  # New key based on your YAML

# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.TRAIN_CSV_PATH = ''
_C.DATA.CACHE_DIR = None
_C.DATA.DATA_PERCENTAGE = 1.0
_C.DATA.TRAINING_PERCENTAGE = 0.9
_C.DATA.GROUPING = 'global'
_C.DATA.PIN_MEMORY = False
_C.DATA.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.SAVECKP_FREQ = 1
_C.TRAIN.VAL_EVERY = 2
_C.TRAIN.BASE_LR = 1.5e-4
_C.TRAIN.MIN_LR = 1.5e-8
_C.TRAIN.WEIGHT_DECAY = 0.05

_C.TRAIN.BETA1 = 0.9
_C.TRAIN.BETA2 = 0.95

_C.TRAIN.MOMENTUM = 0.9

_C.TRAIN.TEMPERATURE = 0.5
_C.TRAIN.PER_WARMUP = 0.01
_C.TRAIN.WARMUP_EPOCHS = 1
_C.TRAIN.GRAD_CLIP = 3.0
_C.TRAIN.OPTIMIZER = 'AdamW'
_C.TRAIN.SCHEDULER = 'cosine'

# -----------------------------------------------------------------------------
# Vision Transformer (ViT) Settings
# -----------------------------------------------------------------------------
_C.VIT = CN()
_C.VIT.INPUT_SIZE = 128
_C.VIT.PATCH_SIZE = 16
_C.VIT.IN_CHANS = 1
_C.VIT.DROPOUT_RATE = 0.
_C.VIT.PATCH_EMBED = 'conv'
_C.VIT.POS_EMBED = 'sincos'
_C.VIT.SPATIAL_DIMS = 3
_C.VIT.HIDDEN_SIZE = 768
_C.VIT.NUM_LAYERS = 12
_C.VIT.MLP_DIM = 3072
_C.VIT.NUM_HEADS = 12
_C.VIT.USE_BIAS = False
_C.VIT.USE_FLASH_ATTN = False
_C.VIT.SMOOTH_MCI_LABEL = True  # New key based on your YAML

# -----------------------------------------------------------------------------
# wandb Settings
# -----------------------------------------------------------------------------
_C.WANDB = CN()
_C.WANDB.WANDB_ENABLE = True
_C.WANDB.PROJECT = 'SEGMENTATION'
_C.WANDB.RUN = 'seg_train'  # New key based on your YAML

# -----------------------------------------------------------------------------
# Logging Settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.FILENAME = 'seg_train'


# -----------------------------------------------------------------------------
# Misc Settings
# -----------------------------------------------------------------------------
_C.SEED = 42
_C.OUTPUT = './logs'

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # BASE handling skipped since no BASE in YAML example

    print('=> merge config from {}'.format(cfg_file))
    config.set_new_allowed(True)
    config.merge_from_file(cfg_file)
    config.freeze()


def get_config(cfg_file):
    """Get a yacs CfgNode object from a YAML config file."""
    config = _C.clone()
    _update_config_from_file(config, cfg_file)
    return config


