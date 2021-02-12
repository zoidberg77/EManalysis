import os, sys
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 0
_C.SYSTEM.NUM_CPUS = 4
_C.SYSTEM.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------------------------------------------------------
# Autoencoder
# -----------------------------------------------------------------------------
_C.AUTOENCODER = CN()
_C.AUTOENCODER.ARCHITECTURE = 'vae_3d'
_C.AUTOENCODER.REGION_LIMIT = None
_C.AUTOENCODER.CHUNKS_CPU = 4
_C.AUTOENCODER.UPPER_BOUND = 0
_C.AUTOENCODER.LOWER_BOUND = 100000000
_C.AUTOENCODER.TARGET = [64, 64, 64]
_C.AUTOENCODER.OUPUT_FILE_VOLUMES = 'features/mito.h5'
_C.AUTOENCODER.DATASET_NAME = 'mito_volumes'
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.EM_PATH = ''
_C.DATASET.LABEL_PATH = ''
_C.DATASET.FILE_FORMAT = 'png'
_C.DATASET.DATAINFO = 'features/data_info.json'
_C.DATASET.SIZEF = ''

# -----------------------------------------------------------------------------
# Feature Extration
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Clustermodel
# -----------------------------------------------------------------------------


def get_cfg_defaults():
    '''
    Get a yacs CfgNode object with default values define above.
    :returns: clone
    '''
    return _C.clone()
