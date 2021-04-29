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
_C.AUTOENCODER.EPOCHS = 10
_C.AUTOENCODER.BATCH_SIZE = 2
_C.AUTOENCODER.OUTPUT_FOLDER = 'features/'
_C.AUTOENCODER.LOG_INTERVAL = 10
_C.AUTOENCODER.LATENT_SPACE = 100
_C.AUTOENCODER.LATENT_SPACE_PTC = 512
_C.AUTOENCODER.MAX_MEAN = 0.001
_C.AUTOENCODER.MAX_VAR = 0.001
_C.AUTOENCODER.MAX_GRADIENT = 1.0
_C.AUTOENCODER.FEATURES = ['shape', 'texture']
_C.AUTOENCODER.PTC_NUM_POINTS = 10000
_C.AUTOENCODER.PTC_SAMPLE_MODE = None
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.EM_PATH = ''
_C.DATASET.LABEL_PATH = ''
_C.DATASET.GT_PATH = ''
_C.DATASET.CHUNK_SIZE = [100, 4096, 4096]
_C.DATASET.FILE_FORMAT = 'png'
_C.DATASET.ROOTF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'features/')
_C.DATASET.ROOTD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'datasets/')
_C.DATASET.DATAINFO = 'features/data_info.json'
# -----------------------------------------------------------------------------
# Mode
# := sets different run options.
# -----------------------------------------------------------------------------
_C.MODE = CN()
# Choose between 'infer' and 'train'. Set to train in order to 'train' the autoencoder.
# Set to 'infer' to use the features and cluster.
_C.MODE.PROCESS = 'infer'
# Choose between 'iter' and 'full'. Dataprocessing decides how the data is processed.
# 'iter' iterates over the data slice by slice in order to avoid memory erros.
# 'full' enables to read in the full stack of your data.
_C.MODE.DPRC = 'iter'
# -----------------------------------------------------------------------------
# Clustermodel
# -----------------------------------------------------------------------------
_C.CLUSTER = CN()
_C.CLUSTER.ALG = 'kmeans'
#Choose the features you want to cluster.
_C.CLUSTER.FEAT_LIST = ['sizef', 'distf', 'shapef', 'textf', 'circf']
#Please make sure these weights make the dimension of the features. Means 4 features 4 weight factors.
_C.CLUSTER.WEIGHTSF = [1, 1, 1 ,1, 1]
_C.CLUSTER.N_CLUSTER = 5
_C.CLUSTER.OUTPUTPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs/')
_C.CLUSTER.NEUROGLANCER = False
_C.CLUSTER.GENERATE_MASKS = False
_C.CLUSTER.VISUALIZATION = False

def get_cfg_defaults():
    '''
    Get a yacs CfgNode object with default values define above.
    :returns: clone
    '''
    return _C.clone()
