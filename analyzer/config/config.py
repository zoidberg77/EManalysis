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
_C.AUTOENCODER.MAX_MEAN = 0.001
_C.AUTOENCODER.MAX_VAR = 0.001
_C.AUTOENCODER.MAX_GRADIENT = 1.0
_C.AUTOENCODER.FEATURES = ['shape', 'texture']
_C.AUTOENCODER.LARGE_OBJECT_SAMPLES = 4
_C.AUTOENCODER.MONITOR_PATH = 'models/vae/'
_C.AUTOENCODER.MODEL = '' # models/vae/human/run_2021-08-29/vae_ptc_model_10.pt
# -----------------------------------------------------------------------------
# PointCloud based Learning
# -----------------------------------------------------------------------------
_C.PTC = CN()
_C.PTC.ARCHITECTURE = 'ptc_ae'
_C.PTC.BATCH_SIZE = 1
_C.PTC.EPOCHS = 10
_C.PTC.LR = 0.0001
_C.PTC.WEIGHT_DECAY = 0.0001
_C.PTC.LATENT_SPACE = 512
_C.PTC.FILTER_LIST = [64, 64, 64, 128, 512]
_C.PTC.LINEAR_LAYERS = [1024, 1024]
_C.PTC.SAMPLE_MODE = None
_C.PTC.RECON_NUM_POINTS = 5000
_C.PTC.SAMPLE_SIZE = 4096
_C.PTC.BLUE_NOISE_SAMPLE_POINTS = 10
_C.PTC.LOG_INTERVAL = 10
_C.PTC.DEVICE = 'cpu'
_C.PTC.INPUT_DATA = ''
_C.PTC.INPUT_DATA_SAMPLED = '' # wn_pts.h5, mc_pts.h5, bn_pts.h5
_C.PTC.FEATURE_NAME = 'ptc_shapef'
_C.PTC.OUTPUT_FOLDER = 'features/'
_C.PTC.MONITOR_PATH = 'models/ptc/'
_C.PTC.RECONSTRUCTION_DATA = 'rec_pts.h5'
_C.PTC.MODEL = '' # models/ptc/human/run_2021-08-29/vae_ptc_model_10.pt
# -----------------------------------------------------------------------------
# Self-Supervised Learning
# -----------------------------------------------------------------------------
_C.SSL = CN()
_C.SSL.MODEL = 'siamnet'
_C.SSL.ENCODER = 'resnet3d'
_C.SSL.BATCH_SIZE = 256
_C.SSL.EPOCHS = 25
_C.SSL.ITERATION_SAVE = 5000
_C.SSL.OPTIMIZER = 'sgd'
_C.SSL.OPTIMIZER_LR = 0.05
_C.SSL.OPTIMIZER_WEIGHT_DECAY = 0.0005
_C.SSL.OPTIMIZER_MOMENTUM = 0.9
_C.SSL.K_KNN = 5
_C.SSL.TRAIN_PORTION = 0.7
_C.SSL.USE_PREP_DATASET = ''
_C.SSL.FEATURE_NAME = 'cl_shapef'
_C.SSL.OUTPUT_FOLDER = 'features/'
_C.SSL.MONITOR_PATH = 'models/cl/'
_C.SSL.LOG_INTERVAL = 10
_C.SSL.STATE_MODEL = '' # models/ptc/human/run_2021-08-29/cl_model_10.pt
_C.SSL.VALIDATION = False
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
_C.DATASET.CHUNKS_PATH = ''
_C.DATASET.EXCLUDE_BORDER_OBJECTS = True
# -----------------------------------------------------------------------------
# Mode
# := sets different run options.
# -----------------------------------------------------------------------------
_C.MODE = CN()
# Choose between 'infer' and 'train'. Set to train in order to 'train' the autoencoder.
# Set to 'infer' to use the features and cluster.
_C.MODE.PROCESS = ''
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
_C.CLUSTER.FEAT_LIST = ['sizef', 'distf', 'shapef', 'texturef', 'circf', 'ptc_shapef']
#Please make sure these weights make the dimension of the features. Means 4 features 4 weight factors.
_C.CLUSTER.WEIGHTSF = [1, 1, 1 ,1, 1, 1]
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
