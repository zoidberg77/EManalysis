import torch
from .siamnet import SiameseNet
from .resnet import ResNet3D

def get_encoder(cfg):
    '''Choosing the encoder model for the self-supervised model.'''
    if cfg.SSL.ENCODER == 'resnet3d':
        encoder = ResNet3D()
    encoder.d_output = encoder.avgpool
    #encoder.fc = torch.nn.Identity()
    return encoder

def get_model(cfg):
    '''Get the self-supervised model that is wanted.'''

    if cfg.SSL.MODEL == 'siamnet':
        model =  SiameseNet(get_encoder(cfg))
    elif cfg.SSL.MODEL == 'byol':
        raise NotImplementedError
    elif cfg.SSL.MODEL == 'simclr':
        raise NotImplementedError
    elif cfg.SSL.MODEL == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
