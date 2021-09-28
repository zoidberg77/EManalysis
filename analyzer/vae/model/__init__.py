from .ptc_vae import PTCvae
from .random_ptc_ae import RandomPtcAe

def get_ptc_model(cfg):
    '''Get the ptc model that is wanted.'''
    if cfg.SSL.MODEL == 'ptc':
        model = PTCvae(num_points=cfg.PTC.RECON_NUM_POINTS, latent_space=cfg.PTC.LATENT_SPACE)
    elif cfg.SSL.MODEL == 'random_ptc':
        model = RandomPtcAe(cfg).double()
    elif cfg.SSL.MODEL == 'ptc++':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
