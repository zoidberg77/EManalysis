from .ptc_vae import PTCvae, PointNet, PNAE
from .random_ptc_ae import RandomPtcAe
from .ptcpp import PTCPP

def get_ptc_model(cfg):
    '''Get the ptc model that is wanted.'''
    if cfg.PTC.ARCHITECTURE == 'ptc':
        model = PTCvae(num_points=cfg.PTC.RECON_NUM_POINTS, latent_space=cfg.PTC.LATENT_SPACE)
    elif cfg.PTC.ARCHITECTURE == 'random_ptc':
        model = RandomPtcAe(cfg).double()
    elif cfg.PTC.ARCHITECTURE == 'ptc++':
        model = PTCPP()
    elif cfg.PTC.ARCHITECTURE == 'pnae':
        model = PNAE()
    else:
        raise NotImplementedError
    return model
