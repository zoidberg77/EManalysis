from .random_ptc_ae import RandomPtcAe
from .ptcpp import PTCPP
from .pnae import PNAE, PointNet

def get_ptc_model(cfg):
    '''Get the ptc model that is wanted.'''
    if cfg.PTC.ARCHITECTURE == 'random_ptc':
        model = RandomPtcAe(cfg).double()
    elif cfg.PTC.ARCHITECTURE == 'ptc++':
        model = PTCPP()
    elif cfg.PTC.ARCHITECTURE == 'pnae':
        model = PNAE()
    else:
        raise NotImplementedError
    return model
