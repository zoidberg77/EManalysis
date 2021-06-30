import torch
from yacs.config import CfgNode
from analyzer.cl.engine.lr_scheduler import LRScheduler

def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    '''Build custom optimizer.'''
    assert cfg.SSL.OPTIMIZER in ['sgd']
    if cfg.SSL.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SSL.OPTIMIZER_LR,
                                    momentum=cfg.SSL.OPTIMIZER_MOMENTUM,
                                    weight_decay=cfg.SSL.OPTIMIZER_WEIGHT_DECAY)
    return optimizer

def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    '''Build a LR scheduler.'''
    return LRScheduler(optimizer, cfg.SSL.OPTIMIZER_LR, cfg.SSL.EPOCHS, 1000)
