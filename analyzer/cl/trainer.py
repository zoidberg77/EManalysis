import os, sys
import h5py
import torch
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from analyzer.data.augmentation.augmentor import Augmentor
from analyzer.cl.model import get_model
from analyzer.data import PairDataset
from analyzer.cl.engine.loss import similarity_func
from analyzer.cl.engine.optimizer import build_optimizer, build_lr_scheduler
from analyzer.utils.vis.monitor import build_monitor
from analyzer.cl.engine.classifier import knn_classifier

class CLTrainer():
    '''
    Trainer object, enabling constrastive learning framework.
    :params cfg: (yacs.config.CfgNode): YACS configuration options.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.epochs = self.cfg.SSL.EPOCHS
        self.device = 'cpu'
        if self.cfg.SYSTEM.NUM_GPUS > 0 and torch.cuda.is_available():
            self.device = 'cuda'
        self.model = get_model(self.cfg).to(self.device)

        # Setting up the dataset.
        self.dataset = PairDataset(self.cfg)
        train_length = int(self.cfg.SSL.TRAIN_PORTION * len(self.dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, (train_length, len(self.dataset) - train_length))
        self.train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.SSL.BATCH_SIZE, shuffle=False)
        self.test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.SSL.BATCH_SIZE, shuffle=False)

        # Setting up the optimizer, lr & logger.
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer, len(self.train_dl))

        # Setting the outputpath for each run.
        if self.cfg.MODE.PROCESS == 'cltrain' or self.cfg.MODE.PROCESS == 'cltest':
            time_now = str(datetime.datetime.now()).split(' ')
            if os.path.exists(os.path.join(self.cfg.SSL.MONITOR_PATH, 'run_' + time_now[0])):
                self.output_path = os.path.join(self.cfg.SSL.MONITOR_PATH, 'run_' + time_now[0])
            else:
                self.output_path = os.path.join(self.cfg.SSL.MONITOR_PATH, 'run_' + time_now[0])
                os.makedirs(self.output_path)
        elif self.cfg.MODE.PROCESS == 'clinfer':
            self.state_model = self.cfg.SSL.STATE_MODEL
            self.output_path = self.cfg.SSL.STATE_MODEL.rsplit('/', 1)[0]
        else:
            raise ValueError('No valid process. Choose \'cltrain\' or \'clinfer\'.')

    def train(self):
        self.model.train()
        counter = 0
        running_loss = list()
        self.logger = build_monitor(self.cfg, self.output_path, 'train')
        if self.cfg.SSL.VALIDATION == True:
            self.validate_logger = build_monitor(self.cfg, self.output_path, 'test')

        for epoch in range(0, self.epochs):
            for idx, ((x1, x2), _, _) in enumerate(self.train_dl):
                self.model.zero_grad()
                z1, p1, z2, p2 = self.model.forward(x1.to(self.device, non_blocking=True), x2.to(self.device, non_blocking=True))
                loss = similarity_func(p1, z2) / 2 + similarity_func(p2, z1) / 2
                loss = loss.mean()
                loss.backward()
                running_loss.append(loss.item())
                self.optimizer.step()
                self.lr_scheduler.step()

                if not idx % self.cfg.SSL.LOG_INTERVAL:
                    self.logger.update((sum(running_loss) / len(running_loss)), counter, self.lr_scheduler.get_lr(), epoch)
                counter = counter + 1

            self.save_checkpoint(epoch)
            if self.cfg.SSL.VALIDATION == True:
                self.validate(self.validate_logger)

    def test(self):
        if self.cfg.SSL.STATE_MODEL:
            print('cl model {} loaded and used for testing.'.format(self.cfg.SSL.STATE_MODEL))
            self.model.load_state_dict(torch.load(self.cfg.SSL.STATE_MODEL))
        self.logger = build_monitor(self.cfg, self.output_path, 'test')

        acc = knn_classifier(self.model.encoder, self.train_dl, self.test_dl, self.device, k_knn=self.cfg.SSL.K_KNN)
        self.logger.update(0, 0, 0, 0, acc=acc)

    def validate(self, logger=None):
        self.dataset.cl_mode = 'test'
        acc = knn_classifier(self.model.encoder, self.train_dl, self.test_dl, self.device, k_knn=self.cfg.SSL.K_KNN)
        logger.update(0, 0, 0, 0, acc=acc)
        self.dataset.cl_mode = 'train'

    def infer_feat_vector(self):
        '''infers the feature vector of every sample.'''
        feat_data_loader = torch.utils.data.DataLoader(self.dataset)
        print('\n-------------------------')
        print('length: ', len(feat_data_loader))

        if self.cfg.SSL.STATE_MODEL:
            print('cl model {} loaded and used for testing.'.format(self.cfg.SSL.STATE_MODEL))
            self.model.load_state_dict(torch.load(self.cfg.SSL.STATE_MODEL))
            self.model.eval()
        else:
            raise ValueError('Please adjust the SSL.STATE_MODEL in config for infering.')

        with h5py.File(os.path.join(self.cfg.SSL.OUTPUT_FOLDER, self.cfg.SSL.FEATURE_NAME), 'w') as h5f:
            h5f.create_dataset(name='cl', shape=(len(self.dataset), self.cfg.SSL.LATENT_SPACE))
            h5f.create_dataset(name='id', shape=(len(self.dataset),))

            with torch.no_grad():
                for idx, (sample, ids, gt_labels) in enumerate(feat_data_loader):
                    features = self.model.infer(sample.to(self.device, non_blocking=True))
                    #features = F.normalize(features, dim=1)

                    x = features.cpu().numpy()
                    h5f['cl'][idx] = x
                    h5f['id'][idx] = int(ids[0])
            h5f.close()

    def classify(self):
        '''classifing all single segments by kNN.'''
        pass

    def save_checkpoint(self, idx: int):
        '''Save the model at certain checkpoints.'''
        # state = {'iteration': idx + 1,
        # 		 'state_dict': self.model.state_dict(),
        # 		 'optimizer': self.optimizer.state_dict(),
        # 		 'lr_scheduler': self.lr_scheduler.state_dict()}

        state = self.model.state_dict()
        filename = 'cl_model_{}.pt'.format(idx)
        filename = os.path.join(self.output_path, filename)
        torch.save(state, filename)
