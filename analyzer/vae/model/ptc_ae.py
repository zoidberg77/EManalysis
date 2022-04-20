import numpy as np
import h5py
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

def ChamferDistance(x, y):  # for example, x = batch,2025,3 y = batch,2048,3
    #   compute chamfer distance between tow point clouds x and y

    x_size = x.size()
    y_size = y.size()
    assert (x_size[0] == y_size[0])
    assert (x_size[2] == y_size[2])
    x = torch.unsqueeze(x, 1)  # x = batch,1,2025,3
    y = torch.unsqueeze(y, 2)  # y = batch,2048,1,3

    x = x.repeat(1, y_size[1], 1, 1)  # x = batch,2048,2025,3
    y = y.repeat(1, 1, x_size[1], 1)  # y = batch,2048,2025,3

    x_y = x - y
    x_y = torch.pow(x_y, 2)  # x_y = batch,2048,2025,3
    x_y = torch.sum(x_y, 3, keepdim=True)  # x_y = batch,2048,2025,1
    x_y = torch.squeeze(x_y, 3)  # x_y = batch,2048,2025
    x_y_row, _ = torch.min(x_y, 1, keepdim=True)  # x_y_row = batch,1,2025
    x_y_col, _ = torch.min(x_y, 2, keepdim=True)  # x_y_col = batch,2048,1

    x_y_row = torch.mean(x_y_row, 2, keepdim=True)  # x_y_row = batch,1,1
    x_y_col = torch.mean(x_y_col, 1, keepdim=True)  # batch,1,1
    x_y_row_col = torch.cat((x_y_row, x_y_col), 2)  # batch,1,2
    chamfer_distance, _ = torch.max(x_y_row_col, 2, keepdim=True)  # batch,1,1
    # chamfer_distance = torch.reshape(chamfer_distance,(x_size[0],-1))  #batch,1
    # chamfer_distance = torch.squeeze(chamfer_distance,1)    # batch
    chamfer_distance = torch.mean(chamfer_distance)
    return chamfer_distance

class ChamferLoss(pl.LightningModule):
    # chamfer distance loss
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, x, y):
        return ChamferDistance(x, y)

class STN3d(pl.LightningModule):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if batchsize == 1:
            self.eval()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1).cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetfeat(pl.LightningModule):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # x = batch,1024,n(n=2048)
        x = torch.max(x, 2, keepdim=True)[0]  # x = batch,1024,1
        x = x.view(-1, 1024)  # x = batch,1024
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans

class FoldingNetEnc(pl.LightningModule):
    def __init__(self, k):
        super(FoldingNetEnc, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        if batchsize == 1:
            self.eval()
        x, trans = self.feat(x)  # x = batch,1024
        x = F.relu(self.bn1(self.fc1(x)))  # x = batch,256
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)


        return x, trans

class FoldingNetDecFold1(nn.Module):
    def __init__(self):
        super(FoldingNetDecFold1, self).__init__()
        self.conv1 = nn.Conv1d(514, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

        self.relu = nn.ReLU()

    def forward(self, x):  # input x = batch,514,45^2
        x = self.relu(self.conv1(x))  # x = batch,512,45^2
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x

class FoldingNetDecFold2(pl.LightningModule):
    def __init__(self):
        super(FoldingNetDecFold2, self).__init__()
        self.conv1 = nn.Conv1d(515, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):  # input x = batch,515,45^2
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class FoldingNetDec(pl.LightningModule):
    def __init__(self, k):
        super(FoldingNetDec, self).__init__()
        self.fc1 = nn.Linear(k, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fold1 = FoldingNetDecFold1()
        self.fold2 = FoldingNetDecFold2()

    def forward(self, x):  # input x = batch, 512
        x = self.fc1(x)
        x = self.fc2(x)
        batch_size = x.size(0)
        x = torch.unsqueeze(x, 1)  # x = batch,1,512
        x = x.repeat(1, 45 ** 2, 1)  # x = batch,45^2,512
        code = x
        code = x.transpose(2, 1)  # x = batch,512,45^2

        meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        grid = GridSamplingLayer(batch_size, meshgrid)  # grid = batch,45^2,2
        grid = torch.from_numpy(grid)

        grid = grid.cuda()

        x = torch.cat((x, grid), 2)  # x = batch,45^2,514
        x = x.transpose(2, 1)  # x = batch,514,45^2

        x = self.fold1(x)  # x = batch,3,45^2
        p1 = x  # to observe

        x = torch.cat((code, x), 1)  # x = batch,515,45^2

        x = self.fold2(x)  # x = batch,3,45^2

        return x, p1

def GridSamplingLayer(batch_size, meshgrid):
    '''
    output Grid points as a NxD matrix
    params = {
    'batch_size': 8
    'meshgrid': [[-0.3,0.3,45],[-0.3,0.3,45]]
    }
    '''

    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)  # MxD
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)

    return g

class FoldingNet(pl.LightningModule):

    def __init__(self,
                 cfg):
        super(FoldingNet, self).__init__()
        k = cfg.PTC.LATENT_SPACE
        self.cfg = cfg
        self.encoder = FoldingNetEnc(k)
        self.decoder = FoldingNetDec(k)

    def forward(self, x):
        code, tran = self.encoder(x)
        x, x_middle = self.decoder(code)
        return x, x_middle,code

    def step(self, x):
        x = x.transpose(2,1)
        recon_pc, mid_pc, code = self.forward(x)
        loss = self.loss(x.transpose(2,1), recon_pc.transpose(2,1), mid_pc.transpose(2,1))
        return loss, {"loss": loss}, recon_pc, code

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, logs, reconstruction, _ = self.step(x)
        #self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, logs, reconstruction, _ = self.step(x)
        #self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


    def loss(self, points, recon_pc, mid_pc):
        chamferloss = ChamferLoss()
        loss = chamferloss(points, recon_pc)
        mid_loss = chamferloss(points, mid_pc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = int(y[0])
        loss, logs, reconstruction, code = self.step(x)
        
        with h5py.File(self.cfg.DATASET.ROOTF + "shapef.h5", "a") as featuref:
            featuref["id"][batch_idx] = y
            featuref["shape"][batch_idx] = code.cpu()
            featuref["output"][batch_idx] = reconstruction.transpose(2,1).cpu()[0]
        return loss

class PtcAeDataModule(pl.LightningDataModule):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.cpus = cfg.SYSTEM.NUM_CPUS
        self.batch_size = cfg.PTC.BATCH_SIZE
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float())
            # transforms.Lambda(self.helper_pickle)
        ])
        self.dataset = dataset

    def setup(self, stage=None):
        train_length = int(0.9 * len(self.dataset))
        test_length = len(self.dataset) - train_length
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, (train_length, test_length))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.cpus, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.cpus, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=1, shuffle=False)