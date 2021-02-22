import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from analyzer.vae.model import unet


class Trainer:
    def __init__(self, dataset, train_percentage, optimizer_type, loss_function, cfg):
        self.cfg = cfg
        self.dataset = dataset
        self.model_type = cfg.AUTOENCODER.ARCHITECTURE
        self.epochs = cfg.AUTOENCODER.EPOCHS
        self.optimizer_type = optimizer_type
        self.device = 'cpu'
        if cfg.SYSTEM.NUM_GPUS > 0:
            self.device = 'cuda'
        self.loss_function = loss_function
        if self.model_type == "unet_3d":
            self.model = unet.UNet3D(input_shape=cfg.AUTOENCODER.TARGET, latent_space=cfg.AUTOENCODER.LATENT_SPACE)
        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

        train_length = int(train_percentage * len(dataset))
        test_length = len(dataset) - train_length
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, test_length))
        self.train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.AUTOENCODER.BATCH_SIZE, shuffle=True)
        self.test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.AUTOENCODER.BATCH_SIZE, shuffle=True)

    def train(self):
        self.model.train()
        self.model.to(self.device)
        running_total_loss = []
        running_reconstruction_loss = []
        running_kld_loss = []
        for epoch in range(1, self.epochs + 1):
            for i, data in enumerate(self.train_dl):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                reconstruction, mu, log_var = self.model(data)
                loss, recon_loss, kld_loss = self.loss(reconstruction, data, mu, log_var)
                running_total_loss.append(loss.item())
                running_reconstruction_loss.append(recon_loss.item())
                running_kld_loss.append(kld_loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if not i % self.cfg.AUTOENCODER.LOG_INTERVAL and i > 0:
                    self.save_images(data, reconstruction, i, epoch, "train")
                    norm = i + i * (epoch - 1)
                    print("[{}/{}] Train reconstruction loss: {}".format(i, int(len(self.train_dl.dataset)/self.train_dl.batch_size), loss/norm))
                    print("[{}/{}] Train kld loss: {}".format(i, int(len(self.train_dl.dataset)/self.train_dl.batch_size), recon_loss/norm))
                    print("[{}/{}] Train total loss: {} \n".format(i, int(len(self.train_dl.dataset)/self.train_dl.batch_size), kld_loss/norm))

            
            train_total_loss = sum(running_total_loss) / len(running_total_loss)
            train_reconstruction_loss = sum(running_reconstruction_loss) / len(running_reconstruction_loss)
            train_kld_loss = sum(running_kld_loss) / len(running_kld_loss)
            self.current_epoch = epoch
            print("Epoch {}: Train reconstruction loss: {}".format(self.current_epoch, train_total_loss))
            print("Epoch {}: Train kld loss: {}".format(self.current_epoch, train_reconstruction_loss))
            print("Epoch {}: Train total loss: {} \n".format(self.current_epoch, train_kld_loss))
            test_loss = self.test()

        plt.axis("on")
        plt.legend(["total loss", "reconstruction loss", "kld loss"])
        plt.plot(running_total_loss)
        plt.plot(running_reconstruction_loss)
        plt.plot(running_kld_loss)
        plt.ylabel("log(train loss)")
        plt.xlabel("# iterations")
        plt.yscale("log")
        plt.title("Train Loss over {} Epochs".format(self.epochs))
        plt.savefig(self.cfg.AUTOENCODER.EVALUATION_IMAGES_OUTPUTDIR + "train_loss.png")
        plt.clf()
        return train_total_loss, test_loss

    def loss(self, reconstruction, input, mu, log_var):
        recons_loss = 0.0
        if self.loss_function == "l1":
            recons_loss = torch.nn.functional.l1_loss(reconstruction, input, reduction="mean")
        else:
            recons_loss = torch.nn.functional.mse_loss(reconstruction, input)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        #kld_loss /= self.train_dl.batch_size
        loss = recons_loss + kld_loss
        return loss, recons_loss, kld_loss

    def test(self):
        self.model.eval()
        self.model.to(self.device)
        running_total_loss = []
        running_reconstruction_loss = []
        running_kld_loss = []
        with torch.no_grad():
            for i, data in enumerate(self.test_dl):
                data = data.to(self.device)
                reconstruction, mu, log_var = self.model(data)
                loss, recon_loss, kld_loss = self.loss(reconstruction, data, mu, log_var)
                running_total_loss.append(loss.item())
                running_reconstruction_loss.append(recon_loss.item())
                running_kld_loss.append(kld_loss.item())
                if not i % self.cfg.AUTOENCODER.LOG_INTERVAL and i > 0:
                    self.save_images(data, reconstruction, i, self.current_epoch, "test")

            test_total_loss = sum(running_total_loss) / len(self.test_dl.dataset)
            test_reconstruction_loss = sum(running_reconstruction_loss) / len(self.test_dl.dataset)
            test_kld_loss = sum(running_kld_loss) / len(self.test_dl.dataset)
            print("Epoch {}: Test reconstruction loss: {}".format(self.current_epoch, test_reconstruction_loss))
            print("Epoch {}: Test kld loss: {}".format(self.current_epoch, test_kld_loss))
            print("Epoch {}: Test total loss: {} \n".format(self.current_epoch, test_total_loss))

            plt.axis("on")
            plt.legend(["total loss", "reconstruction loss", "kld loss"])
            plt.plot(running_total_loss)
            plt.plot(running_reconstruction_loss)
            plt.plot(running_kld_loss)
            plt.ylabel("log(test loss)")
            plt.xlabel("# iterations")
            plt.yscale("log")
            plt.title("Test Loss over {} Epochs".format(self.epochs))
            plt.savefig(self.cfg.AUTOENCODER.EVALUATION_IMAGES_OUTPUTDIR + "test_loss.png")
            plt.clf()
            return test_total_loss

    def save_images(self, inputs, reconstructions, iteration,epoch , prefix):
        for i in range(len(inputs)):
            original_image = None
            reconstruction_image = None
            original_item = inputs[i][0]
            reconstruction_item = reconstructions[i][0]
            for j in range(0, len(original_item), 10):
                if original_image is None:
                    original_image = original_item[j].detach().cpu().numpy()
                    reconstruction_image = reconstruction_item[j].detach().cpu().numpy()
                else:
                    original_image = np.concatenate((original_image, original_item[j].detach().cpu().numpy()), 0)
                    reconstruction_image = np.concatenate(
                        (reconstruction_image, reconstruction_item[j].detach().cpu().numpy()), 0)

            reconstruction_image -= reconstruction_image.min()
            reconstruction_image /= reconstruction_image.max()
            evaluation_image = np.concatenate((original_image, reconstruction_image), 1)

            plt.axis('off')
            plt.imsave(self.cfg.AUTOENCODER.EVALUATION_IMAGES_OUTPUTDIR +'{}_{}_{}.png'.format(epoch, iteration + i, prefix),
                       evaluation_image,
                       cmap="gray")
            return
