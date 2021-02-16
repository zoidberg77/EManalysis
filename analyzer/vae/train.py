import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from analyzer.vae.model import unet


class Trainer:
    def __init__(self, dataset, batch_size, train_percentage, model_type, epochs, optimizer_type, loss_function, cfg,
                 device="cpu"):

        self.cfg = cfg
        self.dataset = dataset
        self.model_type = model_type
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.device = device
        self.loss_function = loss_function
        if self.model_type == "unet_3d":
            self.model = unet.UNet3D()
        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters())

        train_length = int(train_percentage * len(dataset))
        test_length = len(dataset) - train_length
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, test_length))
        self.train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    def train(self):
        self.model.train()
        self.model.to(self.device)
        running_total_loss = 0.0
        running_reconstruction_loss = 0.0
        running_kld_loss = 0.0
        train_total_loss = 0.0
        for epoch in range(1, self.epochs):
            for i, data in tqdm(enumerate(self.train_dl),
                                total=int(len(self.train_dl.dataset) / self.train_dl.batch_size)):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                reconstruction, mu, log_var, latent_space = self.model(data)
                loss, recon_loss, kld_loss = self.loss(reconstruction, data, mu, log_var, latent_space)
                running_total_loss += loss
                running_reconstruction_loss += recon_loss
                running_kld_loss += kld_loss
                loss.backward()
                self.optimizer.step()
                self.show_images(data, reconstruction, i)
                if not i % 10 and i > 0:
                    train_total_loss = running_total_loss / (i * self.train_dl.batch_size)
                    train_reconstruction_loss = running_reconstruction_loss / (i * self.train_dl.batch_size)
                    train_kld_loss = running_kld_loss / (i * self.train_dl.batch_size)
                    print("Train reconstruction loss: {}".format(train_total_loss))
                    print("Train kld loss: {}".format(train_reconstruction_loss))
                    print("Train total loss: {}".format(train_kld_loss))

            train_total_loss = running_total_loss / (len(self.train_dl.dataset))
            train_reconstruction_loss = running_reconstruction_loss / (len(self.train_dl.dataset))
            train_kld_loss = running_kld_loss / (len(self.train_dl.dataset))
            print("Train reconstruction loss: {}".format(train_total_loss))
            print("Train kld loss: {}".format(train_reconstruction_loss))
            print("Train total loss: {}".format(train_kld_loss))
            self.evaluate()
        return train_total_loss

    def loss(self, reconstruction, input, mu, log_var, latent_space):
        if self.loss_function == "l1":
            recons_loss = torch.nn.functional.l1_loss(reconstruction, input)
        else:
            recons_loss = torch.nn.functional.mse_loss(reconstruction, input)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kld_loss /= self.train_dl.batch_size * latent_space
        loss = recons_loss + kld_loss
        return loss, recons_loss, kld_loss

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)
        running_total_loss = 0.0
        running_reconstruction_loss = 0.0
        running_kld_loss = 0.0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.test_dl),
                                total=int(len(self.test_dl.dataset) / self.test_dl.batch_size)):
                data = data.to(self.device)
                reconstruction, mu, log_var, latent_space = self.model(data)
                loss, recon_loss, kld_loss = self.loss(reconstruction, data, mu, log_var, latent_space)

                running_total_loss += loss
                running_reconstruction_loss += recon_loss
                running_kld_loss += kld_loss
            test_total_loss = running_total_loss / (len(self.test_dl.dataset))
            test_reconstruction_loss = running_reconstruction_loss / (len(self.test_dl.dataset))
            test_kld_loss = running_kld_loss / (len(self.test_dl.dataset))
            print("Evaluation reconstruction loss: {}".format(test_reconstruction_loss))
            print("Evaluation kld loss: {}".format(test_kld_loss))
            print("Evaluation total loss: {}".format(test_total_loss))
            return test_total_loss

    def show_images(self, inputs, reconstructions, iteration):

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

            evaluation_image = np.concatenate((original_image, reconstruction_image), 1)
            plt.axis('off')
            plt.imsave(self.cfg.AUTOENCODER.EVALUATION_IMAGES_OUTPUTDIR + '{}.png'.format(iteration+i), evaluation_image,
                       cmap="gray")
            return
