import os

import torch
from tqdm import tqdm
from analyzer.vae.model import unet


class Trainer:
    def __init__(self, dataset, batch_size, train_percentage, model_type, epochs, optimizer_type, loss_function, device="cpu"):

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

        train_length = int(train_percentage* len(dataset))
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
                if not i % 50 and i > 0:
                    train_total_loss = running_total_loss / (len(self.train_dl.dataset))
                    train_reconstruction_loss = running_reconstruction_loss / (len(self.train_dl.dataset))
                    train_kld_loss = running_kld_loss / (len(self.train_dl.dataset))
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
        recons_loss = None
        if self.loss_function == "l1":
            recons_loss = torch.nn.functional.l1_loss(reconstruction, input)
        else:
            recons_loss = torch.nn.functional.mse_loss(reconstruction, input)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kld_loss /= self.train_dl.batch_size*latent_space[-3]*latent_space[-2]*latent_space[-1]
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
