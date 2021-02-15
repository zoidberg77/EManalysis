import time

import torch
from tqdm import tqdm
from analyzer.vae.model import unet


class Trainer:
    def __init__(self, dataloader, model_type, epochs, optimizer_type, loss_function, device="cpu"):

        self.dataloader = dataloader
        self.model_type = model_type
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.device = device
        self.loss_function = loss_function
        if self.model_type == "unet":
            self.model = unet.UNet3D()
        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters())

    def fit(self):
        self.model.train()
        running_loss = 0.0
        for epoch in range(1, self.epochs):
            for i, data in tqdm(enumerate(self.dataloader),
                                total=int(len(self.dataloader.dataset) / self.dataloader.batch_size)):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                reconstruction, mu, log_var, latent_space = self.model(data)
                loss, recon_loss, kld_loss = self.loss(reconstruction, data, mu, log_var, latent_space)
                running_loss += loss
                loss.backward()
                self.optimizer.step()
                if not i % 10 and i > 0:

                    print("reconstruction loss {} + KLD loss {} = total loss {}".format(recon_loss, kld_loss, loss))

            train_loss = running_loss / (len(self.dataloader.dataset) * epoch)
            print("Epoch {} : running loss: {}".format(epoch, train_loss))
            return train_loss

    def loss(self, reconstruction, input, mu, log_var, latent_space):
        recons_loss = None
        if self.loss_function == "l1":
            recons_loss = torch.nn.functional.l1_loss(reconstruction, input)
        else:
            recons_loss = torch.nn.functional.mse_loss(reconstruction, input)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kld_loss /= self.dataloader.batch_size*latent_space[-3]*latent_space[-2]*latent_space[-1]
        loss = recons_loss + kld_loss
        return loss, recons_loss, kld_loss
