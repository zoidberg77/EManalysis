import torch
import torch.nn as nn


class Conv3dVAE_simple(nn.Module):
    def __init__(self):
        super(Conv3dVAE_simple, self).__init__()
        kernel_size = 4
        stride = 1
        padding = 0
        hidden_dims = [16, 32, 64]
        modules = []
        in_channels = 1

        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=dim, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm3d(dim),
                    nn.LeakyReLU())

            )
            in_channels = dim
        self.encoder = nn.Sequential(*modules)

        hidden_dims.reverse()
        modules = []

        for dim in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[dim], out_channels=hidden_dims[dim + 1], kernel_size=kernel_size,
                                       stride=stride, padding=padding),
                    nn.BatchNorm3d(hidden_dims[dim + 1]),
                    nn.LeakyReLU())
            )
            in_channels = dim

        modules.append(
            nn.Sequential(
                nn.ConvTranspose3d(hidden_dims[-1], out_channels=1, kernel_size=kernel_size, stride=stride,
                                   padding=padding),
            ))
        self.decoder = nn.Sequential(*modules)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        x = self.encoder(x)
        mu = x
        log_var = x
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z)
        reconstruction = torch.sigmoid(x)
        return reconstruction, mu, log_var


class Conv3dVAE(nn.Module):
    def __init__(self):
        super(Conv3dVAE, self).__init__()
        kernel_size = 4
        stride = 1
        padding = 0
        hidden_dims = [4, 16]
        modules = []
        in_channels = 1
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
                    nn.BatchNorm3d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv3d(hidden_dims[-1], out_channels=hidden_dims[0],
                          kernel_size=kernel_size, stride=stride, padding=padding),
                nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
                nn.BatchNorm3d(hidden_dims[0]))
        )

        self.encoder = nn.Sequential(*modules)

        modules = []

        modules.append(
            nn.Sequential(
                nn.ConvTranspose3d(hidden_dims[0], out_channels=hidden_dims[-1],
                                   kernel_size=kernel_size, stride=stride, padding=padding),
                nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
                nn.ReLU())
        )
        hidden_dims.reverse()

        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_channels, out_channels=h_dim,
                                       kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.MaxUnpool3d(kernel_size=kernel_size, stride=stride),
                    nn.BatchNorm3d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.ConvTranspose3d(hidden_dims[-1], out_channels=1,
                                   kernel_size=kernel_size, stride=stride, padding=padding),
            ))

        self.decoder = nn.Sequential(*modules)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        x = self.encoder(x)
        mu = x
        log_var = x
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z)
        reconstruction = torch.sigmoid(x)
        return reconstruction, mu, log_var