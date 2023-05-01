import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F


class VAE(nn.Module):
    """
    """
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims = None, **kwargs) -> None:
        super(VAE, self).__init__()
        self.name = "vae"
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, 
                        out_channels = h_dim, 
                        kernel_size= 3, 
                        stride= 2, 
                        padding = 1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(4*hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(4*hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, 4*hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size = 3,
                        stride = 2,
                        padding = 1,
                        output_padding = 1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding = 1
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1], 
                out_channels = 3,
                kernel_size = 3, 
                padding = 1
            ),
            nn.Tanh()
        )

    def encode(self, input):
        """
        """
        encoding = torch.flatten(self.encoder(input), start_dim=1)
        return self.fc_mu(encoding), self.fc_var(encoding)

    def decode(self, z):
        """
        """
        reconstruction = self.decoder_input(z).view(-1, 512, 2, 2)
        return self.final_layer(self.decoder(reconstruction))
    
    def forward(self, x,  **kwargs):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), x, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps*std + mu
    
    def sample(self, n, device, **kwargs):
        z = torch.randn(n, self.latent_dim).to(device)
        return self.decode(z)
    
    def generate(self, x, **kwargs):
        return self.forward(x)[0]

    def loss_function(self, *args, **kwargs):
        reconstruction, x, mu, log_var = args[:4]
        kld_weight = kwargs['M_N'] 
        return kld(reconstruction, x, mu, log_var, kld_weight)

def kld(reconstruction, x, mu, log_var, kld_weight):
    """
    """
    reconstruction_loss = F.mse_loss(reconstruction, x)
    kld_loss = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
    
    return {
        'KLD': -kld_loss.detach(),
        'loss': reconstruction_loss + kld_weight * kld_loss, 
        'Reconstruction_Loss': reconstruction_loss.detach(), 
    }

transform = [
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(148),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
]