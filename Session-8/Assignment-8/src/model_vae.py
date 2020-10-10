import torch
import torch.nn as nn
import torch.nn.functional as F

# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
 
        '''# encoder
        self.enc1 = nn.Linear(in_features=3*64*64, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=32)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=32, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=3*64*64)'''

        # Encoder 
        self.encoder = nn. Sequential(
            nn.Linear(in_features=3*64*64, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=32)
        )

        # Decoder 
        self.decoder = nn. Sequential(
            nn.Linear(in_features=32, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=3*64*64)
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        '''# encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x)
        # get `mu` and `log_var`
        mu = x
        log_var = x
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var'''

        # encoding
        x = self.encoder(x)

        # get `mu` and `log_var`
        mu = x
        log_var = x
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = self.decoder(z)
        reconstruction = torch.sigmoid(x)
        return reconstruction, mu, log_var