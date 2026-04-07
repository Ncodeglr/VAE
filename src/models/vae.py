import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

# Import des fichiers du projet avec les class Encoder et Decoder
from .encoder import ConvEncoder
from .decoder import ConvDecoder

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) Modulaire - Combine l'Encodeur et le Décodeur avec le 'Reparameterization Trick'.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.layer_mode = cfg["data"].get("layer_mode", "real")

        # 1. Création de l'Encoder et Decoder
        self.encoder = ConvEncoder(cfg)
        self.decoder = ConvDecoder(cfg)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
       'Reparameterization Trick' - Permet au gradient de circuler malgré l'échantillonnage aléatoire.
        z = mu + sigma*std 
        """
        # On calcule l'écart-type : sigma = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        
        # On génère un bruit blanc epsilon de la même taille
        eps = torch.randn_like(std)
        
        # On décale et on étire le bruit par mu et std
        return mu + std*eps 

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passage complet dans le VAE.
        Returns:
            reconstruction: L'image reconstruite
            mu: La moyenne de l'espace latent
            logvar: Le log de la variance de l'espace latent
        """
        # 1. ENCODE : Image -> Paramètres de distribution
        mu, logvar = self.encoder(x)
        
        # 2. SAMPLE : Reparameterization trick pour obtenir Z
        z = self.reparameterize(mu, logvar)
        
        # 3. DECODE : Z -> Image reconstruite
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar

    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Génère de nouvelles images à partir de rien (en piochant dans le prior N(0,1)).
        num_samples : Nombre de nouvelles images qu'on veut générer d'un seul coup
        """
        latent_dim = self.cfg["model"]["latent_dim"]
        z = torch.randn(num_samples, latent_dim).to(device)  # On pioche des points au hasard dans l'espace latent
        
        # Le décodeur les transforme en images
        return self.decoder(z)