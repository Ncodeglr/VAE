import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
import torchcvnn.nn as c_nn

# Import des fichiers du projet avec les class Encoder et Decoder
from .encoder import ConvEncoder
from .decoder import ConvDecoder

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) Modulaire - Gère le mode Réel et Complexe (CVNN).
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.layer_mode = cfg["data"].get("layer_mode", "real")

        # 1. Création de l'Encoder et Decoder
        self.encoder = ConvEncoder(cfg)
        self.decoder = ConvDecoder(cfg)

        self._apply_weights() # Initialisation

    def _apply_weights(self):
        """
        Initialise tous les poids du VAE. 
        S'adapte automatiquement si la couche est complexe ou réelle.
        """
        init_type = self.cfg["model"].get("weight_init", "kaiming_normal").lower()
        print(f"⚖️ Initialisation des poids du modèle : {init_type}")
        
        for m in self.modules():
            # On cible les convolutions, convolutions transposées et les couches linéaires
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                
                # Sécurité : vérifier que la couche possède bien des poids
                if getattr(m, 'weight', None) is not None:
                    is_cplx = m.weight.is_complex()
                    
                    if is_cplx and c_nn is not None:
                        # --- INITIALISATION COMPLEXE (cvnn) ---
                        if init_type == "xavier_normal":
                            c_nn.modules.initialization.complex_xavier_normal_(m.weight)
                        elif init_type in ["xavier_uniform", "xavier"]:
                            c_nn.modules.initialization.complex_xavier_uniform_(m.weight)
                        elif init_type in ["kaiming_normal", "kaiming"]:
                            c_nn.modules.initialization.complex_kaiming_normal_(m.weight, nonlinearity='relu')
                        elif init_type == "kaiming_uniform":
                            c_nn.modules.initialization.complex_kaiming_uniform_(m.weight, nonlinearity='relu')
                            
                    elif not is_cplx:
                        # --- INITIALISATION RÉELLE (PyTorch standard) ---
                        if init_type in ["xavier_uniform", "xavier"]:
                            nn.init.xavier_uniform_(m.weight)
                        elif init_type == "xavier_normal":
                            nn.init.xavier_normal_(m.weight)
                        elif init_type in ["kaiming_normal", "kaiming"]:
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        elif init_type == "kaiming_uniform":
                            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                        elif init_type == "rayleigh":
                            with torch.no_grad():
                                sigma = 0.02 
                                m.weight.uniform_(1e-7, 1.0)
                                m.weight.copy_(sigma * torch.sqrt(-2 * torch.log(m.weight)))

                # --- BIAIS ---
                # Les biais sont toujours mis à zéro au départ (qu'ils soient complexes ou réels)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu: torch.Tensor, var_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        widely_linear = self.cfg["model"].get("widely_linear", False)
        
        if mu.is_complex():
            latent_dim = mu.size(1)
            
            # Tirage du bruit blanc complexe circulaire
            eps_real = torch.randn_like(mu.real)
            eps_imag = torch.randn_like(mu.imag)
            eps = torch.complex(eps_real, eps_imag)
            
            if widely_linear:
                # WL-CVAE : On utilise W et V
                W_real = var_params[:, :latent_dim]
                W_imag = var_params[:, latent_dim:2*latent_dim]
                V_real = var_params[:, 2*latent_dim:3*latent_dim]
                V_imag = var_params[:, 3*latent_dim:]

                W = torch.complex(W_real, W_imag)
                V = torch.complex(V_real, V_imag)
            else:
                # CVAE Standard : On utilise W, et on force V à zéro !
                W_real = var_params[:, :latent_dim]
                W_imag = var_params[:, latent_dim:2*latent_dim]
                
                W = torch.complex(W_real, W_imag)
                V = torch.zeros_like(W)  # Si différente de 0 alors distribution non circulaire
                
            # La formule reste la même, mais si V est nul, le terme s'annule tout seul
            z = mu + W * eps + V * torch.conj(eps)
            return z, W, V
            
        else:
            std = torch.exp(0.5 * var_params)
            eps = torch.randn_like(std)
            return mu + std * eps, None, None

    def forward(self, x: torch.Tensor) -> Tuple:
        mu, var_params = self.encoder(x)
        
        if self.layer_mode == "complex":
            # On récupère z, mais aussi W et V
            z, W, V = self.reparameterize(mu, var_params)
            reconstruction = self.decoder(z)
            
            log_sigma2_dec = torch.zeros_like(mu.real[:, 0].mean())
            # On donne W et V à la Loss au lieu de p1 et p2
            return reconstruction, mu, W, V, log_sigma2_dec
        else:
            z, _, _ = self.reparameterize(mu, var_params)
            reconstruction = self.decoder(z)
            return reconstruction, mu, var_params


    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Génère de nouvelles images en piochant dans l'espace latent.
        """
        latent_dim = self.cfg["model"]["latent_dim"]
        
        # Génération d'un bruit initial complexe
        if self.layer_mode == "complex":
            z_real = torch.randn(num_samples, latent_dim, device=device)
            z_imag = torch.randn(num_samples, latent_dim, device=device)
            z = torch.complex(z_real, z_imag)
        else:
            z = torch.randn(num_samples, latent_dim, device=device)
            
        return self.decoder(z)