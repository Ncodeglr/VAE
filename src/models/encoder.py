import torch
import torch.nn as nn
from typing import Tuple, Optional

from .blocks import DoubleConv, Down, is_complex  # Import des fichiers du projet

# Importation de la librairie complexe torchcvnn
try:
    import torchcvnn.nn.modules as c_nn
except ImportError:
    c_nn = None


def get_linear(in_features: int, out_features: int, layer_mode: str = "real") -> nn.Module:
    """Route vers une couche linéaire standard ou complexe."""
    if is_complex(layer_mode):
        return c_nn.Linear(in_features, out_features)
    return nn.Linear(in_features, out_features)


class ConvEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        
        in_channels = cfg["data"]["in_channels"]
        image_size = cfg["data"]["image_size"]
        layer_mode = cfg["data"].get("layer_mode", "real")
        
        # On récupère les listes via le fichier dans le dossier configs
        channels = cfg["model"]["channels"]
        activation = cfg["model"]["activation"]
        normalization = cfg["model"]["normalization"]
        downsampling = cfg["model"]["downsampling"]
        latent_dim = cfg["model"]["latent_dim"]

        layers = []
        current_size = image_size
        
        # 1. Couche initiale (Première DoubleConv sans réduction spatiale)
        layers.append(
            DoubleConv(
                in_channels=in_channels, 
                out_channels=channels[0],  
                activation=activation[0], 
                normalization=normalization[0], 
                size=current_size, 
                layer_mode=layer_mode
            )
        )
        
        # 2. Boucle de descente personnalisée - On boucle sur le reste de la liste channels
        for i in range(len(channels) - 1):
            in_ch = channels[i]       # Ex: 32 puis 64
            out_ch = channels[i + 1]  # Ex: 64 puis 128
            
            layers.append(
                Down(
                    in_channels=in_ch, 
                    out_channels=out_ch, 
                    downsampling=downsampling[i],      # Le pooling spécifique à cette couche 
                    activation=activation[i + 1],      # L'activation spécifique
                    normalization=normalization[i + 1],# La normalisation spécifique
                    size=current_size, 
                    layer_mode=layer_mode
                )
            )
            current_size = current_size // 2
            
        
        
        # --- Création du Pipeline ---
        # L'opérateur '*' (unpacking) déballe la liste Python 'layers' en arguments individuels.
        # nn.Sequential les assemble ensuite en un pipeline continu, ce qui permet d'exécuter toutes les couches automatiquement et dans l'ordre lors du forward(),sans avoir à les écrire une par une à la main.
        self.feature_extractor = nn.Sequential(*layers)
        
        
        # --- Calcul dynamique de la dimension du Bottleneck ---
        # Pour passer du monde 2D (Convolutions) au monde 1D (Couches Linéaires), on doit aplatir (Flatten) le tenseur.
        # La formule est : Profondeur finale * (Hauteur * Largeur).
        # Cela permet à la couche nn.Linear suivante de savoir combien de neurones elle va recevoir en entrée.
        self.flattened_dim = channels[-1] * (current_size*current_size)
        
        self.final_spatial_size = current_size

        # --- Têtes de l'Espace Latent - Paramètres de la distribution ---
        # fc_mu : Prédit la position centrale (moyenne) dans l'espace latent.
        # fc_logvar : Prédit l'incertitude (log-variance) - On utilise le log de la variance pour permettre au réseau de sortir des valeurs entre -inf et +inf (stabilité numérique) tout en garantissant une variance positive après passage à l'exponentielle (exp(logvar)).
        self.fc_mu = get_linear(self.flattened_dim, latent_dim, layer_mode)
        
        # logvar : Même en mode complexe, la variance (incertitude) est représentée par des paramètres réels. 
        if layer_mode == "complex":
             self.fc_logvar = get_linear(self.flattened_dim * 2, latent_dim * 2, layer_mode="real")  # En mode complexe, on prédit souvent la variance des parties Réelle et Imaginaire.
        else:
             self.fc_logvar = get_linear(self.flattened_dim, latent_dim, layer_mode="real")
             


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            features = self.feature_extractor(x) # 1. Extraction spatiale
            features_flat = torch.flatten(features, start_dim=1) # 2. Aplatissement : [Batch, Canaux, H, W] -> [Batch, Canaux * H * W]
            
            # 3. Projection dans l'espace latent
            mu = self.fc_mu(features_flat)
            log_var = self.fc_logvar(features_flat)
            
            return mu, log_var