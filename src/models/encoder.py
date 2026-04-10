import torch
import torch.nn as nn
from typing import Tuple, Optional
from .blocks import DoubleConv, Down, is_complex  # Import des fichiers du projet
import torchcvnn.nn as c_nn


def get_linear(in_features: int, out_features: int, layer_mode: str = "real") -> nn.Module:
    """Route vers une couche linéaire standard ou complexe."""
    if is_complex(layer_mode):
        return nn.Linear(in_features, out_features, dtype=torch.complex64)
    return nn.Linear(in_features, out_features, dtype=torch.float32)


class ConvEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        
        in_channels = cfg["data"]["in_channels"]
        image_size = cfg["data"]["image_size"]
        layer_mode = cfg["data"].get("layer_mode", "real")
        
        # On récupère la liste des canaux
        channels = cfg["model"]["channels"]
        
        def _ensure_list(param, length):
            if isinstance(param, str): return [param] * length
            if isinstance(param, list): return param.copy()
            return param
            
        activation = _ensure_list(cfg["model"]["encoder_activation"], len(channels))
        normalization = _ensure_list(cfg["model"]["encoder_normalization"], len(channels))
        downsampling = _ensure_list(cfg["model"]["encoder_downsampling"], len(channels) - 1)
        
        use_residual = cfg["model"].get("residual", False)
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
                residual=use_residual,
                layer_mode=layer_mode
            )
        )
        
        # 2. Boucle de descente
        for i in range(len(channels) - 1):
            in_ch = channels[i]       # Ex: 32 puis 64
            out_ch = channels[i + 1]  # Ex: 64 puis 128
            
            layers.append(
                Down(
                    in_channels=in_ch, 
                    out_channels=out_ch, 
                    downsampling=downsampling[i],      
                    activation=activation[i + 1],      
                    normalization=normalization[i + 1],
                    size=current_size, 
                    residual=use_residual,
                    layer_mode=layer_mode
                )
            )
            current_size = current_size // 2
            
        
        # --- Création du Pipeline ---
        self.feature_extractor = nn.Sequential(*layers)
        
        # --- Calcul dynamique de la dimension du Bottleneck ---
        self.flattened_dim = channels[-1] * (current_size * current_size)
        self.final_spatial_size = current_size

        # --- Têtes de l'Espace Latent ---
        self.widely_linear = cfg["model"].get("widely_linear", False)
        self.fc_mu = get_linear(self.flattened_dim, latent_dim, layer_mode)  
        
        if layer_mode == "complex":
            if self.widely_linear:
                # WL-CVAE : On a besoin de W_real, W_imag, V_real, V_imag (4 parts)
                self.fc_logvar = get_linear(self.flattened_dim * 2, latent_dim * 4, layer_mode="real")  
            else:
                # CVAE Standard : On a juste besoin de W_real, W_imag (2 parts)
                self.fc_logvar = get_linear(self.flattened_dim * 2, latent_dim * 2, layer_mode="real")  
        else:
            self.fc_logvar = get_linear(self.flattened_dim, latent_dim, layer_mode="real")
    
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        features = self.feature_extractor(x) # 1. Extraction spatiale
        features_flat = torch.flatten(features, start_dim=1) # 2. Aplatissement : [Batch, Canaux, H, W] -> [Batch, Canaux * H * W]
        mu = self.fc_mu(features_flat) # 3. Projection dans l'espace latent
        
        # 4. Traitement spécial pour la variance (Réelle)
        if features_flat.is_complex():
            # On concatène [Réel, Imaginaire] pour ne garder que des Float32
            features_real_imag = torch.cat([features_flat.real, features_flat.imag], dim=1)
            log_var = self.fc_logvar(features_real_imag)
        else:
            log_var = self.fc_logvar(features_flat)
        
        return mu, log_var