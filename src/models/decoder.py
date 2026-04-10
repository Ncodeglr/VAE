import torch
import torch.nn as nn
from typing import Optional

# Importation de la librairie complexe torchcvnn
try:
    import torchcvnn.nn.modules as c_nn
except ImportError:
    c_nn = None

from .blocks import DoubleConv, Up, is_complex 
from .encoder import get_linear                 

def get_final_layer(in_channels: int, out_channels: int, layer_mode: str = "real") -> nn.Module:
    """
    Dernière couche du décodeur : projette les features vers l'espace image.
    On utilise une convolution 1x1 pour changer la profondeur sans toucher au spatial.
    """
    if is_complex(layer_mode):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=torch.complex64)
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=torch.float32)


class ConvDecoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        # --- 1. Extraction des paramètres ---
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        
        dataset_name = data_cfg["dataset_name"]
        in_channels = data_cfg["in_channels"] 
        image_size = data_cfg["image_size"]
        layer_mode = data_cfg.get("layer_mode", "real")
        
        channels = model_cfg["channels"]
        
        def _ensure_list(param, length):
            if isinstance(param, str): return [param] * length
            if isinstance(param, list): return param.copy()
            return param
            
        # 1. Fallback symétrique : Si "decoder_activation" n'existe pas, on prend "encoder_activation" et on l'inverse [::-1]
        raw_act = model_cfg.get("decoder_activation", model_cfg["encoder_activation"][::-1])
        raw_norm = model_cfg.get("decoder_normalization", model_cfg["encoder_normalization"][::-1])
        raw_up = model_cfg.get("decoder_upsampling", ["transpose"] * (len(channels) - 1))
        
        # 2. Création des variables locales (SANS "self." et plus besoin de ré-inverser)
        activation = _ensure_list(raw_act, len(channels))
        normalization = _ensure_list(raw_norm, len(channels))
        upsampling = _ensure_list(raw_up, len(channels) - 1)
        
        latent_dim = model_cfg["latent_dim"]
        
        # --- 2. Calcul de la géométrie inverse ---
        num_layers = len(channels) - 1
        self.final_size = image_size // (2 ** num_layers)
        self.final_channels = channels[-1]
        self.flattened_dim = self.final_channels * (self.final_size ** 2)

        # --- 3. Passage Latent -> Spatial ---
        self.latent_to_features = get_linear(latent_dim, self.flattened_dim, layer_mode)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.final_channels, self.final_size, self.final_size))

        # --- 4. Blocs d'Upsampling ---
        layers = []
        curr_size = self.final_size
        reversed_channels = list(reversed(channels)) # Ex: [128, 64, 32]

        for i in range(len(reversed_channels) - 1):
            layers.append(
                Up(
                    in_channels=reversed_channels[i],
                    out_channels=reversed_channels[i+1],
                    upsampling=upsampling[i],         
                    activation=activation[i],         
                    normalization=normalization[i],   
                    size=curr_size,
                    layer_mode=layer_mode
                )
            )
            curr_size *= 2

        self.upsample_path = nn.Sequential(*layers)

        # --- 5. Tête de sortie finale ---
        self.final_projection = get_final_layer(reversed_channels[-1], in_channels, layer_mode) 
        
        # --- 6. Activation de sortie conditionnelle ---
        if dataset_name.upper() == "MNIST":
            self.output_activation = nn.Sigmoid() 
        elif dataset_name.upper() == "COMPLEXMNIST":
            self.output_activation = nn.Identity() 
        elif dataset_name.upper() == "CIFAR10":
            self.output_activation = nn.Tanh()      
        else:
            self.output_activation = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Passe de l'espace latent (Z) à la reconstruction de l'image.
        """
        x = self.latent_to_features(z) # 1. Projection linéaire : Z (vecteur) -> Features (vecteur plus long) Ex: [Batch, 64] -> [Batch, 6272]
        x = self.unflatten(x) # 2. Dépliage Spatial (Unflatten) : Vecteur 1D -> Cube 3D (Feature Maps) Ex: [Batch, 6272] -> [Batch, 128, 7, 7]
        x = self.upsample_path(x)  # 3. Upsampling (Déconvolutions) : Agrandit l'image et réduit la profondeur Ex: [Batch, 128, 7, 7] -> [Batch, 64, 14, 14] -> [Batch, 32, 28, 28]
        x = self.final_projection(x) # 4. Projection Finale (Conv 1x1) : Ajuste le nombre de canaux finaux Ex: [Batch, 32, 28, 28] -> [Batch, 1, 28, 28] (1 canal pour MNIST)
        return self.output_activation(x) # 5. Activation Finale : Borne les pixels (ex: Sigmoid pour [0,1], Identity sinon)