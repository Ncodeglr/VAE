import torch
import torch.nn as nn
from typing import Optional

# Importation de la librairie complexe torchcvnn
try:
    import torchcvnn.nn.modules as c_nn
except ImportError:
    c_nn = None

from .blocks import DoubleConv, Up, get_linear, is_complex  # Import des fichiers du projet


def get_final_layer(in_channels: int, out_channels: int, layer_mode: str = "real") -> nn.Module:
    """
    Dernière couche du décodeur : projette les features vers l'espace image.
    On utilise une convolution 1x1 pour changer la profondeur sans toucher au spatial.
    """
    if is_complex(layer_mode):
        # Pour le complexe, on utilise la conv 1x1 de torchcvnn
        return c_nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class ConvDecoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        # --- 1. Extraction des paramètres ---
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        
        dataset_name = data_cfg["dataset_name"]
        in_channels = data_cfg["in_channels"] # Sortie finale du décodeur
        image_size = data_cfg["image_size"]
        layer_mode = data_cfg.get("layer_mode", "real")
        
        channels = model_cfg["channels"] # ex: [32, 64, 128]
        latent_dim = model_cfg["latent_dim"]
        
        # --- 2. Calcul de la géométrie inverse ---
        num_layers = len(channels) - 1
        self.final_size = image_size // (2 ** num_layers)
        self.final_channels = channels[-1]
        self.flattened_dim = self.final_channels * (self.final_size ** 2)

        # --- 3. Passage Latent -> Spatial ---
        from .encoder import get_linear # Réutilisation de notre routeur
        self.latent_to_features = get_linear(latent_dim, self.flattened_dim, layer_mode)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.final_channels, self.final_size, self.final_size))

        # --- 4. Blocs d'Upsampling ---
        layers = []
        curr_size = self.final_size
        reversed_channels = list(reversed(channels)) # [128, 64, 32]

        for i in range(len(reversed_channels) - 1):
            layers.append(
                Up(
                    in_channels=reversed_channels[i],
                    out_channels=reversed_channels[i+1],
                    upsampling=model_cfg.get("upsampling", "transpose"),
                    activation=model_cfg.get("activation", "relu"),
                    normalization=model_cfg.get("normalization", "bn2d"),
                    size=curr_size,
                    layer_mode=layer_mode
                )
            )
            curr_size *= 2

        self.upsample_path = nn.Sequential(*layers)

        # --- 5. Tête de sortie finale ---
        # On réduit les derniers filtres (ex: 32) vers le nombre de canaux image (ex: 1)
        self.final_projection = get_final_layer(reversed_channels[-1], in_channels, layer_mode)
        
        # --- 6. Activation de sortie conditionnelle ---
        if dataset_name.upper() == "MNIST":
            self.output_activation = nn.Sigmoid() # Pour MNIST, les pixels sont dans [0, 1], donc Sigmoid est idéale.
        elif dataset_name.upper() == "CIFAR10":
            self.output_activation = nn.Tanh() # Si normalisé entre -1 et 1 car pour d'autres (ex: radar ou images normalisées [-1, 1]), on pourrait vouloir Tanh ou Identity.
        else:
            self.output_activation = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Z (Batch, Latent) -> Flat (Batch, 3136)
        x = self.latent_to_features(z)
        
        # Flat -> Cube (Batch, 128, 7, 7)
        x = self.unflatten(x)
        
        # Cube -> Image Feature Maps (Batch, 32, 28, 28)
        x = self.upsample_path(x)
        
        # Image Feature Maps -> Image finale (Batch, 1, 28, 28)
        x = self.final_projection(x)
        
        return self.output_activation(x)