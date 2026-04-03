import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Importation de la librairie complexe torchcvnn
try:
    import torchcvnn.nn.modules as c_nn
except ImportError:
    c_nn = None

# --- 1. FONCTIONS UTILITAIRES DE ROUTAGE ET D'ENRICHISSEMENT ---

def is_complex(layer_mode: str) -> bool:
    """Vérifie si le mode de la couche est complexe."""
    return layer_mode.lower() in ["complex", "split"]

def get_conv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, layer_mode: str = "real") -> nn.Module:
    if is_complex(layer_mode):
        return c_nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

def get_conv_transpose2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, layer_mode: str = "real") -> nn.Module:
    if is_complex(layer_mode):
        return c_nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

def get_activation(choice: Optional[str], layer_mode: str = "real") -> nn.Module:
    """Route vers l'activation demandée avec les ajouts complexes spécifiques."""
    if choice is None or choice.lower() == "none":
        return nn.Identity()
    
    choice = choice.lower()
    complex_mode = is_complex(layer_mode)

    if complex_mode:
        if choice in ["relu", "crelu"]: return c_nn.CReLU()
        elif choice == "modrelu": return c_nn.modReLU()
        elif choice == "zrelu": return c_nn.zReLU()
        elif choice == "cardioid": return c_nn.Cardioid()
        else: return c_nn.CReLU() # Fallback complexe
    else:
        # Fallbacks réels
        if choice in ["relu", "crelu"]: return nn.ReLU(inplace=True)
        elif choice == "leaky_relu": return nn.LeakyReLU(0.2, inplace=True)
        elif choice == "gelu": return nn.GELU()
        else: return nn.ReLU(inplace=True)

def get_normalization(choice: Optional[str], channels: int, size: Optional[int] = None, layer_mode: str = "real") -> nn.Module:
    """Route vers la normalisation en intégrant LayerNorm et RMSNorm."""
    if choice is None or choice.lower() == "none":
        return nn.Identity()
    
    choice = choice.lower()
    complex_mode = is_complex(layer_mode)

    if choice == "bn1d":
        return c_nn.BatchNorm1d(channels) if complex_mode else nn.BatchNorm1d(channels)
    elif choice == "bn2d" or choice == "batch":
        return c_nn.BatchNorm2d(channels) if complex_mode else nn.BatchNorm2d(channels)
    elif choice == "ln":
        if size is None: raise ValueError("Le paramètre 'size' est requis pour LayerNorm (ln).")
        return c_nn.LayerNorm([channels, size, size]) if complex_mode else nn.LayerNorm([channels, size, size])
    elif choice == "rms":
        if size is None: raise ValueError("Le paramètre 'size' est requis pour RMSNorm (rms).")
        return c_nn.RMSNorm(size) if complex_mode else nn.RMSNorm(size)
    
    # Fallback supplémentaire
    elif choice == "instance":
        return nn.InstanceNorm2d(channels, affine=True) if not complex_mode else c_nn.BatchNorm2d(channels)
    else:
        return nn.Identity()

def get_pool(choice: str, channels: int, layer_mode: str = "real") -> nn.Module:
    """Route vers le bon downsampling (Max, Avg, ou Strided Conv)."""
    choice = choice.lower()
    complex_mode = is_complex(layer_mode)
    
    if choice in ["max", "maxpool"]:
        return c_nn.MaxPool2d(kernel_size=2, stride=2) if complex_mode else nn.MaxPool2d(kernel_size=2, stride=2)
    elif choice in ["avg", "avgpool"]:
        return c_nn.AvgPool2d(kernel_size=2, stride=2) if complex_mode else nn.AvgPool2d(kernel_size=2, stride=2)
    elif choice == "strided":  # Une convolution 2x2 avec stride de 2 réduit l'image de moitié en apprenant la réduction 
       if complex_mode:
            return nn.Conv2d(channels, channels, kernel_size=2, stride=2, dtype=torch.complex64)
       else:
            return nn.Conv2d(channels, channels, kernel_size=2, stride=2, dtype=torch.float32)
    else:
        return nn.Identity()

def concat(x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Concatène deux tenseurs avec auto-padding."""
    if x2 is None: return x1
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return torch.cat([x2, x1], dim=1)



def get_linear(in_features: int, out_features: int, layer_mode: str = "real") -> nn.Module:
    """Route vers une couche linéaire standard ou complexe."""
    if is_complex(layer_mode):
        try:
            import torchcvnn.nn.modules as c_nn
            return c_nn.Linear(in_features, out_features)
        except ImportError:
            pass # Si cvnn n'est pas installé, on fallback sur du réel ou on lève une erreur
    return nn.Linear(in_features, out_features)




# --- 2. LES BRIQUES DE BASE ---

class SingleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, activation: str = "crelu", normalization: str = "bn2d", size: Optional[int] = None, layer_mode: str = "real"):
        super().__init__()
        bias = (normalization is None or normalization.lower() == "none")
        
        self.conv = get_conv2d(in_channels, out_channels, kernel_size, stride, padding, bias, layer_mode)
        self.norm = get_normalization(normalization, out_channels, size, layer_mode)
        self.act = get_activation(activation, layer_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "crelu", normalization: str = "bn2d", size: Optional[int] = None, residual: bool = False, layer_mode: str = "real"):
        super().__init__()
        self.residual = residual
        bias = (normalization is None or normalization.lower() == "none")
        
        self.conv1 = SingleConv(in_channels, out_channels, activation=activation, normalization=normalization, size=size, layer_mode=layer_mode)
        
        self.conv2 = get_conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias, layer_mode=layer_mode)
        self.norm2 = get_normalization(normalization, out_channels, size, layer_mode)
        self.act2 = get_activation(activation, layer_mode)
        
        if self.residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                get_conv2d(in_channels, out_channels, kernel_size=1, bias=False, layer_mode=layer_mode),
                get_normalization(normalization, out_channels, size, layer_mode)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x) if self.residual else None
        out = self.conv1(x)
        out = self.norm2(self.conv2(out))
        if self.residual:
            out += identity
        return self.act2(out)


# --- 3. LES BLOCS DE CHANGEMENT D'ÉCHELLE ---

class Down(nn.Module):
    """
    Bloc de descente : Réduit la taille spatiale (H, W) par 2,
    puis applique une Double Convolution pour extraire les caractéristiques.
    """
    def __init__(self, in_channels: int, out_channels: int, downsampling: str = "strided", 
                 activation: str = "relu", normalization: str = "batch", 
                 size: int = None, layer_mode: str = "real"):
        super().__init__()
        
        layers = []
        
        # --- 1. L'étape cruciale : Réduire l'image par 2 ---
        if downsampling == "max":
            layers.append(nn.MaxPool2d(2))
        elif downsampling == "avg":
            layers.append(nn.AvgPool2d(2))
        else: 
            # Mode "strided" (Par défaut) : La convolution saute 1 pixel sur 2 (stride=2)
            if is_complex(layer_mode):
                layers.append(c_nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1))
                
        # --- 2. L'étape de traitement : Double Convolution ---
        layers.append(
            DoubleConv(
                in_channels=in_channels, 
                out_channels=out_channels,
                activation=activation,
                normalization=normalization,
                size=size // 2 if size else None, # On prévient que la taille a été divisée
                layer_mode=layer_mode
            )
        )
        
        self.down_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_block(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsampling: str = "transpose", use_skip: bool = False, activation: str = "crelu", normalization: str = "bn2d", size: Optional[int] = None, residual: bool = False, layer_mode: str = "real"):
        super().__init__()
        self.use_skip = use_skip
        
        if upsampling == "transpose": #Division du nombre de cannaux par 2 
            self.up = get_conv_transpose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, layer_mode=layer_mode)
        elif upsampling == "interpolate":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                get_conv2d(in_channels, in_channels // 2, kernel_size=1, layer_mode=layer_mode)
            )
        else:
            raise ValueError(f"Upsampling non supporté : {upsampling}")
            
        conv_in_channels = in_channels if use_skip else in_channels // 2
        
        # Note: Si on utilise LayerNorm, la 'size' donnée à la DoubleConv suivante est doublée
        next_size = size * 2 if size is not None else None
        self.conv = DoubleConv(conv_in_channels, out_channels, activation, normalization, next_size, residual, layer_mode)

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = self.up(x1)
        x = concat(x1, x2) if self.use_skip and x2 is not None else x1
        return self.conv(x)