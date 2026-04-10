import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==============================================================================
# Helper Functions
# ==============================================================================

def _to_spatial_domain(img_complex) -> np.ndarray:
    """
    Utilitaire pour repasser du domaine fréquentiel (FFT) au domaine spatial (Pixels réels).
    Gère automatiquement les Tenseurs PyTorch et les Tableaux NumPy.
    """
    # 1. Sécurité : on vérifie le type d'entrée
    if isinstance(img_complex, np.ndarray):
        t = torch.from_numpy(img_complex)
    else:
        # C'est déjà un tenseur (ex: venant de model.sample()), on le copie juste
        t = img_complex.clone().detach()
        
    # 2. On annule le shift
    t = torch.fft.ifftshift(t)
    
    # 3. On fait l'inverse FFT
    t_spatial = torch.fft.ifft2(t)
    
    # 4. On multiplie par 28 (pour annuler notre division) et on repasse en NumPy
    return (t_spatial.real * 28.0).cpu().numpy()


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_losses(train_losses: list, valid_losses: list = None) -> plt.Figure:
    """Plots the training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    
    if valid_losses is not None:
        plt.plot(epochs, valid_losses, label="Validation Loss", color="#ff7f0e", linewidth=2)
        
    plt.xlabel("Epochs")
    plt.ylabel("Loss (ELBO)")
    plt.title("VAE Learning Curves")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    return fig


def plot_reconstructions(
    inputs: torch.Tensor, 
    outputs: torch.Tensor, 
    dataset_type: str = "grayscale", 
    num_samples: int = 5
) -> plt.Figure:
    """
    Affiche les images originales à côté des reconstructions.
    Gère automatiquement les données complexes (via _to_spatial_domain).
    """
    B, C, H, W = inputs.shape
    num_samples = min(num_samples, B)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    
    # Sécurité si num_samples = 1 (axes devient 1D)
    if num_samples == 1:
        axes = axes[:, None]
    
    for i in range(num_samples):
        # 1. Passage en NumPy
        img_in = inputs[i].detach().cpu().squeeze().numpy()
        img_out = outputs[i].detach().cpu().squeeze().numpy()
        
        # 2. Handle Complex Data (Frequency Domain -> Spatial Domain)
        if np.iscomplexobj(img_in) or "complex" in dataset_type.lower():
            img_in = _to_spatial_domain(img_in)
        if np.iscomplexobj(img_out) or "complex" in dataset_type.lower():
            img_out = _to_spatial_domain(img_out)
            
        # On clamp (force) entre 0 et 1 pour éviter les minuscules erreurs d'approximation
        img_in = np.clip(img_in, 0, 1)
        img_out = np.clip(img_out, 0, 1)
        
        # 3. Affichage
        # Ligne 1 : Original
        axes[0, i].imshow(img_in, cmap="gray")
        axes[0, i].axis("off")
        if i == 0: axes[0, i].set_title("Original")
            
        # Ligne 2 : Reconstruction
        axes[1, i].imshow(img_out, cmap="gray")
        axes[1, i].axis("off")
        if i == 0: axes[1, i].set_title("Reconstruit")
            
    plt.tight_layout()
    return fig


def plot_latent_space(latents: np.ndarray, labels: np.ndarray = None, method: str = "pca") -> plt.Figure:
    """
    Reduces the latent space vectors to 2D for visualization.
    Automatically handles complex latent vectors by concatenating Real and Imaginary parts.
    """
    print(f"🔄 Reducing dimensions via {method.upper()}...")
    
    # Handle Complex Latents: Concatenate [Real, Imag] to make them compatible with sklearn
    if np.iscomplexobj(latents):
        latents = np.concatenate([latents.real, latents.imag], axis=1)

    # Perform Dimensionality Reduction if D > 2
    if latents.shape[1] > 2:
        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else: # Default to PCA
            reducer = PCA(n_components=2, random_state=42)
            
        z_2d = reducer.fit_transform(latents)
    else:
        z_2d = latents

    fig = plt.figure(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        palette = sns.color_palette("tab10", len(unique_labels))
        sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=labels, palette=palette, alpha=0.7, s=40, legend="full")
    else:
        plt.scatter(z_2d[:,0], z_2d[:,1], alpha=0.5, s=20)
        
    plt.title(f"VAE Latent Space ({method.upper()})")
    plt.grid(True, alpha=0.3)
    
    return fig


def plot_generated_samples(generated_images: np.ndarray, dataset_type: str = "mnist") -> plt.Figure:
    """
    Displays a grid of newly generated images sampled from the prior distribution N(0,1).
    Automatically handles complex frequency data by performing an iFFT.
    """
    num_samples = len(generated_images)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.5, grid_size * 2.5))
    
    # Handle the case where there's only 1 sample (axes is not an array)
    if num_samples == 1:
        axes = np.array([axes])
        
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < num_samples:
            img = generated_images[i]
            
            # Handle Complex Data
            if np.iscomplexobj(img) or "complex" in dataset_type.lower():
                img = _to_spatial_domain(img)
            
            # Formatting based on dataset type
            if "mnist" in dataset_type.lower() or "grayscale" in dataset_type.lower() or len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[0] == 1):
                img = img.squeeze()
                axes[i].imshow(np.clip(img, 0, 1), cmap='gray')
            else:
                # Transpose RGB images to (H, W, C)
                if len(img.shape) == 3 and img.shape[0] in [3, 4]:
                    img = np.transpose(img, (1, 2, 0))
                axes[i].imshow(np.clip(img, 0, 1))
                
        axes[i].axis('off')
        
    plt.suptitle("New Generations (Sampled from z ~ N(0,1))", fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_complex_reconstructions(
    inputs: torch.Tensor, 
    outputs: torch.Tensor, 
    num_samples: int = 5
) -> plt.Figure:
    """
    Affichage expert pour le domaine complexe : Sépare la Magnitude et la Phase.
    Inspiré des standards d'imagerie radar et du repository cvnn.
    """
    B = inputs.shape[0]
    num_samples = min(num_samples, B)
    
    # On crée une grille : 4 lignes (Amp In, Phase In, Amp Out, Phase Out)
    fig, axes = plt.subplots(4, num_samples, figsize=(2.5 * num_samples, 10))
    if num_samples == 1: axes = axes[:, None]
    
    for i in range(num_samples):
        # 1. Extraction en Numpy
        img_in = inputs[i].detach().cpu().squeeze().numpy()
        img_out = outputs[i].detach().cpu().squeeze().numpy()
        
        # 2. Séparation Magnitude (Abs) et Phase (Angle)
        amp_in = np.abs(img_in)
        phase_in = np.angle(img_in)
        
        amp_out = np.abs(img_out)
        phase_out = np.angle(img_out)
        
        # --- Ligne 1 : Amplitude Originale ---
        ax = axes[0, i]
        ax.imshow(amp_in, cmap="gray")
        ax.axis("off")
        if i == 0: 
            ax.set_title("Original", fontweight='bold')
            ax.set_ylabel("Amplitude", size='large', fontweight='bold')
            
        # --- Ligne 2 : Phase Originale ---
        ax = axes[1, i]
        # vmin/vmax forcent la phase à rester entre -PI et +PI. 
        # 'twilight' est une colormap cyclique parfaite pour les angles.
        ax.imshow(phase_in, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.axis("off")
        if i == 0: 
            ax.set_ylabel("Phase", size='large', fontweight='bold')

        # --- Ligne 3 : Amplitude Reconstruite ---
        ax = axes[2, i]
        ax.imshow(amp_out, cmap="gray")
        ax.axis("off")
        if i == 0: 
            ax.set_title("Reconstruit", fontweight='bold')
            ax.set_ylabel("Amplitude", size='large', fontweight='bold')

        # --- Ligne 4 : Phase Reconstruite ---
        ax = axes[3, i]
        ax.imshow(phase_out, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.axis("off")
        if i == 0: 
            ax.set_ylabel("Phase", size='large', fontweight='bold')

    fig.tight_layout()
    return fig