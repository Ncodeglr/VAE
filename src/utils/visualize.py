import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_losses(train_losses: list, valid_losses: list = None) -> plt.Figure:
    """Trace la courbe d'apprentissage façon cvnn."""
    epochs = range(1, len(train_losses) + 1)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    if valid_losses is not None:
        plt.plot(epochs, valid_losses, label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (ELBO)")
    plt.title("Courbes d'apprentissage du VAE")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    return fig

def plot_reconstructions(inputs: torch.Tensor, outputs: torch.Tensor, num_samples: int = 5) -> plt.Figure:
    """Affiche les images originales à côté des reconstructions."""
    B, C, H, W = inputs.shape
    num_samples = min(num_samples, B)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    
    for i in range(num_samples):
        # Préparation de l'image (marche pour MNIST, en 2D)
        img_in = inputs[i].detach().cpu().squeeze().numpy()
        img_out = outputs[i].detach().cpu().squeeze().numpy()
        
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

def plot_latent_space(latents: np.ndarray, labels: np.ndarray, method="pca") -> plt.Figure:
    """Réduit l'espace latent (ex: 16D) en 2D pour le visualiser."""
    print(f"🔄 Réduction de dimension via {method.upper()}...")
    
    if latents.shape[1] > 2:
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else: # pca
            reducer = PCA(n_components=2, random_state=42)
        z_2d = reducer.fit_transform(latents)
    else:
        z_2d = latents

    fig = plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    
    sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=labels, palette=palette, alpha=0.7, s=40, legend="full")
    plt.title(f"Espace Latent du VAE ({method.upper()})")
    plt.grid(True, alpha=0.3)
    return fig