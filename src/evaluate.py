import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path

from src.models.vae import VAE
from src.data.loader import get_dataloaders
from src.utils.visualize import (
    plot_reconstructions, 
    plot_latent_space, 
    plot_generated_samples,
    plot_complex_reconstructions # <-- Le nouvel affichage Amplitude/Phase !
)
from configs.vae_mnist_config import vae_cfg

def extract_latents(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    """Extrait les vecteurs latents 'mu' et les labels associés."""
    latents_list, labels_list = [], []
    model.eval()
    
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            
            # Passage dans l'encodeur uniquement
            mu, _ = model.encoder(x)
            
            # --- Traitement des latents complexes ---
            if mu.is_complex():
                # On sépare la partie réelle et imaginaire
                mu_real = mu.real
                mu_imag = mu.imag
                # On les concatène côte à côte (ex: un vecteur de 32 devient 64)
                mu_processed = torch.cat([mu_real, mu_imag], dim=1)
                latents_list.append(mu_processed.cpu().numpy())
            else:
                latents_list.append(mu.cpu().numpy())
                
            labels_list.append(y.cpu().numpy())
            
    X = np.concatenate(latents_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y

def compute_linear_probing(X_train, y_train, X_test, y_test) -> float:
    """Entraîne un modèle linéaire simple sur l'espace latent pour évaluer sa qualité."""
    print("🧠 Lancement du Linear Probing...")
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def main():
    cfg = vae_cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["data"]["dataset_name"]
    layer_mode = cfg["data"].get("layer_mode", "real")
    
    # --- Trouver le dernier run ---
    base_dir = Path("./runs") / cfg["project_name"]
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    if not runs:
        print("❌ Aucun entraînement trouvé ! Lance train.py d'abord.")
        return
        
    logdir = runs[-1] # On prend le dernier
    print(f"📂 Évaluation du dossier le plus récent : {logdir}")

    print("📥 Chargement des données...")
    train_loader, valid_loader, test_loader = get_dataloaders(cfg)
    
    print("🤖 Chargement du modèle...")
    model = VAE(cfg).to(device)
    
    # Charger les poids du meilleur modèle
    model_path = logdir / "best_model.pt"
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    else:
        print("⚠️ Aucun modèle entraîné trouvé. L'évaluation se fera sur des poids aléatoires.")

    # ---------------------------------------------------------
    # 1. Qualité de Reconstruction
    # ---------------------------------------------------------
    print("\n🖼️ Génération des reconstructions spatiales...")
    x_test, _ = next(iter(test_loader))
    x_test = x_test.to(device)
    with torch.no_grad():
        recons = model(x_test)[0]
    
    fig_recons = plot_reconstructions(x_test, recons, dataset_type=dataset_name, num_samples=8)
    recons_path = logdir / "reconstructions.png"
    fig_recons.savefig(recons_path, bbox_inches='tight', dpi=150)
    plt.close(fig_recons) 
    print(f"📁 Reconstructions sauvegardées ici : {recons_path.resolve()}")
    
    # --- NOUVEAU : Affichage Expert si le modèle est complexe ---
    if layer_mode == "complex":
        print("🌌 Génération du tableau de bord Complexe (Amplitude & Phase)...")
        fig_complex = plot_complex_reconstructions(x_test, recons, num_samples=8)
        complex_path = logdir / "reconstructions_complexes.png"
        fig_complex.savefig(complex_path, bbox_inches='tight', dpi=150)
        plt.close(fig_complex)
        print(f"📁 Tableau de bord Complexe sauvegardé ici : {complex_path.resolve()}")

    # ---------------------------------------------------------
    # 2. Qualité de l'Espace Latent (Linear Probing)
    # ---------------------------------------------------------
    print("\n🔍 Extraction de l'espace latent...")
    X_train, y_train = extract_latents(model, train_loader, device)
    X_test, y_test = extract_latents(model, test_loader, device)
    
    acc = compute_linear_probing(X_train, y_train, X_test, y_test)
    print(f"✅ Accuracy du Linear Probing (Qualité Latente) : {acc * 100:.2f}%")
    
    # ---------------------------------------------------------
    # 3. Visualisation de l'Espace Latent (PCA)
    # ---------------------------------------------------------
    fig_latent = plot_latent_space(X_test, y_test, method="pca")
    latent_path = logdir / "latent_space_pca.png"
    fig_latent.savefig(latent_path, bbox_inches='tight', dpi=150)
    plt.close(fig_latent) 
    print(f"📁 Espace Latent sauvegardé ici : {latent_path.resolve()}")

# ---------------------------------------------------------
    # 4. Génération pure (L'imagination du modèle)
    # ---------------------------------------------------------
    print("\n✨ Création de nouvelles images inédites...")
    
    with torch.no_grad():
        # L'ajout est ici : on génère sur GPU, puis on ramène de force sur CPU !
        generated_tensors = model.sample(num_samples=16, device=device).cpu()
        
    fig_generations = plot_generated_samples(
        generated_images=generated_tensors, 
        dataset_type=dataset_name
    )
    
    generations_path = logdir / "generations.png"
    fig_generations.savefig(generations_path, bbox_inches='tight', dpi=300)
    print(f"📁 Nouvelles générations sauvegardées ici : {generations_path.resolve()}")
    
    # Affiche la dernière figure sur ton écran si tu le souhaites
    plt.show() 
    plt.close(fig_generations)
    
    print("\n🎉 Évaluation terminée avec succès !")

if __name__ == "__main__":
    main()