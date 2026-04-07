import torch
import numpy as np
import matplotlib.pyplot as plt  # ⬅️ NOUVEAU : Indispensable pour gérer les figures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path

from src.models.vae import VAE
from src.data.loader import get_dataloaders
from src.utils.visualize import plot_reconstructions, plot_latent_space, plot_generated_samples
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

    # 1. Qualité de Reconstruction
    print("\n🖼️ Génération des reconstructions...")
    x_test, _ = next(iter(test_loader))
    x_test = x_test.to(device)
    with torch.no_grad():
        recons, _, _ = model(x_test)
    
    fig_recons = plot_reconstructions(x_test, recons, num_samples=8)
    recons_path = logdir / "reconstructions.png"
    fig_recons.savefig(recons_path, bbox_inches='tight')
    plt.close(fig_recons)  # ⬅️ NOUVEAU : On ferme la figure pour vider la RAM
    print(f"📁 Reconstructions sauvegardées ici : {recons_path.resolve()}")
    
    # 2. Qualité de l'Espace Latent (Linear Probing)
    print("\n🔍 Extraction de l'espace latent...")
    X_train, y_train = extract_latents(model, train_loader, device)
    X_test, y_test = extract_latents(model, test_loader, device)
    
    acc = compute_linear_probing(X_train, y_train, X_test, y_test)
    print(f"✅ Accuracy du Linear Probing (Qualité Latente) : {acc * 100:.2f}%")
    
    # 3. Visualisation de l'Espace Latent (PCA)
    fig_latent = plot_latent_space(X_test, y_test, method="pca")
    latent_path = logdir / "latent_space_pca.png"
    fig_latent.savefig(latent_path, bbox_inches='tight')
    plt.close(fig_latent)  # ⬅️ NOUVEAU : On libère la RAM
    print(f"📁 Espace Latent sauvegardé ici : {latent_path.resolve()}")

    # 4. Génération pure (L'imagination du modèle)
    print("\n✨ Création de nouvelles images inédites...")
    
    # ÉTAPE A : On demande au modèle de générer les images (Tenseurs PyTorch)
    with torch.no_grad():
        raw_generated = model.sample(num_samples=16, device=device)
        # On les bascule sur le CPU et on les convertit en tableau NumPy
        generated_images_np = raw_generated.cpu().numpy()
        
    # ÉTAPE B : On donne ces images Numpy à ta nouvelle fonction d'affichage
    fig_generations = plot_generated_samples(
        generated_images=generated_images_np, 
        dataset_type="mnist"
    )
    
    generations_path = logdir / "generations.png"
    fig_generations.savefig(generations_path, bbox_inches='tight', dpi=300)
    print(f"📁 Nouvelles générations sauvegardées ici : {generations_path.resolve()}")
    
    plt.show() 
    plt.close(fig_generations)
    
    print("\n🎉 Évaluation terminée !")

if __name__ == "__main__":
    main()