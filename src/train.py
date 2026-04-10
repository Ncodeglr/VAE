import os
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Callable
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

# Imports locaux de notre projet
from .data.loader import get_dataloaders
from .models.vae import VAE
from .models.losses import get_vae_loss
from configs.vae_mnist_config import vae_cfg  # Ton fichier de configuration

class ModelCheckpoint:
    """Sauvegarde automatique du meilleur modèle et du dernier modèle."""
    def __init__(self, savepath: Path):
        self.savepath = savepath
        self.savepath.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def update(self, current_loss: float, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        # Toujours sauvegarder le dernier état pour la reprise
        last_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": current_loss,
        }
        torch.save(last_state, self.savepath / "last_model.pt")

        # Sauvegarder si c'est le meilleur
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save(last_state, self.savepath / "best_model.pt")
            return True
        return False

def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> dict:
    """Exécute une époque d'entraînement avec accumulation optimisée."""
    model.train()
    loss_fn.train()

    running_sums = defaultdict(float)
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # 1. Préparation des données (non-blocking pour accélérer le GPU)
        inputs, _ = batch
        inputs = inputs.to(device, non_blocking=True)
        bs = inputs.size(0)

        # 2. Forward & Backward
        optimizer.zero_grad()
        outputs = model(inputs) # outputs = (recons, mu, logvar) en mode Réel OU (recons, mu, var, delta, log_sigma2_dec) en mode Complexe
        
        # Le loss_fn renvoie un tuple (loss_scalaire, dict_metriques)
        loss_output = loss_fn(outputs, inputs)
        
        if isinstance(loss_output, tuple):
            loss, batch_metrics = loss_output
        else:
            loss = loss_output
            batch_metrics = {}

        loss.backward()
        
        # Gradient Clipping (optionnel mais recommandé pour VAE)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 3. Suivi optimisé des métriques
        current_loss_val = loss.item()
        pbar.set_postfix({"loss": f"{current_loss_val:.4f}"})

        # Accumulation sur CPU (évite la fuite de VRAM)
        with torch.no_grad():
            total_samples += bs
            running_sums["loss"] += current_loss_val * bs
            
            for k, v in batch_metrics.items():
                if torch.is_tensor(v):
                    running_sums[k] += v.detach().cpu() * bs
                else:
                    running_sums[k] += v * bs

    # 4. Finalisation des moyennes
    avg_metrics = {}
    for k, v in running_sums.items():
        val = v / total_samples
        avg_metrics[k] = val.item() if torch.is_tensor(val) else val

    return avg_metrics

def validate_one_epoch(
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device
) -> dict:
    """Exécute une époque de validation sans calculer les gradients."""
    model.eval()
    loss_fn.eval()

    running_sums = defaultdict(float)
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Validation")
        for batch in pbar:
            inputs, _ = batch
            inputs = inputs.to(device, non_blocking=True)
            bs = inputs.size(0)

            outputs = model(inputs)
            loss_output = loss_fn(outputs, inputs)
            
            if isinstance(loss_output, tuple):
                loss, batch_metrics = loss_output
            else:
                loss = loss_output
                batch_metrics = {}

            current_loss_val = loss.item()
            pbar.set_postfix({"loss": f"{current_loss_val:.4f}"})

            total_samples += bs
            running_sums["loss"] += current_loss_val * bs
            
            for k, v in batch_metrics.items():
                if torch.is_tensor(v):
                    running_sums[k] += v.cpu() * bs
                else:
                    running_sums[k] += v * bs

    avg_metrics = {}
    for k, v in running_sums.items():
        val = v / total_samples
        avg_metrics[k] = val.item() if torch.is_tensor(val) else val

    return avg_metrics

def main():
    """Script principal d'entraînement."""
    # 1. Configuration initiale
    cfg = vae_cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- MODIFICATION ICI : Ajout du Timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # ex: 20260403_140037
    logdir = Path("./runs") / cfg["project_name"] / f"run_{timestamp}"
    logdir.mkdir(parents=True, exist_ok=True) # On crée le dossier s'il n'existe pas
    # ---------------------------------------------
    
    print(f"🚀 Lancement sur {device}")
    print(f"📁 Sauvegarde dans : {logdir}")

    # 2. Données
    train_loader, valid_loader, test_loader = get_dataloaders(cfg)

    # 3. Modèle, Loss et Optimiseur
    model = VAE(cfg).to(device)
    loss_fn = get_vae_loss(cfg).to(device)
    
    # Séparation des paramètres pour l'optimiseur (comme dans cvnn)
    # On empêche le weight_decay sur la variance du décodeur si elle existe
    base_lr = cfg.get("optim", {}).get("lr", 1e-3)
    model_params = [p for n, p in model.named_parameters()]
    loss_params = list(loss_fn.parameters())
    
    groups = [{"params": model_params, "lr": base_lr, "weight_decay": 1e-5}]
    if loss_params:
        groups.append({"params": loss_params, "lr": base_lr, "weight_decay": 0.0})
        
    optimizer = torch.optim.AdamW(groups)

    # 4. Préparation de l'entraînement
    checkpoint = ModelCheckpoint(logdir)
    n_epochs = cfg.get("nepochs", 50)
    history = defaultdict(list)

    # 5. Boucle d'entraînement
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
        
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_metrics = validate_one_epoch(model, valid_loader, loss_fn, device)

        # Historique et affichage
        history["train_loss"].append(train_metrics["loss"])
        history["valid_loss"].append(valid_metrics["loss"])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | Valid Loss: {valid_metrics['loss']:.4f}")
        if "bpd" in valid_metrics:
            print(f"Valid BPD: {valid_metrics['bpd']:.4f} | Active Neurons: {valid_metrics['active_pct']*100:.1f}%")

        # Sauvegarde
        is_best = checkpoint.update(valid_metrics["loss"], epoch, model, optimizer)
        if is_best:
            print("Nouveau meilleur modèle sauvegardé !")

    print(f"\n✅ Entraînement terminé. Modèles sauvegardés dans {logdir}")

if __name__ == "__main__":
    main()