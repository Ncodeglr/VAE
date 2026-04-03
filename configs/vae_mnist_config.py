"""
Fichier de configuration principal pour le VAE sur MNIST.
Toutes les hyperparamètres de l'architecture, des données et de l'entraînement 
sont centralisés ici.
"""

vae_cfg = {
    # --- Infos Projet ---
    "project_name": "VAE_MNIST_Modular",
    "nepochs": 10,  # Nombre d'époques pour l'entraînement
    
    # --- Données ---
    "data": {
        "dataset_name": "MNIST",
        "batch_size": 64,
        "num_workers": 2,         # Nombre de cœurs CPU pour charger les images
        "data_path": "./data",    # Dossier où MNIST sera téléchargé
        "valid_ratio": 0.1,       # 10% des données réservées pour la validation
        "layer_mode": "real",     # "real" (MNIST) ou "complex" (Radar)
    },
    
    # --- Architecture du Modèle ---
    "model": {
        # Géométrie : Le nombre d'étapes définit le nombre de down/up sampling.
        # Ici : Entrée(1) -> 32 -> 64 -> 128. (Divise l'image par 2 à chaque étape)
        "channels": [32, 64, 128], 
        
        # Le "goulot d'étranglement" (la taille du résumé Z)
        "latent_dim": 32,          
        
        # Le comportement des blocs (tes briques modulaires)
        "activation": "relu",      # Options: "relu", "leaky_relu", "crelu"...
        "normalization": "batch",  # Options: "batch", "ln" (LayerNorm), "none"
        "downsampling": "strided", # Options: "strided", "max", "avg"
        "upsampling": "transpose", # Options: "transpose", "nearest"
        
        # Paramètres mathématiques du VAE (lus par get_vae_loss)
        "loss_schedule": "beta",   # "beta", "capacity" ou "freebits"
        "cov_mode": "diag",        # "diag" (Diagonale standard) ou "full"
        "beta": 1.0,               # Beta-VAE : Poids de la KL Divergence (1.0 = standard)
        
        # Paramètres avancés de cvnn pour la reconstruction
        "decoder_variance": {
            "learned_variance": False,
            "min_log_sigma": -10.0
        }
    },
    
    # --- Optimiseur (Descente de gradient) ---
    "optim": {
        "algo": "AdamW",           # Adam avec un meilleur Weight Decay
        "lr": 1e-3,                # Learning rate (taux d'apprentissage)
        "params": {
            "weight_decay": 1e-5   # Régularisation L2
        }
    }
}