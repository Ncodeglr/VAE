"""
Fichier de configuration principal pour le VAE sur MNIST.
Toutes les hyperparamètres de l'architecture, des données et de l'entraînement 
sont centralisés ici.
"""

vae_cfg = {
    # --- Infos Projet ---
    "project_name": "VAE_MNIST_Modular",
    "nepochs": 30,  # Nombre d'époques pour l'entraînement
    
    # --- Données ---
    "data": {
        "dataset_name": "COMPLEXMNIST",
        "batch_size": 64,
        "num_workers": 2,         # Nombre de cœurs CPU pour charger les images
        "data_path": "./data",    # Dossier où MNIST sera téléchargé
        "valid_ratio": 0.1,       # 10% des données réservées pour la validation
        "layer_mode": "complex",     # "real" ou "complex" 
    },
    
    # --- Architecture du Modèle ---
    "model": {
        # Géométrie : Le nombre d'étapes définit le nombre de down/up sampling.
        "channels": [32, 64, 128], # Ici : Entrée(1) -> 32 -> 64 -> 128. (Divise l'image par 2 à chaque étape)
        
        "latent_dim": 32, # Le "goulot d'étranglement" (la taille du résumé Z)

        "widely_linear": False,  # True = WL-CVAE (Ellipse), False = CVAE standard (Cercle)      
        
        "weight_init": "xavier_uniform",  # Options: "xavier_normal", "xavier_uniform", "kaiming_normal", "kaiming_uniform", "rayleigh"
        
        #---------------------------------Le comportement des blocs (les Briques Modulaires pour chaque layer)------------------------------------------#
        # --- ENCODEUR ---
        "encoder_activation": ["crelu", "crelu", "crelu"],  
        "encoder_normalization": ["batch", "batch", "none"],  
        "encoder_downsampling": ["strided", "strided"], 
        
        # --- DÉCODEUR ---
        "decoder_activation": ["crelu", "crelu", "crelu"], 
        "decoder_normalization": ["batch", "batch", "none"],
        "decoder_upsampling": ["transpose", "transpose"],
        #-----------------------------------------------------------------------------------------------------------------------------#
        
        "residual": False,  # Active les connexions résiduelles dans les DoubleConv
        
        
        # Paramètres mathématiques du VAE (lus par get_vae_loss)
        "loss_schedule": "beta",   # "beta", "capacity" ou "freebits"
        "cov_mode": "diag",        # "diag" (Diagonale standard) ou "full"
        "beta": 1,               # Beta-VAE : Poids de la KL Divergence (1.0 = standard)
        
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