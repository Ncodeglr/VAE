"""
Custom loss functions for CVNN models.
This module provides custom loss implementations for both real and complex-valued
neural networks, including advanced losses like Focal Loss with class weighting.
"""

from typing import Optional, Union, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchcvnn.nn.modules as c_nn
except ImportError:
    c_nn = None

# ---- Constantes et Utilitaires Classification ----

LN2 = math.log(2.0)

def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    weight_mode: str = "inverse_frequency",
    ignore_index: int = -100,
) -> torch.Tensor:
    if device is None:
        device = targets.device

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        mask = targets != ignore_index
        valid = targets[mask]
    else:
        valid = targets

    counts = (
        torch.bincount(valid.flatten(), minlength=num_classes)
        .to(dtype=torch.float32)
        .clamp(min=1.0)
    )
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        counts[ignore_index] = float("inf")

    if weight_mode == "inverse_frequency":
        w = 1.0 / counts
    elif weight_mode == "balanced":
        N = valid.numel()
        C = num_classes - (1 if ignore_index is not None and 0 <= ignore_index < num_classes else 0)
        w = N / (C * counts)
    elif weight_mode == "log_frequency":
        w = torch.log1p(1.0 / counts)
    else:
        raise ValueError(f"Unknown weight_mode {weight_mode!r}")

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        w[ignore_index] = 0.0

    norm_C = num_classes - (1 if ignore_index is not None and 0 <= ignore_index < num_classes else 0)
    w = w * norm_C / w.sum()
    return w.to(dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        use_class_weights: bool = True,
        weight_mode: str = "balanced",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_class_weights = use_class_weights
        self.weight_mode = weight_mode

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N, C = inputs.shape[:2]
        if self.use_class_weights and self.weight is None:
            class_w = compute_class_weights(targets, C, device=inputs.device, weight_mode=self.weight_mode, ignore_index=self.ignore_index)
        else:
            class_w = self.weight

        logits = inputs.permute(0, *range(2, inputs.ndim), 1).reshape(-1, C)
        if logits.dtype != torch.float32 and logits.is_floating_point():
            logits = logits.to(dtype=torch.float32)

        t = targets.view(-1).long()
        valid = t != self.ignore_index

        logpt = F.log_softmax(logits, dim=1)
        pt = logpt.exp()

        logpt = logpt[valid, t[valid]]
        pt = pt[valid, t[valid]]

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[t[valid]]
        else:
            alpha_t = self.alpha

        focal = alpha_t * (1 - pt) ** self.gamma * (-logpt)

        if class_w is not None:
            if class_w.dtype != torch.float32 and class_w.is_floating_point():
                class_w = class_w.to(dtype=torch.float32)
            ce = F.cross_entropy(logits, t, weight=class_w, ignore_index=self.ignore_index, reduction="none")[valid]
            focal = alpha_t * (1 - pt) ** self.gamma * ce

        if self.reduction == "mean": return focal.mean()
        elif self.reduction == "sum": return focal.sum()
        else: return focal


# ---- KL helpers ----

def kl_diag(mu: torch.Tensor, sigma2: torch.Tensor, delta: torch.Tensor=None) -> torch.Tensor:
    """KL Divergence for Diagonal Gaussian (Real or Complex)."""
    sigma2 = sigma2.clamp_min(1e-12)
    
    if delta is None:
        log_sigma2 = torch.log(sigma2)
        kl = 0.5 * (sigma2 + mu.pow(2) - 1.0 - log_sigma2).sum(dim=1)
        return kl.mean()
    else:
        det_P = (sigma2**2 - delta.abs()**2).clamp_min(1e-12)
        log_det_P = torch.log(det_P)
        kl = (sigma2 + mu.abs().pow(2) - 1.0) - 0.5 * log_det_P
        return kl.sum(dim=1).mean()

def kl_full(mu: torch.Tensor, Sigma: torch.Tensor, Delta: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    """KL Divergence for Full Covariance (Real or Complex)."""
    if Delta is not None:
        B, D = mu.shape
        Re, Im = torch.real, torch.imag
        
        A = Sigma + Delta
        Bm = Sigma - Delta
        
        C11 = 0.5 * Re(A); C12 = -0.5 * Im(Bm)
        C21 = 0.5 * Im(A); C22 = 0.5 * Re(Bm)
        
        row1 = torch.cat([C11, C12], dim=-1)
        row2 = torch.cat([C21, C22], dim=-1)
        C = torch.cat([row1, row2], dim=-2) 

        I2 = torch.eye(2*D, device=C.device, dtype=C.dtype)
        C = C + (eps * I2)
        _, logdet = torch.linalg.slogdet(C)
        
        tr = torch.einsum('bii->b', C)
        m_real = torch.cat([mu.real, mu.imag], dim=-1)
        quad = (m_real**2).sum(dim=-1)

        kl = 0.5 * (tr + quad - logdet - 2*D)
        return kl.mean()
    else:
        L = Sigma
        B, D = mu.shape
        tr = (L.pow(2)).sum(dim=(1,2))
        diag = torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-12)
        logdet = 2.0 * torch.log(diag).sum(dim=1)
        quad = mu.pow(2).sum(dim=1)
        kl = 0.5 * (tr + quad - D - logdet)
        return kl.mean()
    
def vae_metrics(inputs, recons, mu, sigma_or_Sigma, delta_or_Delta, *, 
                cov_mode: str, kl_raw, kl_used,
                log_sigma2_dec: torch.Tensor, cap_pen=None, beta=None) -> dict:
    B = inputs.size(0)
    D_x = inputs[0].numel()
    D_z = mu.size(1)

    diff = (recons - inputs).abs().pow(2).view(B, -1).sum(1)
    err2 = diff.mean().detach() 
    
    sigma2_dec = torch.exp(log_sigma2_dec).detach() if log_sigma2_dec is not None else inputs.new_tensor(1.0, dtype=torch.float32)
    
    mse_per_elem = err2 / D_x
    ratio = mse_per_elem / sigma2_dec 
    
    Elogdet = inputs.new_tensor(0.0, dtype=torch.float32)
    circularity = inputs.new_tensor(0.0, dtype=torch.float32)
    Edelta2 = inputs.new_tensor(0.0, dtype=torch.float32)
    
    if cov_mode == "diag":
        sigma2 = sigma_or_Sigma.detach()
        delta = delta_or_Delta.detach() if delta_or_Delta is not None else None
        
        Esigma2 = sigma2.mean()
        
        if delta is not None:
            det = (sigma2**2 - delta.abs()**2).clamp_min(1e-12)
            Elogdet = torch.log(det).mean()
            Edelta2 = delta.abs().pow(2).mean()
            rho = delta.abs() / sigma2.clamp_min(1e-9)
            circularity = rho.mean()
        else:
            Elogdet = torch.log(sigma2.clamp_min(1e-12)).mean()
            
    else:
        Sigma = sigma_or_Sigma.detach()
        Delta = delta_or_Delta.detach() if delta_or_Delta is not None else None
        
        if Delta is not None:
            diag_sigma = torch.diagonal(Sigma.real, dim1=-2, dim2=-1)
            Esigma2 = diag_sigma.mean()
            Edelta2 = (Delta.abs().pow(2)).sum(dim=(1,2)).div(D_z**2).mean()
        else:
            L = Sigma
            Esigma2 = (L.pow(2)).sum(dim=(1,2)).div(D_z**2).mean()
            diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
            logdet_L = 2.0 * torch.log(diag_L.clamp_min(1e-12)).sum(dim=1)
            Elogdet = logdet_L.mean()

    Emu2 = mu.abs().pow(2).mean().detach()

    if mu.is_complex():
        mu_var = mu.real.detach().var(dim=0) + mu.imag.detach().var(dim=0)
    else:
        mu_var = mu.detach().var(dim=0)
    
    active_dims_count = (mu_var > 0.01).sum().float()
    percent_active = active_dims_count / D_z
    
    kl_total_raw = kl_raw.detach() if torch.is_tensor(kl_raw) else inputs.new_tensor(kl_raw, dtype=torch.float32)
    kl_used_t = kl_used.detach() if torch.is_tensor(kl_used) else inputs.new_tensor(kl_used, dtype=torch.float32)
    
    kl_per_dim_nats = kl_total_raw / D_z
    bpd_raw = kl_per_dim_nats / math.log(2.0)

    m = {
        'kl_total': kl_total_raw, 
        'kl_used': kl_used_t,
        'kl_per_dim': kl_per_dim_nats, 
        'bpd': bpd_raw,
        'mse_elem': mse_per_elem, 
        'ratio_mse_var': ratio,
        'E_mu2': Emu2, 
        'E_sigma2': Esigma2, 
        'E_delta2': Edelta2,
        'E_logdet': Elogdet,
        'circularity': circularity,
        'active_pct': percent_active
    }
    
    if cap_pen is not None: 
        m['cap_pen'] = cap_pen.detach() if torch.is_tensor(cap_pen) else inputs.new_tensor(cap_pen, dtype=torch.float32)
    if beta is not None: 
        m['beta'] = beta.detach() if torch.is_tensor(beta) else inputs.new_tensor(beta, dtype=torch.float32)
        
    return m

# ---- ELBO Loss ----

class ELBOLoss(nn.Module):
    def __init__(self,
                 schedule: str = "beta",
                 gamma: float = 50.0,
                 freebits_bpd: float = 0.10,
                 cov_mode: str = "diag",
                 beta_max: float = 1.0,
                 bpd_target: float = 0.20,
                 learned_variance: bool = False,
                 min_log_sigma: Optional[float] = None):
        super().__init__()
        if schedule is not None:
            assert schedule in {"capacity", "freebits", "beta"}
        assert cov_mode in {"diag","full"}
        self.schedule = schedule
        self.gamma = gamma
        self.freebits_nats = freebits_bpd * LN2
        self.cov_mode = cov_mode
        self.bpd_target = bpd_target
        self.beta_max = beta_max
        self.learned_variance = learned_variance
        self.min_log_sigma = min_log_sigma

    def _recon_nll(self, inputs, recons, log_sigma2_dec):
        B = inputs.size(0)
        if self.learned_variance and self.min_log_sigma is not None:
            log_sigma2_dec = torch.clamp(log_sigma2_dec, min=self.min_log_sigma)  
        err2 = (recons - inputs).abs().pow(2).view(B, -1).sum(1)
        
        sigma2_dec = torch.exp(log_sigma2_dec)
        D_x = inputs[0].numel()
        nll = 0.5 * (err2 / sigma2_dec + D_x * (math.log(2 * math.pi) + log_sigma2_dec)).mean()
        return nll

    def _kl_raw(self, mu, var_or_L):
        if self.cov_mode == "diag":
            sigma2 = var_or_L.clamp_min(1e-12)
            return kl_diag(mu, sigma2), sigma2
        else:
            L = var_or_L
            return kl_full(mu, L), L

    def forward(self, outputs, inputs, anneal=None):
        # outputs expects: [recons, mu, var, _, log_sigma2_dec] 
        # Adaptations to match your current simple VAE if it only returns [recons, mu, logvar]
        if len(outputs) == 3:
            recons, mu, logvar = outputs
            var_or_L = torch.exp(logvar) # convert logvar to var since cvnn expects variance
            log_sigma2_dec = torch.zeros_like(logvar[:, 0].mean()) # Default dec variance
        else:
            recons, mu, var_or_L, _, log_sigma2_dec = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

        beta_t = 1.0
        C_t = 0.0
        cap_pen = 0.0
        
        if anneal is not None:
            if self.schedule == "capacity":
                D_z = mu.size(1)
                C_max = self.bpd_target * D_z * math.log(2.0)
                C_t = C_max * anneal.progress()
            elif self.schedule == "beta":
                beta_t = self.beta_max * anneal.progress()
            
            if self.training: 
                anneal.step()        
        
        recon = self._recon_nll(inputs, recons, log_sigma2_dec)
        kl_raw, sigma2_or_L = self._kl_raw(mu, var_or_L)

        if self.schedule == "capacity":
            assert C_t is not None, "Provide C_t for capacity schedule."
            dev = kl_raw - C_t
            cap_pen = torch.abs(dev)
            loss = recon + self.gamma * cap_pen
            kl_used = kl_raw
            
        elif self.schedule == "freebits":
            assert self.cov_mode == "diag", "Freebits only supported for diagonal covariance."
            sigma2 = sigma2_or_L
            log_sigma2 = torch.log(sigma2.clamp_min(1e-12))
            kl_elem = 0.5 * (sigma2 + mu.pow(2) - 1.0 - log_sigma2)
            kl_fb = (kl_elem - self.freebits_nats).clamp_min(0.0).sum(1).mean()
            loss = recon + kl_fb
            kl_used = kl_fb
            
        else:  # "beta"
            assert beta_t is not None, "Provide beta_t for beta schedule."
            loss = recon + beta_t * kl_raw
            kl_used = kl_raw
    
        metrics = vae_metrics(inputs, recons, mu, sigma2_or_L, delta_or_Delta=None,
                              cov_mode=self.cov_mode, kl_raw=kl_raw, kl_used=kl_used,
                              log_sigma2_dec=log_sigma2_dec, beta=beta_t, cap_pen=cap_pen)
        return loss, metrics
        
class ComplexELBOLoss(nn.Module):
    def __init__(self,
                 schedule: str = "beta",
                 gamma: float = 50.0,
                 freebits_bpd: float = 0.10,
                 cov_mode: str = "diag",
                 beta_max: float = 1.0,
                 bpd_target: float = 0.20,
                 standard_reparam: bool = True,
                 learned_variance: bool = False,
                 min_log_sigma = None):
        super().__init__()
        if schedule is not None:
            assert schedule in {"capacity", "freebits", "beta"}
        assert cov_mode in {"diag", "full"}
        
        self.schedule = schedule
        self.gamma = gamma
        self.freebits_nats = freebits_bpd * LN2
        self.cov_mode = cov_mode
        self.bpd_target = bpd_target
        self.beta_max = beta_max
        self.standard_reparam = standard_reparam
        self.learned_variance = learned_variance
        self.min_log_sigma = min_log_sigma

    def _recon_nll(self, inputs, recons, log_sigma2_dec):
        B = inputs.size(0)
        
        if self.learned_variance and self.min_log_sigma is not None:
            log_sigma2_dec = torch.clamp(log_sigma2_dec, min=self.min_log_sigma)

        err2 = (recons - inputs).abs().pow(2).view(B, -1).sum(1)
        sigma2_dec = torch.exp(log_sigma2_dec)
        D_x = inputs[0].numel()
        nll = 0.5 * (err2 / sigma2_dec + D_x * (math.log(2 * math.pi) + log_sigma2_dec)).mean()
        return nll

    def _compute_params(self, W, V):
        if self.cov_mode == "diag":
            sigma2 = W.abs().pow(2) + V.abs().pow(2)
            delta = 2.0 * W * V
            return sigma2, delta
        else:
            Wt = W.transpose(-1, -2)
            Vt = V.transpose(-1, -2)
            Wh = W.conj().transpose(-1, -2)
            Vh = V.conj().transpose(-1, -2)
            Sigma = (W @ Wh) + (V @ Vh)
            Delta = (W @ Vt) + (V @ Wt)
            return Sigma, Delta

    def forward(self, outputs, inputs, anneal=None):
        recons, mu, p1, p2, log_sigma2_dec = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

        beta_t = 1.0
        C_t = 0.0
        cap_pen = 0.0

        if anneal is not None:
            if self.schedule == "capacity":
                D_z = mu.size(1)
                C_max = self.bpd_target * D_z * math.log(2.0)
                C_t = C_max * anneal.progress()
            elif self.schedule == "beta":
                beta_t = self.beta_max * anneal.progress()
            if self.training: 
                anneal.step()        

        recon_loss = self._recon_nll(inputs, recons, log_sigma2_dec)

        if self.standard_reparam:
            sigma_or_Sigma = p1
            delta_or_Delta = p2
        else:
            sigma_or_Sigma, delta_or_Delta = self._compute_params(p1, p2)

        if self.cov_mode == "diag":
            kl_raw = kl_diag(mu, sigma_or_Sigma, delta_or_Delta)
        else:
            kl_raw = kl_full(mu, sigma_or_Sigma, delta_or_Delta)

        if self.schedule == "capacity":
            cap_pen = torch.abs(kl_raw - C_t)
            loss = recon_loss + self.gamma * cap_pen
            kl_used = kl_raw
        elif self.schedule == "freebits":
            sigma2, delta = sigma_or_Sigma, delta_or_Delta
            det_P = (sigma2**2 - delta.abs()**2).clamp_min(1e-12)
            kl_elem = (sigma2 + mu.abs().pow(2) - 1.0) - 0.5 * torch.log(det_P)
            kl_fb = (kl_elem - self.freebits_nats).clamp_min(0.0).sum(1).mean()
            loss = recon_loss + kl_fb
            kl_used = kl_fb
        else:
            beta = beta_t if beta_t is not None else None
            loss = recon_loss + beta * kl_raw
            kl_used = kl_raw

        metrics = vae_metrics(inputs, recons, mu, sigma_or_Sigma, delta_or_Delta,
                              cov_mode=self.cov_mode,
                              kl_raw=kl_raw, kl_used=kl_used,
                              log_sigma2_dec=log_sigma2_dec,
                              cap_pen=cap_pen, beta=beta_t)
                         
        return loss, metrics

# ---- Aiguilleur pour le Dictionnaire de Config ----

def get_vae_loss(cfg: Dict[str, Any]) -> nn.Module:
    """
    Retourne la fonction de perte ELBO adaptée (Réelle ou Complexe) 
    en lisant la configuration du modèle.
    """
    layer_mode = cfg["data"].get("layer_mode", "real")
    
    # Récupération des paramètres de Loss (avec valeurs par défaut de cvnn)
    schedule = cfg["model"].get("loss_schedule", "beta")
    cov_mode = cfg["model"].get("cov_mode", "diag")
    beta_max = cfg["model"].get("beta", 1.0)
    
    if layer_mode in ["complex", "split"]:
        return ComplexELBOLoss(
            schedule=schedule,
            cov_mode=cov_mode,
            beta_max=beta_max,
            standard_reparam=False  # <--- MODIFICATION ICI : On utilise le mode "W et V"
        )
    else:
        return ELBOLoss(
            schedule=schedule,
            cov_mode=cov_mode,
            beta_max=beta_max,
        )