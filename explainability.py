# =============================================================================
# src/explainability.py  —  Grad-CAM + SHAP + ODE Trajectory + Risk Charts
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from config import DEVICE, SEQ_LEN, CLASS_NAMES, RESULTS_DIR

# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for 1D ECG signals.
    Hooks into the BiLSTM hidden states to compute importance weights.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # output = (output, (h_n, c_n)) for LSTM
            if isinstance(output, tuple):
                self.activations = output[0].detach()  # [B, T, 2H]
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()

        self.model.lstm.register_forward_hook(forward_hook)
        self.model.lstm.register_full_backward_hook(backward_hook)

    def compute(self, x, t_span, target_class=None):
        """
        Compute Grad-CAM saliency map for input x.
        Returns: saliency [SEQ_LEN] numpy array
        """
        self.model.eval()
        x = x.to(DEVICE).requires_grad_(False)

        # Forward pass
        logits, mu, lv = self.model(x, t_span)

        if target_class is None:
            target_class = logits.argmax(1).item()

        # Backward pass on target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros(SEQ_LEN)

        # Global average pooling of gradients over time
        weights     = self.gradients.mean(dim=2, keepdim=True)  # [B, T, 1]
        cam         = (weights * self.activations).sum(dim=2)    # [B, T]
        cam         = torch.relu(cam)[0].cpu().numpy()           # [T]

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # Resize to SEQ_LEN if needed
        if len(cam) != SEQ_LEN:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(cam))
            x_new = np.linspace(0, 1, SEQ_LEN)
            cam   = interp1d(x_old, cam)(x_new)

        return cam.astype(np.float32)


def plot_gradcam(model, test_ds, n_per_class=1, save=True):
    """
    Plot Grad-CAM heatmaps overlaid on ECG signals — one per class.
    """
    t_span  = torch.linspace(0, 1, SEQ_LEN).to(DEVICE)
    gradcam = GradCAM(model)
    model.eval()

    # Collect one sample per class
    class_samples = defaultdict(list)
    for i in range(len(test_ds)):
        x, y = test_ds[i]
        lbl  = y.item()
        if len(class_samples[lbl]) < n_per_class:
            class_samples[lbl].append((x, lbl))
        if all(len(v) >= n_per_class for v in class_samples.values()):
            break

    n_classes = len(class_samples)
    if n_classes == 0:
        print('  No samples for Grad-CAM.')
        return

    fig, axes = plt.subplots(n_classes, 1,
                             figsize=(14, 3 * n_classes),
                             squeeze=False)
    fig.suptitle('Grad-CAM: ECG Regions Most Important for Classification',
                 fontsize=13, fontweight='bold', y=1.01)

    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    for row, (cls, samples) in enumerate(sorted(class_samples.items())):
        x, true_lbl = samples[0]
        xb          = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, _, _ = model(xb, t_span)
            pred_lbl     = logits.argmax(1).item()

        cam     = gradcam.compute(xb, t_span, target_class=pred_lbl)
        ecg_sig = x[:, 0].numpy()   # ECG channel
        time    = np.linspace(0, SEQ_LEN / 360, SEQ_LEN)

        ax = axes[row][0]

        # Plot ECG
        ax.plot(time, ecg_sig, color='#455A64', lw=1.0,
                alpha=0.8, label='ECG signal', zorder=3)

        # Overlay heatmap as filled background
        ax.fill_between(time, ecg_sig.min() - 0.3, ecg_sig.max() + 0.3,
                        alpha=cam * 0.5,
                        color=colors[cls % len(colors)],
                        label='Grad-CAM importance', zorder=2)

        correct = '✓' if pred_lbl == true_lbl else '✗'
        ax.set_title(
            f'True: {CLASS_NAMES[true_lbl]}  |  '
            f'Pred: {CLASS_NAMES[pred_lbl]} {correct}',
            fontsize=9, pad=4)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.legend(loc='upper right', fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / 'gradcam_results.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f'  Grad-CAM saved: {path}')
    plt.close()


# ── SHAP ──────────────────────────────────────────────────────────────────────

def plot_shap(model, test_ds, n_background=30, n_samples=5, save=True):
    """
    SHAP GradientExplainer attribution maps for ECG channel.
    """
    try:
        import shap
    except ImportError:
        print('  SHAP not installed. Run: pip install shap')
        return

    t_span = torch.linspace(0, 1, SEQ_LEN).to(DEVICE)
    model.eval()

    # Collect background samples
    background = []
    for i in range(min(n_background, len(test_ds))):
        x, _ = test_ds[i]
        background.append(x.numpy())
    background = np.stack(background)   # [N, SEQ_LEN, 3]
    bg_tensor  = torch.tensor(background, dtype=torch.float32).to(DEVICE)

    # Wrapper that only takes x (SHAP needs single input)
    class ModelWrapper(nn.Module):
        def __init__(self, model, t_span):
            super().__init__()
            self.model  = model
            self.t_span = t_span

        def forward(self, x):
            logits, _, _ = self.model(x, self.t_span)
            return logits

    wrapper = ModelWrapper(model, t_span)

    try:
        explainer = shap.GradientExplainer(wrapper, bg_tensor)
    except Exception as e:
        print(f'  SHAP explainer init failed: {e}')
        return

    # Collect test samples
    test_samples = []
    test_labels  = []
    for i in range(min(n_samples, len(test_ds))):
        x, y = test_ds[i]
        test_samples.append(x.numpy())
        test_labels.append(y.item())

    test_tensor = torch.tensor(
        np.stack(test_samples), dtype=torch.float32).to(DEVICE)

    try:
        shap_values = explainer.shap_values(test_tensor)
        # shap_values: list of [N, SEQ_LEN, 3] per class
    except Exception as e:
        print(f'  SHAP computation failed: {e}')
        return

    # Plot ECG channel SHAP values
    n   = len(test_samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), squeeze=False)
    fig.suptitle('SHAP Attribution Maps — ECG Channel Importance',
                 fontsize=13, fontweight='bold', y=1.01)

    time = np.linspace(0, SEQ_LEN / 360, SEQ_LEN)

    for i in range(n):
        ax       = axes[i][0]
        ecg_sig  = test_samples[i][:, 0]
        true_lbl = test_labels[i]

        with torch.no_grad():
            logits, _, _ = model(
                torch.tensor(test_samples[i]).unsqueeze(0).to(DEVICE),
                t_span)
            pred_lbl = logits.argmax(1).item()

        # SHAP for predicted class, ECG channel
        if isinstance(shap_values, list):
            sv = shap_values[pred_lbl][i, :, 0]
        else:
            sv = shap_values[i, :, 0]

        # Normalise SHAP — flatten to 1D
        sv_norm   = np.array(sv).flatten()[:SEQ_LEN]
        sv_norm   = sv_norm / (np.abs(sv_norm).max() + 1e-8)
        time_shap = np.linspace(0, SEQ_LEN / 360, len(sv_norm))
        ecg_plot  = np.array(ecg_sig).flatten()[:len(sv_norm)]

        pos_mask = sv_norm > 0
        neg_mask = sv_norm < 0
        zeros    = np.zeros_like(sv_norm)

        ax.plot(time_shap, ecg_plot, color='#37474F', lw=0.8,
                alpha=0.9, zorder=3, label='ECG')
        ax.fill_between(time_shap, zeros, sv_norm,
                        where=pos_mask, color='#E53935', alpha=0.6,
                        label='Positive (pushes toward class)', zorder=2)
        ax.fill_between(time_shap, zeros, sv_norm,
                        where=neg_mask, color='#1E88E5', alpha=0.6,
                        label='Negative (pushes away)', zorder=2)
        ax.axhline(0, color='gray', lw=0.5, linestyle='--')

        correct = '✓' if pred_lbl == true_lbl else '✗'
        ax.set_title(
            f'True: {CLASS_NAMES[true_lbl]}  |  '
            f'Pred: {CLASS_NAMES[pred_lbl]} {correct}',
            fontsize=9, pad=4)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('SHAP value', fontsize=8)
        ax.legend(loc='upper right', fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / 'shap_results.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f'  SHAP saved: {path}')
    plt.close()


# ── ODE Trajectory ────────────────────────────────────────────────────────────

def plot_ode_trajectories(model, test_ds, n_per_class=1, save=True):
    """
    Visualise latent ODE trajectories for one sample per class.
    Shows the first 3 latent dimensions over time.
    """
    t_span = torch.linspace(0, 1, SEQ_LEN).to(DEVICE)
    model.eval()

    class_samples = defaultdict(list)
    for i in range(len(test_ds)):
        x, y = test_ds[i]
        if len(class_samples[y.item()]) < n_per_class:
            class_samples[y.item()].append((x, y.item()))
        if all(len(v) >= n_per_class for v in class_samples.values()):
            break

    from torchdiffeq import odeint
    colors  = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    n_cls   = len(class_samples)
    fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 4), squeeze=False)
    fig.suptitle('Latent ODE Trajectories (First 3 Dimensions)',
                 fontsize=12, fontweight='bold')

    for col, (cls, samples) in enumerate(sorted(class_samples.items())):
        x, lbl = samples[0]
        xb     = x.unsqueeze(0).to(DEVICE)
        ax     = axes[0][col]

        with torch.no_grad():
            mu, lv = model.encoder(xb)
            z0     = model.encoder.reparam(mu, lv)
            z_traj = odeint(model.ode_func, z0, t_span,
                            method='euler',
                            options={'step_size': 0.05})
            # z_traj: [T, B, latent_dim]
            traj = z_traj[:, 0, :3].cpu().numpy()  # [T, 3]

        time = np.linspace(0, 1, SEQ_LEN)
        dim_labels = ['z₁', 'z₂', 'z₃']
        dim_colors = ['#E53935', '#1E88E5', '#43A047']

        for d in range(3):
            ax.plot(time, traj[:, d],
                    color=dim_colors[d], lw=1.2,
                    label=dim_labels[d], alpha=0.85)

        ax.set_title(CLASS_NAMES[lbl], fontsize=8, fontweight='bold',
                     color=colors[cls % len(colors)])
        ax.set_xlabel('Normalised time', fontsize=7)
        ax.set_ylabel('Latent value', fontsize=7)
        ax.legend(fontsize=7, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / 'ode_trajectories.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f'  ODE trajectories saved: {path}')
    plt.close()


# ── Risk Score Bar Chart ───────────────────────────────────────────────────────

def plot_risk_scores(model, test_ds, scorer, n_samples=10, save=True):
    """
    Bar chart of stroke, heart failure, and arrhythmia risk scores
    for n_samples test patients.
    """
    from src.risk_scorer import ClinicalRiskScorer
    t_span = torch.linspace(0, 1, SEQ_LEN).to(DEVICE)
    model.eval()

    patient_ids   = []
    stroke_risks  = []
    hf_risks      = []
    arr_risks     = []
    true_classes  = []
    pred_classes  = []

    for i in range(min(n_samples, len(test_ds))):
        x, y = test_ds[i]
        with torch.no_grad():
            logits, _, _ = model(x.unsqueeze(0).to(DEVICE), t_span)
            probs = torch.softmax(logits, 1).cpu().numpy()[0]

        risks = scorer.score(probs)
        patient_ids.append(f'P{i+1:02d}')
        stroke_risks.append(risks['Stroke risk %'])
        hf_risks.append(risks['Heart failure risk %'])
        arr_risks.append(risks['Arrhythmia risk %'])
        true_classes.append(y.item())
        pred_classes.append(probs.argmax())

    x_pos = np.arange(len(patient_ids))
    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle('Clinical Risk Scoring — Patient-Level Results',
                 fontsize=13, fontweight='bold')

    # ── Top: grouped bar chart ─────────────────────────────────────────────────
    ax1 = axes[0]
    b1  = ax1.bar(x_pos - width, stroke_risks, width,
                  label='Stroke risk %', color='#E53935', alpha=0.85)
    b2  = ax1.bar(x_pos,          hf_risks,     width,
                  label='Heart failure risk %', color='#FB8C00', alpha=0.85)
    b3  = ax1.bar(x_pos + width,  arr_risks,    width,
                  label='Arrhythmia risk %', color='#8E24AA', alpha=0.85)

    ax1.axhline(30, color='gray',    lw=1, linestyle='--', alpha=0.5,
                label='LOW/MOD threshold (30%)')
    ax1.axhline(60, color='#B71C1C', lw=1, linestyle='--', alpha=0.5,
                label='MOD/HIGH threshold (60%)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(patient_ids, fontsize=8)
    ax1.set_ylabel('Risk (%)', fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_title('Risk Scores per Patient', fontsize=10)
    ax1.spines[['top', 'right']].set_visible(False)

    # Add value labels on bars
    for bar in [b1, b2, b3]:
        for rect in bar:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.5,
                     f'{h:.0f}', ha='center', va='bottom', fontsize=6)

    # ── Bottom: true vs predicted class ───────────────────────────────────────
    ax2     = axes[1]
    cls_map = {i: c[:12] for i, c in enumerate(CLASS_NAMES)}
    colors_true = ['#455A64'] * len(patient_ids)
    colors_pred = []
    for t, p in zip(true_classes, pred_classes):
        colors_pred.append('#2E7D32' if t == p else '#C62828')

    ax2.bar(x_pos - 0.2, true_classes, 0.35,
            color=colors_true, alpha=0.7, label='True class')
    ax2.bar(x_pos + 0.2, pred_classes, 0.35,
            color=colors_pred, alpha=0.7, label='Pred class (green=✓, red=✗)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(patient_ids, fontsize=8)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([c[:18] for c in CLASS_NAMES], fontsize=7)
    ax2.set_title('True vs Predicted Classes', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / 'risk_scores_chart.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f'  Risk scores chart saved: {path}')
    plt.close()


# ── Master explainability runner ──────────────────────────────────────────────

def run_all_explainability(model, test_ds, scorer):
    """Run all explainability analyses and save all plots."""
    print('\n=== Explainability Analysis ===')

    print('\n[XAI 1/4] Computing Grad-CAM heatmaps...')
    try:
        plot_gradcam(model, test_ds)
    except Exception as e:
        print(f'  Grad-CAM error: {e}')

    print('\n[XAI 2/4] Computing SHAP attribution maps...')
    try:
        plot_shap(model, test_ds)
    except Exception as e:
        print(f'  SHAP error: {e}')

    print('\n[XAI 3/4] Plotting ODE latent trajectories...')
    try:
        plot_ode_trajectories(model, test_ds)
    except Exception as e:
        print(f'  ODE trajectory error: {e}')

    print('\n[XAI 4/4] Plotting clinical risk score charts...')
    try:
        plot_risk_scores(model, test_ds, scorer)
    except Exception as e:
        print(f'  Risk chart error: {e}')

    print('\n=== Explainability complete ===')
    print(f'  All plots saved to: {RESULTS_DIR}')
    print('  Files:')
    for fname in ['gradcam_results.png', 'shap_results.png',
                  'ode_trajectories.png', 'risk_scores_chart.png']:
        path = RESULTS_DIR / fname
        status = '✅' if path.exists() else '❌ not saved'
        print(f'    {fname}: {status}')
