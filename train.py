# =============================================================================
# src/train.py  —  Training loop + evaluation
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns

from config import DEVICE, EPOCHS, LR, SEQ_LEN, RESULTS_DIR, CHECKPOINT_DIR, CLASS_NAMES


def elbo_loss(logits, targets, mu, lv, beta=0.005):
    ce = nn.CrossEntropyLoss()(logits, targets)
    kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
    return ce + beta * kl, ce.item(), kl.item()


def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    t_span    = torch.linspace(0, 1, SEQ_LEN).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader))

    history      = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_val_acc = 0.0
    best_state   = None

    print(f'Training {epochs} epoch(s) on {DEVICE}...')

    for epoch in range(epochs):
        # Train
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits, mu, lv = model(xb, t_span)
            loss, _, _     = elbo_loss(logits, yb, mu, lv)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss    += loss.item() * len(xb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += len(xb)

        # Validate
        model.eval()
        vl_loss = vl_correct = vl_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits, mu, lv = model(xb, t_span)
                loss, _, _     = elbo_loss(logits, yb, mu, lv)
                vl_loss    += loss.item() * len(xb)
                vl_correct += (logits.argmax(1) == yb).sum().item()
                vl_total   += len(xb)

        tl = tr_loss / tr_total
        vl = vl_loss / vl_total
        ta = tr_correct / tr_total
        va = vl_correct / vl_total

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)

        if va > best_val_acc:
            best_val_acc = va
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        print(f'  Epoch {epoch+1:3d}/{epochs}  '
              f'loss={tl:.4f}/{vl:.4f}  acc={ta:.3f}/{va:.3f}')

    model.load_state_dict(best_state)
    ckpt = CHECKPOINT_DIR / 'cardiac_node_best.pt'
    torch.save(model.state_dict(), ckpt)
    print(f'Best val acc: {best_val_acc*100:.1f}%  Checkpoint: {ckpt}')
    return history, best_val_acc


def plot_training_curves(history, save=True):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history['train_loss'], label='Train')
    a1.plot(history['val_loss'],   label='Val')
    a1.set_title('Loss'); a1.legend()
    a2.plot(history['train_acc'], label='Train')
    a2.plot(history['val_acc'],   label='Val')
    a2.set_title('Accuracy'); a2.legend(); a2.set_ylim(0, 1)
    plt.tight_layout()
    if save:
        p = RESULTS_DIR / 'training_curves.png'
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f'Training curves saved: {p}')
    plt.close()


def evaluate(model, test_loader, save=True):
    t_span = torch.linspace(0, 1, SEQ_LEN).to(DEVICE)
    model.eval()
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb   = xb.to(DEVICE), yb.to(DEVICE)
            logits, _, _ = model(xb, t_span)
            p        = torch.softmax(logits, 1)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(yb.cpu().numpy())
            probs.extend(p.cpu().numpy())

    preds  = np.array(preds)
    labels = np.array(labels)
    probs  = np.array(probs)

    present = sorted(set(labels))
    names   = [CLASS_NAMES[i] for i in present]

    print('\nClassification Report:')
    print(classification_report(labels, preds,
                                labels=present,
                                target_names=names,
                                digits=3, zero_division=0))
    try:
        yb  = label_binarize(labels, classes=present)
        auc = roc_auc_score(yb, probs[:, :len(present)],
                            multi_class='ovr', average='macro')
        print(f'Macro ROC-AUC: {auc:.4f}')
    except Exception as e:
        print(f'AUC: {e}')

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=present)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=names, yticklabels=names)
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right', fontsize=8)

    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Oranges', ax=axes[1],
                xticklabels=names, yticklabels=names)
    axes[1].set_title('Confusion Matrix (normalised)')
    axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right', fontsize=8)

    plt.tight_layout()
    if save:
        p = RESULTS_DIR / 'confusion_matrix.png'
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f'Confusion matrix saved: {p}')
    plt.close()
    return preds, labels, probs
