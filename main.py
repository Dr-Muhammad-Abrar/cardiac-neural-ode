#!/usr/bin/env python3
# =============================================================================
# main.py  —  Cardiac Neural ODE — All 8 PhysioNet Datasets
# =============================================================================
# Usage:
#   python main.py                  full pipeline (50 epochs)
#   python main.py --fast           quick test (2 epochs, reduced data)
#   python main.py --verify-only    check dataset paths only
#   python main.py --no-train       skip training, load saved checkpoint
#   python main.py --epochs 20      custom epoch count
#   python main.py --no-xai         skip explainability (faster)
# =============================================================================

import sys
print("="*60, flush=True)
print("  Cardiac Neural ODE — starting...", flush=True)
print("="*60, flush=True)

import argparse
import warnings
import torch
import numpy as np

warnings.filterwarnings('ignore')

from config import (
    DEVICE, SEED, EPOCHS, LR, SEQ_LEN,
    CHECKPOINT_DIR, RESULTS_DIR, verify_datasets
)
from src.data_loader    import load_all_datasets, load_mitbih_beats, load_bidmc_segments
from src.dataset        import get_dataloaders
from src.model          import CardiacNeuralODE
from src.train          import train, evaluate, plot_training_curves
from src.ode_fitting    import run_windkessel_demo
from src.risk_scorer    import ClinicalRiskScorer, run_risk_demo
from src.explainability import run_all_explainability

torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--epochs',      type=int, default=EPOCHS)
    p.add_argument('--no-train',    action='store_true')
    p.add_argument('--fast',        action='store_true',
                   help='2 epochs, 6 MIT-BIH records only')
    p.add_argument('--no-xai',      action='store_true',
                   help='Skip explainability analysis')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'  Device : {DEVICE}')
    print(f'  Epochs : {args.epochs}')
    print(f'  Mode   : {"FAST TEST" if args.fast else "FULL"}')
    print()

    # ── 0. Verify ─────────────────────────────────────────────────────────────
    all_ok = verify_datasets()
    if args.verify_only:
        print('Verify-only mode — done.')
        return

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print('[1/7] Loading datasets...')
    if args.fast:
        ecg_beats  = load_mitbih_beats(max_records=6)
        ptb_segs   = []
        ptbxl_segs = []
        multimodal = load_bidmc_segments(max_patients=10)
    else:
        ecg_beats, ptb_segs, ptbxl_segs, multimodal = load_all_datasets()

    # ── 2. Build DataLoaders ──────────────────────────────────────────────────
    print('[2/7] Building DataLoaders...')
    tr_l, vl_l, te_l, tr_ds, vl_ds, te_ds = \
        get_dataloaders(ecg_beats, ptb_segs, ptbxl_segs, multimodal)

    # ── 3. Build model ────────────────────────────────────────────────────────
    print('[3/7] Building model...')
    model = CardiacNeuralODE().to(DEVICE)
    print(f'  Parameters: {model.count_parameters():,}')

    # ── 4. Train ──────────────────────────────────────────────────────────────
    ckpt = CHECKPOINT_DIR / 'cardiac_node_best.pt'
    if args.no_train and ckpt.exists():
        print(f'[4/7] Loading checkpoint: {ckpt}')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        history = None
    else:
        print(f'[4/7] Training...')
        history, best_acc = train(model, tr_l, vl_l,
                                  epochs=args.epochs, lr=LR)
        if history:
            plot_training_curves(history)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print('[5/7] Evaluating...')
    evaluate(model, te_l)

    # ── 6. ODE + Risk ─────────────────────────────────────────────────────────
    print('[6/7] Windkessel fitting and risk scoring...')
    run_windkessel_demo(multimodal)
    scorer = ClinicalRiskScorer()
    run_risk_demo(model, te_ds, scorer)

    # ── 7. Explainability ─────────────────────────────────────────────────────
    if not args.no_xai:
        print('[7/7] Running explainability analysis...')
        run_all_explainability(model, te_ds, scorer)
    else:
        print('[7/7] Explainability skipped (--no-xai)')

    print('\n' + '='*60)
    print('  Pipeline complete!')
    print(f'  Results    : {RESULTS_DIR}')
    print(f'  Checkpoint : {ckpt}')
    print('\n  Output files:')
    files = [
        'training_curves.png',
        'confusion_matrix.png',
        'windkessel_fit.png',
        'gradcam_results.png',
        'shap_results.png',
        'ode_trajectories.png',
        'risk_scores_chart.png',
    ]
    for f in files:
        p = RESULTS_DIR / f
        status = '✅' if p.exists() else '⏳ not yet'
        print(f'    {f:30s}: {status}')
    print('='*60)


if __name__ == '__main__':
    main()
