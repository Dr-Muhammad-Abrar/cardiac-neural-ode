# =============================================================================
# src/risk_scorer.py  —  Clinical risk scoring
# =============================================================================

import numpy as np
import torch
from config import DEVICE, SEQ_LEN, CLASS_NAMES


class ClinicalRiskScorer:
    SW = np.array([0.02, 0.10, 0.08, 0.55, 0.25])  # stroke
    HW = np.array([0.02, 0.50, 0.45, 0.15, 0.35])  # heart failure
    AW = np.array([0.03, 0.15, 0.10, 0.60, 0.70])  # arrhythmia

    def score(self, probs):
        def w(weights):
            s = float(np.dot(probs, weights))
            return min(max(s, 0.0), 1.0) * 100
        return {
            'Stroke risk %':        w(self.SW),
            'Heart failure risk %': w(self.HW),
            'Arrhythmia risk %':    w(self.AW),
        }


def run_risk_demo(model, test_ds, scorer, n=6):
    t_span = torch.linspace(0, 1, SEQ_LEN).to(DEVICE)
    model.eval()
    print('\n=== Clinical Risk Scoring Demo ===')
    for i in range(min(n, len(test_ds))):
        x, y = test_ds[i]
        with torch.no_grad():
            logits, _, _ = model(x.unsqueeze(0).to(DEVICE), t_span)
            probs = torch.softmax(logits, 1).cpu().numpy()[0]
        pred = probs.argmax()
        ok   = '✅' if pred == y.item() else '❌'
        print(f'\nSample {i+1} | True: {CLASS_NAMES[y.item()][:25]:25s} '
              f'| Pred: {CLASS_NAMES[pred][:25]:25s} {ok}')
        for rname, rval in scorer.score(probs).items():
            bar   = '▓' * int(rval / 5) + '░' * (20 - int(rval / 5))
            level = 'HIGH' if rval > 60 else ('MOD' if rval > 30 else 'LOW ')
            print(f'  {rname:22s}: [{bar}] {rval:5.1f}%  {level}')
