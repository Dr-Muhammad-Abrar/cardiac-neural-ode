# =============================================================================
# src/dataset.py
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter

from config import SEQ_LEN, BATCH_SIZE, SEED, MAX_PER_CLASS
from src.data_loader import resample_to_length, normalise


class CardiacDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))


def _make_x(seg_ecg, seg_abp=None, seg_spo2=None):
    ecg = normalise(resample_to_length(
        np.asarray(seg_ecg, dtype=np.float32), SEQ_LEN))
    hrv = normalise(np.gradient(ecg))
    abp = (normalise(resample_to_length(
        np.asarray(seg_abp, dtype=np.float32), SEQ_LEN))
           if seg_abp is not None
           else np.zeros(SEQ_LEN, dtype=np.float32))
    sp2 = (normalise(resample_to_length(
        np.asarray(seg_spo2, dtype=np.float32), SEQ_LEN))
           if seg_spo2 is not None
           else hrv * 0.5)
    return np.stack([ecg, abp, sp2], axis=-1).astype(np.float32)


def build_unified_samples(ecg_beats, ptb_segs, ptbxl_segs, multimodal):
    bins = {i: [] for i in range(5)}

    print('Building unified dataset...')

    # ECG beats — (seg, label)
    print(f'  ECG beats  : {len(ecg_beats):,}')
    for item in ecg_beats:
        lbl = min(int(item[1]), 4)
        bins[lbl].append(_make_x(item[0]))

    # PTB — (seg, label) or (seg, label, diagnosis)
    print(f'  PTB segs   : {len(ptb_segs):,}')
    for item in ptb_segs:
        lbl = min(int(item[1]), 4)
        bins[lbl].append(_make_x(item[0]))

    # PTB-XL — (seg, label)
    print(f'  PTB-XL segs: {len(ptbxl_segs):,}')
    for item in ptbxl_segs:
        lbl = min(int(item[1]), 4)
        bins[lbl].append(_make_x(item[0]))

    # Multi-modal dicts
    print(f'  Multi-modal: {len(multimodal):,}')
    for d in multimodal:
        lbl = min(int(d['label']), 4)
        bins[lbl].append(_make_x(d['ecg'], d.get('abp'), d.get('spo2')))

    # Cap and assemble
    print(f'\n  Per-class cap = {MAX_PER_CLASS:,}')
    names   = ['Normal','MI/LBBB','RBBB/Cardio','Atrial','Ventricular']
    samples = []
    for lbl in range(5):
        b = bins[lbl]
        np.random.shuffle(b)
        used = b[:MAX_PER_CLASS]
        for x in used:
            samples.append((x, lbl))
        print(f'    Class {lbl} {names[lbl]:15s}: '
              f'{len(b):6,} avail  {len(used):6,} used')

    np.random.shuffle(samples)
    print(f'\n  Total samples: {len(samples):,}')
    return samples


def get_dataloaders(ecg_beats, ptb_segs, ptbxl_segs, multimodal):
    samples = build_unified_samples(ecg_beats, ptb_segs, ptbxl_segs, multimodal)
    ds      = CardiacDataset(samples)
    n       = len(ds)
    n_tr    = int(n * 0.70)
    n_vl    = int(n * 0.15)
    n_te    = n - n_tr - n_vl

    tr, vl, te = random_split(
        ds, [n_tr, n_vl, n_te],
        generator=torch.Generator().manual_seed(SEED))

    tr_l = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    vl_l = DataLoader(vl, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_l = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'  Train={n_tr:,}  Val={n_vl:,}  Test={n_te:,}\n')
    return tr_l, vl_l, te_l, tr, vl, te
