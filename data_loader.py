# =============================================================================
# src/data_loader.py  —  All 8 PhysioNet datasets
# =============================================================================

import warnings
import tarfile
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

import wfdb
warnings.filterwarnings('ignore')

from config import (
    MITBIH_DIR, MITBIH_RECORDS, MITBIH_LABEL_MAP,
    VFDB_DIR,   VFDB_RECORDS,
    NSRDB_DIR,  NSRDB_RECORDS,
    SVDB_DIR,   SVDB_RECORDS,
    BIDMC_DIR,  CHALLENGE_DIR,
    PTBXL_DIR,  PTBXL_META_CSV, PTBXL_SCP_CSV,
    PTBXL_FS,   PTBXL_LABEL_MAP,
    PTB_DIR,    PTB_LABEL_MAP,
    SEQ_LEN,    MAX_PER_CLASS,
    MAX_PTB_RECORDS, MAX_PTBXL, MAX_BIDMC, MAX_CHALLENGE,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def resample_to_length(arr, n):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == n:
        return arr
    if len(arr) < 2:
        return np.zeros(n, dtype=np.float32)
    x = np.linspace(0, 1, len(arr))
    return interp1d(x, arr, kind='linear',
                    fill_value='extrapolate')(
        np.linspace(0, 1, n)).astype(np.float32)


def normalise(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def safe_filter(ecg, fs, low=0.5, high=40.0, order=4):
    """Bandpass filter safe for any sampling rate."""
    ecg = np.asarray(ecg, dtype=np.float32)
    nyq = fs / 2.0
    if nyq <= low + 0.1:
        return ecg
    h = min(high, nyq * 0.95)
    if low >= h:
        return ecg
    try:
        b, a = sp_signal.butter(order, [low / nyq, h / nyq], btype='band')
        return sp_signal.filtfilt(b, a, ecg).astype(np.float32)
    except Exception:
        return ecg


# ── 1. MIT-BIH Arrhythmia ─────────────────────────────────────────────────────

def load_mitbih_beats(window_size=360, channel=0, max_records=None):
    records = MITBIH_RECORDS if max_records is None else MITBIH_RECORDS[:max_records]
    beats   = []
    print(f'Loading MIT-BIH Arrhythmia ({len(records)} records)...')
    for rid in records:
        rp = str(MITBIH_DIR / rid)
        if not (MITBIH_DIR / f'{rid}.hea').exists():
            continue
        try:
            rec = wfdb.rdrecord(rp)
            ann = wfdb.rdann(rp, 'atr')
        except Exception:
            continue
        ecg  = safe_filter(rec.p_signal[:, channel].astype(np.float32), rec.fs)
        half = window_size // 2
        for s, sym in zip(ann.sample, ann.symbol):
            lbl = MITBIH_LABEL_MAP.get(sym, -1)
            if lbl == -1 or s - half < 0 or s + half > len(ecg):
                continue
            beats.append((normalise(ecg[s-half:s+half]), lbl))
    c = Counter(l for _, l in beats)
    print(f'  MIT-BIH beats: {len(beats):,}  '
          f'N={c[0]} L={c[1]} R={c[2]} A={c[3]} V={c[4]}')
    return beats


# ── 2. MIT-BIH Ventricular ────────────────────────────────────────────────────

def load_vfdb_beats(window_size=360, channel=0):
    beats = []
    print(f'Loading MIT-BIH Ventricular ({len(VFDB_RECORDS)} records)...')
    for rid in VFDB_RECORDS:
        rp = str(VFDB_DIR / rid)
        if not (VFDB_DIR / f'{rid}.hea').exists():
            continue
        try:
            rec = wfdb.rdrecord(rp)
            ann = wfdb.rdann(rp, 'atr')
        except Exception:
            continue
        ecg  = safe_filter(rec.p_signal[:, min(channel, rec.n_sig-1)].astype(np.float32), rec.fs)
        half = window_size // 2
        for s in ann.sample[::10]:
            if s - half < 0 or s + half > len(ecg):
                continue
            beats.append((normalise(ecg[s-half:s+half]), 4))
    print(f'  VFDB beats: {len(beats):,}')
    return beats


# ── 3. MIT-BIH Normal Sinus Rhythm ───────────────────────────────────────────

def load_nsrdb_beats(window_size=360, channel=0, per_rec=200):
    beats = []
    print(f'Loading MIT-BIH Normal SR ({len(NSRDB_RECORDS)} records)...')
    for rid in NSRDB_RECORDS:
        rp = str(NSRDB_DIR / rid)
        if not (NSRDB_DIR / f'{rid}.hea').exists():
            continue
        try:
            rec = wfdb.rdrecord(rp)
        except Exception:
            continue
        ecg  = safe_filter(rec.p_signal[:, min(channel, rec.n_sig-1)].astype(np.float32), rec.fs)
        half = window_size // 2
        step = max(1, (len(ecg) - window_size) // per_rec)
        cnt  = 0
        for s in range(half, len(ecg) - half, step):
            beats.append((normalise(ecg[s-half:s+half]), 0))
            cnt += 1
            if cnt >= per_rec:
                break
    print(f'  NSRDB beats: {len(beats):,}')
    return beats


# ── 4. MIT-BIH Supraventricular ───────────────────────────────────────────────

def load_svdb_beats(window_size=360, channel=0):
    LBL = {'N':0,'A':3,'S':3,'a':3,'J':3,'j':3,'V':4,'F':4,'e':0,'R':2,'L':1}
    beats = []
    print(f'Loading MIT-BIH Supraventricular ({len(SVDB_RECORDS)} records)...')
    for rid in SVDB_RECORDS:
        rp = str(SVDB_DIR / rid)
        if not (SVDB_DIR / f'{rid}.hea').exists():
            continue
        try:
            rec = wfdb.rdrecord(rp)
            ann = wfdb.rdann(rp, 'atr')
        except Exception:
            continue
        ecg  = safe_filter(rec.p_signal[:, min(channel, rec.n_sig-1)].astype(np.float32), rec.fs)
        half = window_size // 2
        for s, sym in zip(ann.sample, ann.symbol):
            lbl = LBL.get(sym, -1)
            if lbl == -1 or s - half < 0 or s + half > len(ecg):
                continue
            beats.append((normalise(ecg[s-half:s+half]), lbl))
    print(f'  SVDB beats: {len(beats):,}')
    return beats


# ── 5. PTB Diagnostic ─────────────────────────────────────────────────────────

def load_ptb_segments(duration_sec=5, target_fs=360, channel=1,
                      max_records=MAX_PTB_RECORDS):
    print(f'Loading PTB Diagnostic (up to {max_records} records)...')
    segs  = []
    count = 0
    for hea in sorted(PTB_DIR.rglob('*.hea')):
        if count >= max_records:
            break
        rp = str(hea.with_suffix(''))
        try:
            rec     = wfdb.rdrecord(rp)
            ecg     = rec.p_signal[:, min(channel, rec.n_sig-1)].astype(np.float32)
            n_orig  = int(duration_sec * rec.fs)
            n_tgt   = int(duration_sec * target_fs)
            if len(ecg) < n_orig:
                ecg = np.pad(ecg, (0, n_orig - len(ecg)))
            seg = ecg[:n_orig]
            if rec.fs != target_fs:
                from scipy.signal import resample as sp_resample
                seg = sp_resample(seg, n_tgt).astype(np.float32)
            seg = normalise(safe_filter(seg, target_fs))

            # Strict keyword parsing
            lbl = 0
            if rec.comments:
                txt = ' '.join(str(c) for c in rec.comments).lower()
                if 'myocardial infarction' in txt:
                    lbl = 1
                elif any(k in txt for k in ['cardiomyopathy','myocarditis',
                                             'hypertrophy','valvular']):
                    lbl = 2
                elif 'bundle branch block' in txt:
                    lbl = 3
                elif 'dysrhythmia' in txt:
                    lbl = 4

            segs.append((seg, lbl))
            count += 1
        except Exception:
            continue

    c = Counter(l for _, l in segs)
    print(f'  PTB segments: {len(segs):,}  '
          f'0={c[0]} 1={c[1]} 2={c[2]} 3={c[3]} 4={c[4]}')
    return segs


# ── 6. PTB-XL ─────────────────────────────────────────────────────────────────

def load_ptbxl_segments(max_records=MAX_PTBXL, channel=1):
    import pandas as pd
    import ast
    print(f'Loading PTB-XL (up to {max_records} records)...')
    if not PTBXL_META_CSV.exists():
        print(f'  PTB-XL metadata not found: {PTBXL_META_CSV}')
        return []
    try:
        df = pd.read_csv(PTBXL_META_CSV, index_col='ecg_id')
    except Exception as e:
        print(f'  PTB-XL CSV error: {e}')
        return []

    def get_lbl(s):
        try:
            for code in ast.literal_eval(s):
                if code in PTBXL_LABEL_MAP:
                    return PTBXL_LABEL_MAP[code]
        except Exception:
            pass
        return 0

    df['label'] = df['scp_codes'].apply(get_lbl)
    df = df.sample(frac=1, random_state=42).head(max_records)
    segs = []
    for _, row in df.iterrows():
        try:
            rp  = str(PTBXL_DIR / row['filename_lr'])
            rec = wfdb.rdrecord(rp)
            ecg = rec.p_signal[:, min(channel, rec.n_sig-1)].astype(np.float32)
            ecg = normalise(safe_filter(ecg, PTBXL_FS))
            segs.append((resample_to_length(ecg, SEQ_LEN), int(row['label'])))
        except Exception:
            continue

    c = Counter(l for _, l in segs)
    print(f'  PTB-XL segments: {len(segs):,}  '
          f'0={c[0]} 1={c[1]} 2={c[2]} 3={c[3]} 4={c[4]}')
    return segs


# ── 7. BIDMC ──────────────────────────────────────────────────────────────────

def load_bidmc_segments(max_patients=MAX_BIDMC):
    print('Loading BIDMC PPG+ABP...')
    segs = []

    def best_ch(names, keywords):
        nl = [n.lower() for n in names]
        for kw in keywords:
            for i, n in enumerate(nl):
                if kw in n:
                    return i
        return 0

    for hea in sorted(BIDMC_DIR.glob('*.hea'))[:max_patients]:
        rp = str(hea.with_suffix(''))
        try:
            rec   = wfdb.rdrecord(rp)
            sig   = rec.p_signal
            names = rec.sig_name
            fs    = rec.fs
            ei    = best_ch(names, ['ii','ecg','ekg'])
            ai    = best_ch(names, ['abp','art','bp'])
            pi    = best_ch(names, ['ppg','pleth','spo2'])
            slen  = int(10 * fs)
            for k in range(min(len(sig) // slen, 5)):
                s = k * slen
                segs.append({
                    'ecg':   normalise(resample_to_length(
                                 safe_filter(sig[s:s+slen, ei].astype(np.float32), fs),
                                 SEQ_LEN)),
                    'abp':   normalise(resample_to_length(
                                 sig[s:s+slen, ai].astype(np.float32), SEQ_LEN)),
                    'spo2':  normalise(resample_to_length(
                                 sig[s:s+slen, pi].astype(np.float32), SEQ_LEN)),
                    'label': 0, 'source': 'bidmc',
                })
        except Exception:
            continue

    print(f'  BIDMC segments: {len(segs):,}')
    return segs


# ── 8. PhysioNet Challenge 2015 ───────────────────────────────────────────────

def load_challenge2015_segments(max_records=MAX_CHALLENGE):
    print(f'Loading Challenge 2015 (up to {max_records})...')

    # Auto-extract tar.gz if present
    tar = CHALLENGE_DIR / 'entry.tar.gz'
    if tar.exists():
        print('  Extracting entry.tar.gz...')
        try:
            with tarfile.open(tar, 'r:gz') as tf:
                tf.extractall(CHALLENGE_DIR)
            print('  Done.')
        except Exception as e:
            print(f'  Extract error: {e}')

    # Collect all .hea files recursively
    # Check training subfolder first, then search recursively
    training_sub = CHALLENGE_DIR / 'training'
    if training_sub.exists():
        hea_files = list(training_sub.glob('*.hea'))[:max_records]
    else:
        hea_files = list(CHALLENGE_DIR.rglob('*.hea'))[:max_records]
    print(f'  Found {len(hea_files)} .hea files')
    if not hea_files:
        return []

    segs     = []
    lbl_map  = {'a': 1, 'b': 0, 'n': 0}

    def best_ch(names, keywords):
        nl = [n.lower() for n in names]
        for kw in keywords:
            for i, n in enumerate(nl):
                if kw in n:
                    return i
        return 0

    for hea in hea_files:
        rp = str(hea.with_suffix(''))
        try:
            rec  = wfdb.rdrecord(rp)
            sig  = rec.p_signal
            fs   = rec.fs
            names = rec.sig_name
            ei   = best_ch(names, ['ii','ecg','v','avr'])
            ai   = best_ch(names, ['abp','art','bp','pa'])
            pi   = best_ch(names, ['spo2','pleth','ppg'])
            slen = min(int(60 * fs), len(sig))
            lbl  = lbl_map.get(hea.stem[-1].lower(), 0)
            segs.append({
                'ecg':   normalise(resample_to_length(
                             safe_filter(sig[:slen, ei].astype(np.float32), fs),
                             SEQ_LEN)),
                'abp':   normalise(resample_to_length(
                             sig[:slen, ai].astype(np.float32), SEQ_LEN)),
                'spo2':  normalise(resample_to_length(
                             sig[:slen, pi].astype(np.float32), SEQ_LEN)),
                'label': lbl, 'source': 'challenge2015',
            })
        except Exception:
            continue

    print(f'  Challenge 2015 segments: {len(segs):,}')
    return segs


# ── Master loader ─────────────────────────────────────────────────────────────

def load_all_datasets():
    print('\n' + '='*55)
    print('  Loading all 8 PhysioNet datasets')
    print('='*55)

    ecg_beats  = (load_mitbih_beats() + load_vfdb_beats()
                  + load_nsrdb_beats() + load_svdb_beats())
    ptb_segs   = load_ptb_segments()
    ptbxl_segs = load_ptbxl_segments()
    multimodal = load_bidmc_segments() + load_challenge2015_segments()

    print(f'\n{"="*55}')
    print(f'  ECG beats   : {len(ecg_beats):>8,}')
    print(f'  PTB segments: {len(ptb_segs):>8,}')
    print(f'  PTB-XL segs : {len(ptbxl_segs):>8,}')
    print(f'  Multi-modal : {len(multimodal):>8,}')
    print(f'{"="*55}\n')
    return ecg_beats, ptb_segs, ptbxl_segs, multimodal
