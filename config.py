# =============================================================================
# config.py
# =============================================================================
import sys
import torch
from pathlib import Path

# ── Flush all prints immediately ──────────────────────────────────────────────
import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)
builtins.print = print

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
CACHE_DIR      = ROOT_DIR / 'data' / 'physionet_cache'
RESULTS_DIR    = ROOT_DIR / 'results'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset folders ───────────────────────────────────────────────────────────
MITBIH_DIR    = CACHE_DIR / 'mit-bih-arrhythmia-database-1.0.0'
VFDB_DIR      = CACHE_DIR / 'mit-bih-malignant-ventricular-ectopy-database-1.0.0'
NSRDB_DIR     = CACHE_DIR / 'mit-bih-normal-sinus-rhythm-database-1.0.0'
SVDB_DIR      = CACHE_DIR / 'mit-bih-supraventricular-arrhythmia-database-1.0.0'
BIDMC_DIR     = CACHE_DIR / 'bidmc-ppg-and-respiration-dataset-1.0.0'
CHALLENGE_DIR = CACHE_DIR / 'reducing-false-arrhythmia-alarms-in-the-icu-the-physionetcomputing-in-cardiology-challenge-2015-1.0.0'
PTBXL_DIR     = CACHE_DIR / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
PTB_DIR       = CACHE_DIR / 'ptb-diagnostic-ecg-database-1.0.0'

ALL_DIRS = {
    'MIT-BIH Arrhythmia':        MITBIH_DIR,
    'MIT-BIH Ventricular':       VFDB_DIR,
    'MIT-BIH Normal SR':         NSRDB_DIR,
    'MIT-BIH Supraventricular':  SVDB_DIR,
    'BIDMC PPG+ABP':             BIDMC_DIR,
    'PhysioNet Challenge 2015':  CHALLENGE_DIR,
    'PTB-XL':                    PTBXL_DIR,
    'PTB Diagnostic':            PTB_DIR,
}

def verify_datasets():
    print('\n=== Dataset Path Verification ===')
    all_ok = True
    for name, path in ALL_DIRS.items():
        ok = path.exists()
        print(f'  {name:30s}: {"OK" if ok else "NOT FOUND — " + str(path)}')
        if not ok:
            all_ok = False
    print('All datasets found.\n' if all_ok else '\nWARNING: some paths missing.\n')
    return all_ok

# ── MIT-BIH records ───────────────────────────────────────────────────────────
MITBIH_RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]
MITBIH_LABEL_MAP = {
    'N':0,'L':1,'R':2,'A':3,'V':4,'F':4,'f':3,'e':0,'j':0,
    'a':3,'J':0,'S':3,'/':0,'Q':4,'B':3,'!':4,
}

VFDB_RECORDS = [
    '418','419','420','421','422','423','424','425','426','427',
    '428','429','430','431','432','433','434','435','436','437','438','439'
]

NSRDB_RECORDS = [
    '16265','16272','16273','16420','16483','16539','16773',
    '16786','16795','17052','17453','18177','18184','19088',
    '19090','19093','19140','19830'
]

SVDB_RECORDS = [
    '800','801','802','803','804','805','806','807','808','809',
    '810','811','812','820','821','822','823','824','825','826',
    '827','828','829','840','841','842','843','844','845','846',
    '847','848','849','850','851','852','853','854','855','856',
    '857','858','859','860','861','862','863','864','865','866',
    '867','868','869','870','871','872','873','874','875','876',
    '877','878','879','880','881','882','883','884','885','886',
    '887','888','889','890','891','892','893','894'
]

# ── PTB-XL ────────────────────────────────────────────────────────────────────
PTBXL_META_CSV  = PTBXL_DIR / 'ptbxl_database.csv'
PTBXL_SCP_CSV   = PTBXL_DIR / 'scp_statements.csv'
PTBXL_FS        = 100
PTBXL_LABEL_MAP = {
    'NORM':0,'IMI':1,'ASMI':1,'ILMI':1,'AMI':1,'ALMI':1,
    'INJAS':1,'LMI':1,'INJAL':1,'IPLMI':1,'IPMI':1,'PMI':1,
    'LBBB':1,'RBBB':2,'IRBBB':2,'CLBBB':1,'CRBBB':2,
    'LAFB':3,'LPFB':3,'IVCD':3,
    'STACH':3,'SVTAC':3,'PSVT':3,'AFIB':3,'AFLT':3,
    'BIGU':4,'TRIGU':4,'PVC':4,'VPB':4,
}

# ── PTB ───────────────────────────────────────────────────────────────────────
PTB_LABEL_MAP = {
    'Healthy control':0, 'Myocardial infarction':1,
    'Cardiomyopathy':2,  'Bundle branch block':3,
    'Dysrhythmia':4,     'Myocarditis':2,
    'Hypertrophy':2,     'Valvular heart disease':2,
}

# ── Model ─────────────────────────────────────────────────────────────────────
SEQ_LEN    = 256
LATENT_DIM = 32
HIDDEN_DIM = 64
LSTM_H     = 64
N_CLASSES  = 5

CLASS_NAMES = [
    'Normal / Stable',
    'MI / LBBB',
    'Cardiomyopathy / RBBB',
    'Atrial / Supraventricular',
    'Ventricular / Dangerous',
]

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 50
LR              = 3e-4
BETA_KL         = 0.005
MAX_PER_CLASS   = 2000
MAX_PTB_RECORDS = 549
MAX_PTBXL       = 5000
MAX_BIDMC       = 53
MAX_CHALLENGE   = 500
