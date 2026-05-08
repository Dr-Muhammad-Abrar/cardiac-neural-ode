# =============================================================================
# src/ode_fitting.py  —  Windkessel ODE fitting via CasADi / IPOPT
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import RESULTS_DIR

try:
    import casadi as ca
    CASADI_OK = True
except ImportError:
    CASADI_OK = False
    print('CasADi not installed — Windkessel fitting will be skipped.')


def fit_windkessel(abp_norm, t_vec):
    if not CASADI_OK:
        return None
    N  = len(t_vec) - 1
    dt = float(t_vec[1] - t_vec[0])
    R  = ca.SX.sym('R'); C = ca.SX.sym('C'); HR = ca.SX.sym('HR')
    P  = ca.SX.sym('P', N + 1)
    g, lbg, ubg = [], [], []
    for k in range(N):
        phase = ca.fmod(float(t_vec[k]) * HR, 1.0)
        Q_k   = ca.if_else(phase < 0.35, ca.sin(phase / 0.35 * 3.14159), 0.0)
        g.append(P[k+1] - P[k] - dt * (Q_k - P[k] / R) / C)
        lbg.append(0.0); ubg.append(0.0)
    obs = ca.DM(abp_norm)
    w   = ca.vertcat(R, C, HR, P)
    nlp = {'x': w, 'f': ca.sumsqr(P - obs), 'g': ca.vertcat(*g)}
    opt = {'ipopt': {'print_level': 0, 'max_iter': 300}}
    try:
        sol   = ca.nlpsol('S', 'ipopt', nlp, opt)(
            x0  = [1.0, 1.0, 1.2] + list(abp_norm),
            lbx = [0.1, 0.1, 0.5] + [-5.0] * (N + 1),
            ubx = [5.0, 5.0, 3.0] + [ 5.0] * (N + 1),
            lbg = lbg, ubg = ubg)
        w_opt = np.array(sol['x']).flatten()
        R_o, C_o, HR_o = w_opt[0], w_opt[1], w_opt[2]
        P_o   = w_opt[3:]
        mse   = float(np.mean((P_o - abp_norm) ** 2))
        return {'R': R_o, 'C': C_o, 'HR_bps': HR_o, 'P_fit': P_o, 'mse': mse}
    except Exception as e:
        print(f'  IPOPT error: {e}')
        return None


def run_windkessel_demo(multimodal, save=True):
    bidmc = [s for s in multimodal if s.get('source') == 'bidmc']
    if not bidmc:
        print('No BIDMC segments for Windkessel demo.')
        return None
    abp   = bidmc[0]['abp'][:200]
    t_vec = np.linspace(0, len(abp) / 125.0, len(abp))
    print('Running Windkessel fit...')
    result = fit_windkessel(abp, t_vec)
    if result:
        print(f"  R={result['R']:.4f}  C={result['C']:.4f}  "
              f"HR={result['HR_bps']*60:.1f} bpm  MSE={result['mse']:.6f}")
        plt.figure(figsize=(10, 3))
        plt.plot(t_vec, abp,             'b-', alpha=0.6, label='Observed ABP')
        plt.plot(t_vec, result['P_fit'], 'r-', lw=1.5,   label='Windkessel fit')
        plt.xlabel('Time (s)'); plt.ylabel('Normalised ABP')
        plt.title('Windkessel ODE — CasADi/IPOPT Fitting')
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        if save:
            p = RESULTS_DIR / 'windkessel_fit.png'
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f'  Saved: {p}')
        plt.close()
    return result
