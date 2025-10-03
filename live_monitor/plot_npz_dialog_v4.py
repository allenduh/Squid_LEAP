#!/usr/bin/env python3
"""
plot_npz_dialog_v4.py

Pick an exp_block_data.npz via a file dialog and generate a 4‑page PDF:

  1) RAW TRACES (μ subtracted) — plot per ROI. Annotate with Δ (from page‑4 Gaussian fit),
     σ (from residual, page 3), S/N=Δ/σ, and μ_raw.
  2) TRACES (HP 0.2 Hz) — draw two red dotted lines at fitted low/high levels (means of the two
     equal‑height Gaussians). Annotate with Δ, σ, S/N.
  3) RESIDUAL (HP ~20 Hz + 60 Hz notch) — compute σ = std(residual); draw ±σ as red dotted lines.
  4) HISTOGRAMS (HP 0.2 Hz) — overlay **two equal‑height Gaussians** with **σ fixed to residual σ**.
     Means are fitted (LS) and can naturally overlap if signal is tiny; no x‑axis numbers.

Dependencies: numpy, matplotlib; SciPy optional (for Butterworth, iirnotch). Fitting uses NumPy only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations

# --- optional SciPy (preferred for filters) ---
try:
    from scipy.signal import butter, filtfilt, iirnotch  # type: ignore
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# --- optional Tkinter dialog ---
def pick_npz_dialog(initial: str | None = None) -> Path:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    filetypes = [("NumPy NPZ","*.npz"),("All","*.*")]
    p = filedialog.askopenfilename(title="Select exp_block_data.npz",
                                   initialdir=initial or str(Path.cwd()),
                                   filetypes=filetypes)
    if not p:
        raise SystemExit("No file selected")
    return Path(p)

# ------------------------------ utilities ------------------------------
def detrend_linear(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    t = np.arange(y.size, dtype=float)
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    return y - (a + b*t)

def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y.astype(float, copy=True)
    pad = win//2
    ypad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(win)/float(win)
    sm = np.convolve(ypad, kernel, mode='valid')
    return sm[:y.size]

def butter_highpass(data: np.ndarray, cutoff_hz: float, fs: float, order: int = 1) -> np.ndarray:
    if not SCIPY_OK:
        # Fallback: AC-coupling via moving average subtraction (approximate HP)
        win = max(1, int(round(fs / max(cutoff_hz, 1e-6))))
        return data - moving_average(data, win)
    nyq = 0.5 * fs
    Wn = max(1e-6, min(cutoff_hz / nyq, 0.999999))
    b, a = butter(order, Wn, btype='highpass')
    return filtfilt(b, a, data)

def notch_60(data: np.ndarray, fs: float, f0: float = 60.0, Q: float = 30.0) -> np.ndarray:
    """Apply a 60 Hz notch. If SciPy unavailable or f0 above Nyquist, use LS sin/cos removal."""
    nyq = 0.5 * fs
    if f0 >= nyq or f0 <= 0.0:
        return data.copy()
    if SCIPY_OK:
        w0 = f0 / nyq
        b, a = iirnotch(w0, Q)
        return filtfilt(b, a, data)
    # fallback: fit and subtract a*cos(2πf0t)+b*sin(2πf0t)
    t = np.arange(data.size, dtype=float)/fs
    X = np.vstack([np.cos(2*np.pi*f0*t), np.sin(2*np.pi*f0*t)]).T
    coef, *_ = np.linalg.lstsq(X, data, rcond=None)
    return data - X.dot(coef)

# ------------------------------ Gaussian fit on histogram ------------------------------
def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        sigma = 1e-12
    z = (x - mu)/sigma
    return np.exp(-0.5*z*z)

def fit_two_equal_height_gaussians(hist: np.ndarray, centers: np.ndarray, sigma: float,
                                   topk: int = 15, thr_frac: float = 0.2) -> tuple[float,float,float,float]:
    """
    Fit y ≈ H * (G(x; mu1, sigma) + G(x; mu2, sigma)) in least squares, with unknown mu1, mu2, H.
    We search mu1, mu2 over a compact candidate set (top-k bins by height or above a fraction of max).
    Returns (mu_low, mu_high, H, sse).
    """
    y = np.asarray(hist, float); x = np.asarray(centers, float)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(np.repeat(x, np.maximum(y.astype(int), 0))) + 1e-6  # crude fallback
        if not np.isfinite(sigma) or sigma <= 0: sigma = (x.max()-x.min()+1e-9)/20.0

    # candidate indices: bins above threshold or top-k
    thr = thr_frac * (y.max() if y.size else 0.0)
    cand = np.where(y >= thr)[0]
    if cand.size < 2:
        # fallback to top-k bins
        k = min(topk, max(2, y.size))
        cand = np.argsort(y)[-k:]

    cand = np.unique(np.sort(cand))

    best = (None, None, 0.0, np.inf)  # (mu1, mu2, H, sse)
    for i, j in combinations(cand, 2):
        mu1, mu2 = x[i], x[j]
        gsum = gaussian(x, mu1, sigma) + gaussian(x, mu2, sigma)
        denom = float(np.dot(gsum, gsum)) + 1e-18
        H = float(np.dot(y, gsum) / denom)
        if H < 0:
            H = 0.0
        resid = y - H*gsum
        sse = float(np.dot(resid, resid))
        if sse < best[3]:
            best = (mu1, mu2, H, sse)

    mu1, mu2, H, sse = best
    if mu1 is None or mu2 is None:
        # ultimate fallback: two quantiles with H chosen to match area
        q1, q2 = np.percentile(x, [25, 75])
        mu1, mu2 = float(q1), float(q2)
        gsum = gaussian(x, mu1, sigma) + gaussian(x, mu2, sigma)
        H = float(np.dot(y, gsum) / (np.dot(gsum, gsum)+1e-18))
        sse = float(np.sum((y - H*gsum)**2))
    mu_low, mu_high = (mu1, mu2) if mu1 <= mu2 else (mu2, mu1)
    return mu_low, mu_high, H, sse

# ------------------------------ plotting helpers ------------------------------
def traces_grid(traces: np.ndarray, rows: int, cols: int, scale: float,
                annotations: list[str] | None = None,
                dotted_levels: tuple[np.ndarray,np.ndarray] | None = None,
                dotted_sym_sigma: np.ndarray | None = None,
                title: str = "") -> plt.Figure:
    B, T = traces.shape
    y_min = float(np.min(traces)); y_max = float(np.max(traces))
    pad = 0.05*(y_max - y_min + 1e-12)
    y_lo, y_hi = y_min - pad, y_max + pad
    fig_w = max(8, cols*1.2); fig_h = max(6, rows*1.0)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            ax = axes[r,c]
            ax.plot(traces[i]*scale, lw=0.4)
            if dotted_levels is not None:
                lo, hi = dotted_levels[0][i], dotted_levels[1][i]
                if np.isfinite(lo): ax.axhline(lo*scale, ls=':', color='red', lw=0.7)
                if np.isfinite(hi): ax.axhline(hi*scale, ls=':', color='red', lw=0.7)
            if dotted_sym_sigma is not None and np.isfinite(dotted_sym_sigma[i]):
                s = dotted_sym_sigma[i]*scale
                ax.axhline(+s, ls=':', color='red', lw=0.7)
                ax.axhline(-s, ls=':', color='red', lw=0.7)
            if annotations is not None:
                ax.text(0.02, 0.75, annotations[i], transform=ax.transAxes, fontsize=7,
                        bbox=dict(fc=(1,1,1,0.55), ec='none', pad=0.2))
            ax.text(0.02, 0.03, f"{r},{c}", transform=ax.transAxes, fontsize=7,
                    bbox=dict(fc=(1,1,1,0.45), ec='none', pad=0.2))
            ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel():
        ax.set_ylim(y_lo*scale, y_hi*scale)
    fig.suptitle(title)
    fig.tight_layout(rect=[0,0,1,0.97])
    return fig

def hist_grid_with_gaussians(values_list: list[np.ndarray], rows: int, cols: int,
                             centers_list: list[np.ndarray],
                             g1_list: list[np.ndarray], g2_list: list[np.ndarray],
                             title: str) -> plt.Figure:
    fig_w = max(8, cols*1.2); fig_h = max(6, rows*1.0)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            ax = axes[r,c]
            centers = centers_list[i]; values = values_list[i]
            ax.plot(centers, values, lw=0.9)          # histogram as line
            ax.fill_between(centers, 0, values, alpha=0.15)
            # overlay two Gaussians
            ax.plot(centers, g1_list[i], lw=1.0, linestyle='--')
            ax.plot(centers, g2_list[i], lw=1.0, linestyle='--')
            # hide numbers
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.02, 0.03, f"{r},{c}", transform=ax.transAxes, fontsize=7,
                    bbox=dict(fc=(1,1,1,0.45), ec='none', pad=0.2))
    fig.suptitle(title)
    fig.tight_layout(rect=[0,0,1,0.97])
    return fig

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz_path', nargs='?', help='Path to exp_block_data.npz (optional; dialog opens if omitted)')
    ap.add_argument('--bins', type=int, default=1024, help='Histogram bins for Pages 2/4')
    args = ap.parse_args()

    # Select NPZ
    npz_path = Path(args.npz_path) if args.npz_path else pick_npz_dialog()
    if not npz_path.exists():
        raise SystemExit(f"NPZ not found: {npz_path}")

    Z = np.load(npz_path, allow_pickle=True)
    if 'trace_boxes' not in Z:
        raise SystemExit("NPZ missing 'trace_boxes'")

    traces = Z['trace_boxes'][0]  # (B, T)
    rows = int(Z['rows']); cols = int(Z['cols'])
    mV = float(Z['perturb_mV']) if 'perturb_mV' in Z else 100.0

    B, T = traces.shape
    fps = 5000#T / 10.0  # fps = total frames / 10 s

    # ---- Page 2 prep: HP 0.2 Hz (linearization) ----
    hp02 = np.empty_like(traces, dtype=float)
    for i in range(B):
        hp02[i] = butter_highpass(traces[i].astype(float), cutoff_hz=0.2, fs=fps, order=1)

    # mild flatten for stable histograms (detrend + slight smooth)
    win = max(1, int(round(traces.shape[1] / 200)))
    hp02_flat = np.empty_like(hp02)
    for i in range(B):
        yd = detrend_linear(hp02[i])
        hp02_flat[i] = moving_average(yd, win)

    # ---- Page 3 prep: HP ~20 Hz residual + 60 Hz notch ----
    hp20_cut = min(20.0, 0.45*fps)  # clamp to Nyquist margin if needed
    if hp20_cut < 0.05: hp20_cut = 0.05  # safety floor
    residual = np.empty_like(traces, dtype=float)
    for i in range(B):
        r = butter_highpass(traces[i].astype(float), cutoff_hz=hp20_cut, fs=fps, order=1)
        r = notch_60(r, fs=fps, f0=60.0, Q=30.0)
        residual[i] = r

    # Per-block sigma from residual (std) — used both for S/N and for Gaussian σ in histogram fit
    sigma = np.std(residual, axis=1, ddof=1)

    # ---- Page 4 prep: fit two equal-height Gaussians to each ROI histogram (σ fixed) ----
    values_list, centers_list = [], []
    g1_list, g2_list = [], []
    c_low = np.full(B, np.nan, float); c_high = np.full(B, np.nan, float)
    H_store = np.full(B, np.nan, float)
    for i in range(B):
        ys = hp02_flat[i]
        hist, edges = np.histogram(ys, bins=args.bins)
        centers = 0.5*(edges[1:]+edges[:-1])
        mu1, mu2, H, _ = fit_two_equal_height_gaussians(hist, centers, sigma=float(sigma[i]))
        muL, muH = (mu1, mu2) if mu1 <= mu2 else (mu2, mu1)
        c_low[i], c_high[i] = muL, muH
        H_store[i] = H
        gsum1 = H * gaussian(centers, muL, float(sigma[i]))
        gsum2 = H * gaussian(centers, muH, float(sigma[i]))
        values_list.append(hist); centers_list.append(centers)
        g1_list.append(gsum1); g2_list.append(gsum2)

    # Δ from fitted means; S/N uses σ from residual
    delta = c_high - c_low
    snr = delta / (sigma + 1e-12)

    # ---- Page 1: RAW traces (μ subtracted), annotated with Δ, σ, S/N, μ_raw ----
    mu_raw = traces.mean(axis=1)
    raw_centered = traces - mu_raw[:,None]
    scale = 100.0 / max(1.0, mV)
    ann1 = [f"Δ={delta[i]:.3g}\nσ={sigma[i]:.3g}\nS/N={snr[i]:.3g}\nμ_raw={mu_raw[i]:.3g}" for i in range(B)]
    page1 = traces_grid(raw_centered, rows, cols, scale=scale, annotations=ann1,
                        dotted_levels=None, dotted_sym_sigma=None,
                        title=f"RAW traces (μ subtracted); fps={fps:.3f} Hz, scale=×{scale:.1f} (100/mV)")

    # ---- Page 2: HP 0.2 traces with fitted low/high dotted lines + Δ, σ, S/N ----
    ann2 = [f"Δ={delta[i]:.3g}\nσ={sigma[i]:.3g}\nS/N={snr[i]:.3g}" for i in range(B)]
    page2 = traces_grid(hp02, rows, cols, scale=scale, annotations=ann2,
                        dotted_levels=(c_low, c_high), dotted_sym_sigma=None,
                        title=f"Traces (High‑pass 0.2 Hz) — red dotted = fitted low/high levels (Gaussian σ from residual)")

    # ---- Page 3: Residual (HP ~20 Hz + 60 Hz notch) with ±σ dotted lines ----
    ann3 = [f"σ={sigma[i]:.3g}" for i in range(B)]
    page3 = traces_grid(residual, rows, cols, scale=scale, annotations=ann3,
                        dotted_levels=None, dotted_sym_sigma=sigma,
                        title=f"Residual (High‑pass {hp20_cut:.2f} Hz + 60 Hz notch) — σ from residual")

    # ---- Page 4: Histograms with two equal-height Gaussians overlaid ----
    page4 = hist_grid_with_gaussians(values_list, rows, cols, centers_list, g1_list, g2_list,
                                     title="Histogram (HP 0.2 Hz) with two equal‑height Gaussians (σ fixed from residual)")

    # save next to NPZ
    pdf_path = npz_path.with_name(npz_path.stem + "_npz_dialog_v4_report.pdf")
    with PdfPages(pdf_path) as pdf:
        for F in [page1, page2, page3, page4]:
            pdf.savefig(F)
    print(f"[Report] Saved → {pdf_path}")

if __name__ == '__main__':
    main()
