"""
FeatureExtraction.py
====================
Multi-domain feature extraction for integrated EEG-fNIRS windows. Implements
the six feature groups benchmarked in the paper:

  TDT  temporal dynamics : mean, RMS, skew, kurtosis, ZCR, Hjorth
  TDS  statistical       : mean, std, skew, kurtosis
  TDM  morphological     : slope, peak-time, peak-to-peak, area
  FDS  spectral shape    : centroid, bandwidth, flatness, roll-off, SEF, contrast
  FDP  band power        : alpha/beta/theta/gamma power + ratios
  TFD  time-frequency    : Morlet wavelet energy + MFCC-like cepstral coeffs

Each function takes a windowed signal x of shape (n_frames, n_samples).
Pure NumPy/SciPy, so it runs without TensorFlow.
"""
import numpy as np
from scipy import signal, stats

EPS = 1e-12
_TRAP = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 1.x/2.x


def tdt(x):
    mean = x.mean(1)
    rms = np.sqrt((x ** 2).mean(1))
    skew = stats.skew(x, axis=1)
    kurt = stats.kurtosis(x, axis=1)
    zcr = (np.abs(np.diff(np.sign(x), axis=1)) > 0).mean(1)
    d1, d2 = np.diff(x, axis=1), np.diff(x, n=2, axis=1)
    v0, v1, v2 = x.var(1) + EPS, d1.var(1) + EPS, d2.var(1) + EPS
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return np.stack([mean, rms, skew, kurt, zcr, v0, mobility, complexity], axis=1)


def tds(x):
    return np.stack([x.mean(1), x.std(1),
                     stats.skew(x, axis=1), stats.kurtosis(x, axis=1)], axis=1)


def tdm(x):
    t = np.arange(x.shape[1])
    slope = np.array([np.polyfit(t, row, 1)[0] for row in x])
    peak_time = x.argmax(1).astype(float)
    p2p = x.max(1) - x.min(1)
    area = _TRAP(np.abs(x), axis=1)
    return np.stack([slope, peak_time, p2p, area], axis=1)


def fds(x, fs=200):
    f, p = signal.welch(x, fs=fs, axis=1); p = p + EPS
    psum = p.sum(1)
    centroid = (f * p).sum(1) / psum
    bandwidth = np.sqrt(((f - centroid[:, None]) ** 2 * p).sum(1) / psum)
    flatness = stats.gmean(p, axis=1) / p.mean(1)
    cumsum = np.cumsum(p, axis=1) / psum[:, None]
    rolloff = f[(cumsum >= 0.85).argmax(1)]
    sef = f[(cumsum >= 0.95).argmax(1)]
    contrast = p.max(1) / (p.mean(1) + EPS)
    return np.stack([centroid, bandwidth, flatness, rolloff, sef, contrast], axis=1)


def fdp(x, fs=200):
    bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
    f, p = signal.welch(x, fs=fs, axis=1)
    pw = {n: p[:, (f >= lo) & (f < hi)].sum(1) for n, (lo, hi) in bands.items()}
    feats = list(pw.values()) + [pw["alpha"] / (pw["beta"] + EPS),
                                 pw["theta"] / (pw["alpha"] + EPS)]
    return np.stack(feats, axis=1)


def tfd(x, fs=200, n_mfcc=13):
    freqs = np.linspace(4, 40, 8)
    n = min(64, x.shape[1])
    energy = []
    for fr in freqs:
        if hasattr(signal, "morlet2"):
            w = signal.morlet2(n, w=5, s=fs / (2 * np.pi * fr))
        else:                                   # scipy >= 1.13
            t = np.linspace(-n / 2, n / 2, n); s = fs / (2 * np.pi * fr)
            w = np.exp(2j * np.pi * (t / s) * 5) * np.exp(-(t ** 2) / (2 * s ** 2))
        energy.append(np.abs(np.array(
            [np.convolve(row, w, mode="same") for row in x])).mean(1))
    morlet = np.stack(energy, axis=1)
    f, p = signal.welch(x, fs=fs, axis=1)
    mfcc = np.fft.fft(np.log(p + EPS), axis=1).real[:, :n_mfcc]
    return np.concatenate([morlet, mfcc], axis=1)


GROUPS = {"TDT": tdt, "TDS": tds, "TDM": tdm, "FDS": fds, "FDP": fdp, "TFD": tfd}


def extract(x, groups=("FDS", "TDT"), fs=200):
    """Concatenate the requested groups and z-normalize across frames."""
    feats = [GROUPS[g](x, fs) if g in ("FDS", "FDP", "TFD") else GROUPS[g](x)
             for g in groups]
    out = np.concatenate(feats, axis=1)
    return (out - out.mean(0, keepdims=True)) / (out.std(0, keepdims=True) + EPS)
