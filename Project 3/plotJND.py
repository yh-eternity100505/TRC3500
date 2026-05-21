"""
Plot the BPM detection data and fit a logistic (sigmoid) psychometric function.

Detection rate p(BPM) = 1 / (1 + exp(k * (BPM - BPM50)))
  • BPM50 — 50% detection threshold (the "halfway" point)
  • k     — slope at threshold (larger k → sharper transition)

JND is defined here as half the inter-quartile range of the psychometric
function:  JND = (BPM25 - BPM75) / 2

Inputs : bpm_test_results.csv
Outputs: jnd_psychometric.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CSV_IN  = "bpm_test_results.csv"
PNG_OUT = "jnd_psychometric.png"

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
bpm        = df["target_bpm"].to_numpy(dtype=float)
detected   = df["detected_breaths"].to_numpy(dtype=float)
expected   = df["expected_breaths"].to_numpy(dtype=float)
p_detect   = detected / expected            # detection rate in [0, 1]

print("Raw data:")
for b, d, e, p in zip(bpm, detected, expected, p_detect):
    print(f"  {b:5.0f} BPM  →  {int(d):2d}/{int(e):2d} detected   "
          f"p = {p:.2f}")

# ── Logistic (decreasing) ────────────────────────────────────────────────────
def logistic(x, bpm50, k):
    return 1.0 / (1.0 + np.exp(k * (x - bpm50)))

p0 = (50.0, 0.2)        # initial guess: half-point ≈ 50 BPM, moderate slope
popt, _ = curve_fit(logistic, bpm, p_detect, p0=p0, maxfev=10000)
bpm50_fit, k_fit = popt
print(f"\nLogistic fit:  BPM50 = {bpm50_fit:.2f},  k = {k_fit:.3f}")

# Inverse: given probability, find BPM
def bpm_at(p):
    # p = 1 / (1 + exp(k*(x - x0)))  →  x = x0 + log(1/p - 1) / k
    return bpm50_fit + np.log(1.0 / p - 1.0) / k_fit

bpm25 = bpm_at(0.25)
bpm50 = bpm_at(0.50)
bpm75 = bpm_at(0.75)
jnd   = abs(bpm25 - bpm75) / 2.0

print(f"BPM at 75% detection : {bpm75:.2f}")
print(f"BPM at 50% detection : {bpm50:.2f}")
print(f"BPM at 25% detection : {bpm25:.2f}")
print(f"JND (half IQR)       : {jnd:.2f} BPM")

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans",
                     "axes.edgecolor": "#333",
                     "axes.linewidth": 0.8})

fig, ax = plt.subplots(figsize=(9, 5.5))

# Smooth curve
x_grid = np.linspace(min(bpm) - 5, max(bpm) + 10, 400)
y_grid = logistic(x_grid, *popt)
ax.plot(x_grid, y_grid, color="#1f77b4", linewidth=2.2,
        label=f"Logistic fit:  $p = 1\\,/\\,(1 + e^{{{k_fit:.3f}\\,(x-{bpm50_fit:.1f})}})$",
        zorder=3)

# Data points (size proportional to expected count — here all = 10)
ax.scatter(bpm, p_detect, s=110, color="#d62728", edgecolor="white",
           linewidth=1.4, zorder=5, label="Measured data (10 breaths each)")

# Annotate each measured point
for b, p, d in zip(bpm, p_detect, detected):
    ax.annotate(f"{int(d)}/10", xy=(b, p), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=9,
                color="#333")

# 25 / 50 / 75 % threshold lines
for p_thresh, b_thresh, colour, ls in [
    (0.25, bpm25, "#888", ":"),
    (0.50, bpm50, "#ff8c00", "-"),
    (0.75, bpm75, "#888", ":"),
]:
    ax.axhline(p_thresh, color=colour, linestyle=ls, linewidth=1.0, zorder=1)
    ax.axvline(b_thresh, color=colour, linestyle=ls, linewidth=1.0, zorder=1)
    ax.plot(b_thresh, p_thresh, "o", color=colour, markersize=7,
            markeredgecolor="white", zorder=6)

# JND span shading between 25% and 75% points
ax.axvspan(min(bpm25, bpm75), max(bpm25, bpm75),
           color="#ffcc80", alpha=0.25, zorder=0,
           label=f"JND region (25–75 %):  JND = {jnd:.2f} BPM")

# Axis cosmetics
ax.set_xlim(min(bpm) - 5, max(bpm) + 10)
ax.invert_xaxis()                     # high BPM on the left, low BPM on the right
ax.set_ylim(-0.05, 1.10)
ax.set_xlabel("Target breath rate  [BPM]", fontsize=11)
ax.set_ylabel("Detection probability  $p$", fontsize=11)
ax.set_title("Psychometric Curve — Rubber Sensor Breath Detection",
             fontsize=13, fontweight="bold")
ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

# Stats box
txt = (f"BPM₇₅ = {bpm75:.2f}\n"
       f"BPM₅₀ = {bpm50:.2f}\n"
       f"BPM₂₅ = {bpm25:.2f}\n"
       f"JND = {jnd:.2f} BPM\n"
       f"k = {k_fit:.3f}")
ax.text(0.985, 0.03, txt,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.45",
                  facecolor="#f5f5f5", edgecolor="#bbb"))

plt.tight_layout()
plt.savefig(PNG_OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nFigure → {PNG_OUT}")