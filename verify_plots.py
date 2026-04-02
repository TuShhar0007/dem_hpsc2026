"""
verify_plots.py  –  DEM Solver Verification & Visualisation
HPSC 2026 Assignment 1

Reads CSV files produced by dem_solver.cpp and generates
publication-quality Matplotlib figures.

Usage:
    python verify_plots.py

All CSV files must be in the same directory.

LLM usage note: Claude (Anthropic) used to scaffold this script;
all equations and axis labels have been manually checked against
the assignment specification.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless rendering (safe for HPC nodes)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import sys

# ── Publication style ─────────────────────────────────────────────────────────
FONTSIZE   = 11
LINEWIDTH  = 1.6
MARKERSIZE = 4

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : FONTSIZE,
    "axes.labelsize"   : FONTSIZE,
    "axes.titlesize"   : FONTSIZE,
    "legend.fontsize"  : FONTSIZE - 1,
    "xtick.labelsize"  : FONTSIZE - 1,
    "ytick.labelsize"  : FONTSIZE - 1,
    "lines.linewidth"  : LINEWIDTH,
    "lines.markersize" : MARKERSIZE,
    "axes.grid"        : True,
    "grid.alpha"       : 0.35,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
})

DPI    = 300
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Helper to save figures
def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=DPI)
    print(f"  ✓ Saved {path}")
    plt.close(fig)


# ── Physical constants ────────────────────────────────────────────────────────
G  = 9.81   # m/s²
Z0 = 5.0    # initial height used in free-fall test


# =============================================================================
# 1.  FREE FALL: numerical vs analytical trajectory
# =============================================================================

def plot_free_fall():
    fname = "free_fall.csv"
    if not os.path.exists(fname):
        print(f"[skip] {fname} not found – run dem_solver first")
        return

    df = pd.read_csv(fname)
    t       = df["t"].values
    z_num   = df["z_num"].values
    z_ana   = df["z_ana"].values
    vz_num  = df["vz_num"].values
    vz_ana  = df["vz_ana"].values

    # ── Figure 1a: z(t) comparison ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)

    ax = axes[0]
    ax.plot(t, z_ana, "k-",  label="Analytical", zorder=3)
    ax.plot(t, z_num, "r--", label="Numerical",  zorder=2)
    ax.set_ylabel(r"$z$ (m)")
    ax.set_title("Free Fall: Position")
    ax.legend()

    ax = axes[1]
    ax.plot(t, vz_ana, "k-",  label="Analytical")
    ax.plot(t, vz_num, "r--", label="Numerical")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$v_z$ (m/s)")
    ax.set_title("Free Fall: Velocity")
    ax.legend()

    fig.tight_layout()
    savefig(fig, "free_fall_trajectory.pdf")

    # ── Figure 1b: absolute error vs time ───────────────────────────────────
    err = np.abs(z_num - z_ana)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.semilogy(t, err, "b-")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$|z_{\rm num} - z_{\rm ana}|$ (m)")
    ax.set_title("Free Fall: Absolute Positional Error vs Time")
    fig.tight_layout()
    savefig(fig, "free_fall_error_time.pdf")


# =============================================================================
# 2.  ERROR vs TIME-STEP  (convergence study)
# =============================================================================

def plot_convergence():
    fname = "free_fall_error.csv"
    if not os.path.exists(fname):
        print(f"[skip] {fname} not found – run dem_solver first")
        return

    df = pd.read_csv(fname)
    dt   = df["dt"].values
    rmse = df["L2_error"].values

    # Sort by dt
    idx  = np.argsort(dt)
    dt   = dt[idx];  rmse = rmse[idx]

    # Fit a power law  err ~ C * dt^p
    log_dt   = np.log10(dt)
    log_rmse = np.log10(rmse)
    coeffs   = np.polyfit(log_dt, log_rmse, 1)
    order    = coeffs[0]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(dt, rmse, "bo-", label="RMSE")
    # Reference line
    dt_ref = np.array([dt.min(), dt.max()])
    ax.loglog(dt_ref, 10**coeffs[1] * dt_ref**order, "k--",
              label=fr"Slope $\approx {order:.2f}$  (expected 1)")
    ax.set_xlabel(r"$\Delta t$ (s)")
    ax.set_ylabel("RMSE of $z(t)$ (m)")
    ax.set_title("Convergence Study: Error vs Time-step")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "convergence.pdf")
    print(f"    Measured convergence order: {order:.3f}  (semi-implicit Euler → 1st order)")


# =============================================================================
# 3.  BOUNCING PARTICLE
# =============================================================================

def plot_bounce():
    fname = "bounce.csv"
    if not os.path.exists(fname):
        print(f"[skip] {fname} not found – run dem_solver first")
        return

    df = pd.read_csv(fname)
    t  = df["t"].values
    z  = df["z"].values
    KE = df["KE"].values

    # ── Figure 3a: height vs time ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)

    ax = axes[0]
    ax.plot(t, z, "b-", lw=1.2)
    ax.axhline(0.1, color="gray", ls=":", lw=1, label="Particle radius = 0.1 m")
    ax.set_ylabel(r"$z$ (m)")
    ax.set_title("Bouncing Particle: Height vs Time")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(t, KE, "r-", lw=1.2)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel("Kinetic Energy (J)")
    ax.set_title("Bouncing Particle: Kinetic Energy vs Time")

    fig.tight_layout()
    savefig(fig, "bounce.pdf")

    # ── Figure 3b: rebound heights ───────────────────────────────────────────
    R = 0.1
    # Find local maxima in z above the floor contact radius
    from scipy.signal import argrelmax
    peaks_idx = argrelmax(z, order=10)[0]
    peaks_idx = peaks_idx[z[peaks_idx] > R + 0.05]   # clear of floor

    if len(peaks_idx) > 1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        rebound = z[peaks_idx]
        bounce_n = np.arange(1, len(rebound)+1)
        ax.plot(bounce_n, rebound, "gs-")
        ax.set_xlabel("Bounce number")
        ax.set_ylabel("Peak height (m)")
        ax.set_title("Rebound Height vs Bounce Number")
        fig.tight_layout()
        savefig(fig, "bounce_heights.pdf")
    else:
        print("    Not enough bounces detected – increase simulation time")


# =============================================================================
# 4.  KINETIC ENERGY vs TIME  (multi-particle)
# =============================================================================

def plot_energy(N_vals=(200, 1000, 5000)):
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False

    for N in N_vals:
        for method in ("serial", "omp", "neighbor"):
            fname = f"sim_N{N}_{method}_energy.csv"
            if not os.path.exists(fname):
                continue
            df = pd.read_csv(fname, header=0, names=["t", "KE"])
            lbl = f"N={N} ({method})"
            ax.plot(df["t"], df["KE"], label=lbl, lw=1.2)
            plotted = True
            break   # one method per N is enough for the energy plot

    if not plotted:
        print("[skip] No energy CSV files found")
        return

    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel("Kinetic Energy (J)")
    ax.set_title("Kinetic Energy vs Time")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "kinetic_energy.pdf")


# =============================================================================
# 5.  SCALING STUDY: Speedup & Efficiency
# =============================================================================

def plot_scaling():
    fname = "scaling.csv"
    if not os.path.exists(fname):
        print(f"[skip] {fname} not found – run  ./dem scaling  first")
        return

    df = pd.read_csv(fname)

    fig = plt.figure(figsize=(10, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # Filter OMP rows only
    omp = df[df["method"] == "OMP"].copy()
    N_vals = sorted(omp["N"].unique())

    # For speedup we need the serial baseline T1
    serial = df[df["method"] == "SERIAL"][["N", "time_s"]].rename(
        columns={"time_s": "T1"})

    omp = omp.merge(serial, on="N")
    omp["speedup"]    = omp["T1"] / omp["time_s"]
    omp["efficiency"] = omp["speedup"] / omp["threads"]

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(N_vals)))

    # ── Speedup ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    for i, N in enumerate(N_vals):
        sub = omp[omp["N"] == N].sort_values("threads")
        ax1.plot(sub["threads"], sub["speedup"], "o-",
                 color=colors[i], label=f"N={N}")
    p_max = omp["threads"].max()
    ax1.plot([1, p_max], [1, p_max], "k--", lw=1, label="Ideal")
    ax1.set_xlabel("Threads $p$")
    ax1.set_ylabel(r"Speedup $S(p) = T_1 / T_p$")
    ax1.set_title("OpenMP Speedup")
    ax1.legend(fontsize=9)

    # ── Efficiency ───────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for i, N in enumerate(N_vals):
        sub = omp[omp["N"] == N].sort_values("threads")
        ax2.plot(sub["threads"], sub["efficiency"], "s--",
                 color=colors[i], label=f"N={N}")
    ax2.axhline(1.0, color="k", lw=1, ls="--", label="Ideal")
    ax2.set_xlabel("Threads $p$")
    ax2.set_ylabel(r"Efficiency $E(p) = S(p)/p$")
    ax2.set_title("OpenMP Efficiency")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    savefig(fig, "scaling_speedup_efficiency.pdf")

    # ── Runtime table ─────────────────────────────────────────────────────────
    print("\n  Runtime table (seconds):")
    pivot = omp.pivot_table(index="N", columns="threads",
                            values="time_s", aggfunc="mean")
    print(pivot.to_string())

    # ── Neighbour-search comparison ────────────────────────────────────────────
    nbr   = df[df["method"] == "NEIGHBOR"][["N", "time_s"]].rename(
        columns={"time_s": "T_nbr"})
    cmp   = serial.merge(nbr, on="N")
    cmp["speedup_nbr"] = cmp["T1"] / cmp["T_nbr"]

    fig2, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(cmp["N"], cmp["T1"],        "ko-", label="O(N²) serial")
    ax.plot(cmp["N"], cmp["T_nbr"],     "b^-", label="Neighbour search")
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime: Brute-force vs Neighbour Search")
    ax.legend()
    fig2.tight_layout()
    savefig(fig2, "neighbor_runtime.pdf")


# =============================================================================
# 6.  PARTICLE CONFIGURATION SNAPSHOT  (3D scatter)
# =============================================================================

def plot_snapshot(N=200, t_idx=-1):
    for method in ("serial", "omp", "neighbor"):
        fname = f"sim_N{N}_{method}_traj.csv"
        if os.path.exists(fname):
            break
    else:
        print(f"[skip] No trajectory CSV for N={N}")
        return

    df   = pd.read_csv(fname, header=0,
                       names=["t","id","x","y","z","vx","vy","vz"])
    times = sorted(df["t"].unique())
    if len(times) == 0:
        return

    snap_t  = times[t_idx]
    snap_df = df[df["t"] == snap_t]

    speed = np.sqrt(snap_df["vx"]**2 + snap_df["vy"]**2 + snap_df["vz"]**2)

    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(snap_df["x"], snap_df["y"], snap_df["z"],
                     c=speed, cmap="plasma", s=10, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Speed (m/s)", fontsize=9)
    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$ (m)"); ax.set_zlabel("$z$ (m)")
    ax.set_title(f"Particle Configuration  t = {snap_t:.3f} s  (N={N})")
    fig.tight_layout()
    savefig(fig, f"snapshot_N{N}_t{snap_t:.3f}.pdf")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("DEM Solver – Verification & Plotting")
    print("=" * 50)

    print("\n[1] Free Fall")
    plot_free_fall()

    print("\n[2] Convergence Study (error vs Δt)")
    plot_convergence()

    print("\n[3] Bouncing Particle")
    plot_bounce()

    print("\n[4] Kinetic Energy")
    plot_energy()

    print("\n[5] Scaling Study")
    plot_scaling()

    print("\n[6] Configuration Snapshot")
    for Nv in [200, 1000]:
        plot_snapshot(Nv)

    print("\nAll figures written to ./figures/")
