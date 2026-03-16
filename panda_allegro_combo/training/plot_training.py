"""
plot_training.py

Generates a 4-panel publication-quality figure from the Phase 1 (reach)
and Phase 2 (reach+hold) training logs.

Usage:
    python plot_training.py --reach reach_log.txt --hold hold_log.txt

If you only have one log, omit the other flag.
Outputs: training_curves.png  (saved next to this script)

Log format expected (one line per LOG_INTERVAL):
  Update    10 | Rew   91.29 | Succ 100.0% | PG  0.000 | VF  15.9 | dPalm 0.338 | EpLen 86 | ...
"""

import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import os

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

REACH_COLOR = "#2563EB"   # blue
HOLD_COLOR  = "#16A34A"   # green
ALPHA_RAW   = 0.25
SMOOTH_WIN  = 20          # rolling average window

# ── Parsing ───────────────────────────────────────────────────────────────────
LOG_RE = re.compile(
    r"Update\s+(\d+)\s*\|"
    r"\s*Rew\s+([\-\d\.]+)\s*\|"
    r"\s*Succ\s+([\d\.]+)%\s*\|"
    r"\s*PG\s+([\-\d\.]+)\s*\|"
    r"\s*VF\s+([\-\d\.]+)\s*\|"
    r"\s*dPalm\s+([\d\.]+)\s*\|"
    r"\s*EpLen\s+([\d\.]+)"
)

def parse_log(path):
    updates, rewards, succs, pgs, vfs, dpalms, eplens = [], [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                updates.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
                succs.append(float(m.group(3)))
                pgs.append(float(m.group(4)))
                vfs.append(float(m.group(5)))
                dpalms.append(float(m.group(6)))
                eplens.append(float(m.group(7)))
    return {
        "update": np.array(updates),
        "reward": np.array(rewards),
        "success": np.array(succs),
        "pg": np.array(pgs),
        "vf": np.array(vfs),
        "dpalm": np.array(dpalms),
        "eplen": np.array(eplens),
    }

def smooth(x, w):
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")

def smooth_x(updates, w):
    """Return x-axis aligned to smoothed output (valid convolution shrinks array)."""
    if len(updates) < w:
        return updates
    return updates[w-1:]

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(reach=None, hold=None, out="training_curves.png"):
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#FAFAFA")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    panels = [
        ("reward",  "Episode Return",         True,  None),
        ("dpalm",   "Mean Palm–Brick Dist (m)", False, None),
        ("eplen",   "Episode Length (steps)",   False, None),
        ("vf",      "Value Function Loss",       False, None),
    ]

    datasets = []
    if reach is not None:
        datasets.append(("Phase 1 · Reach", reach, REACH_COLOR))
    if hold is not None:
        datasets.append(("Phase 2 · Hold", hold, HOLD_COLOR))

    for ax, (key, ylabel, show_legend, _) in zip(axes, panels):
        for label, d, color in datasets:
            x = d["update"]
            y = d[key]
            # raw (faint)
            ax.plot(x, y, color=color, alpha=ALPHA_RAW, linewidth=0.8)
            # smoothed
            xs = smooth_x(x, SMOOTH_WIN)
            ys = smooth(y, SMOOTH_WIN)
            ax.plot(xs, ys, color=color, label=label)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training Update")
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.set_facecolor("#F8F9FA")

    # Titles
    axes[0].set_title("Episode Return")
    axes[1].set_title("Palm–Brick Distance")
    axes[2].set_title("Episode Length")
    axes[3].set_title("Value Function Loss")

    # Add success annotations on reward panel
    for label, d, color in datasets:
        final_succ = d["success"][-10:].mean()
        axes[0].annotate(
            f"{label}\nFinal success: {final_succ:.1f}%",
            xy=(d["update"][-1], d["reward"][-1]),
            xytext=(-80, 15),
            textcoords="offset points",
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    # Hold curriculum ticks on Phase 2 plots
    if hold is not None:
        hold_d = hold
        prev_h = None
        anneal_xs = []
        # re-parse for anneal events
        with open(args.hold) as f:
            for line in f:
                m = re.search(r"Hold requirement increased: (\d+) → (\d+)", line)
                if m:
                    anneal_xs.append(int(m.group(2)))
        for ax in axes:
            for hx in anneal_xs:
                ax.axvline(hx * 10, color=HOLD_COLOR, alpha=0.15, linewidth=0.8, linestyle=":")

    # Legend on reward panel
    if len(datasets) > 1:
        axes[0].legend(loc="lower right", fontsize=9, framealpha=0.8)

    fig.suptitle(
        "Panda-Lego PPO Training Progress\nPhases 1 & 2 · CPU Sim + GPU Networks",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reach", help="Path to reach training log (Phase 1)")
    parser.add_argument("--hold",  help="Path to hold training log (Phase 2)")
    parser.add_argument("--out",   default="training_curves.png")
    args = parser.parse_args()

    if not args.reach and not args.hold:
        parser.error("Provide at least --reach or --hold log file.")

    reach_data = parse_log(args.reach) if args.reach else None
    hold_data  = parse_log(args.hold)  if args.hold  else None

    plot(reach_data, hold_data, out=args.out)