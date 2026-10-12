"""Draw the figures for the IFAS paper.

    python paper/figures/make_figures.py

Writes vector PDFs next to this file. Every number plotted here is listed in
DATA below with its source, so a figure can be checked against the experiment
notes without opening a plotting tool.

Sources
    doc       stated in docs/hyperparameter-analysis/**.md
    plot      read back from the original plot image in the same folder. The
              value is not written in the notes, so it carries the reading
              error of the image (about 0.0003 on the accuracy axes).
    notebook  printed in the stored outputs of age_classifier_v3.ipynb
    runtime   measured by scripts/benchmark_stages.py and saved in runtime.json
"""
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------- data
AGES = ["16 to 30", "31 to 45", "46 to 60", "Above 60"]

DATA = {
    # Validation accuracy per epoch. Epoch 0 is the end of the linear probe.
    "lr_curves": {                                                    # plot, finals: doc
        "1e-4": [0.787, 0.918, 0.979, 0.992, 0.992],
        "1e-5": [0.787, 0.933, 0.970, 0.981, 0.983],
        "1e-7": [0.787, 0.794, 0.803, 0.803, 0.804],
    },
    # Final validation accuracy (%) against linear probe epochs.           doc
    "probe": {0: 96.80, 1: 97.60, 2: 97.20, 3: 97.35, 4: 97.80},
    # Final validation accuracy (%) against weight decay.
    "wd_acc": {"0": 97.39, "0.001": 97.64, "0.01": 97.53, "0.05": 97.57,   # 0.01-0.1: plot
               "0.1": 97.57, "0.3": 97.71, "0.5": 97.50},                  # rest: doc
    # Train accuracy minus validation accuracy (points) at the final epoch.
    "wd_gap": {"0": 2.14, "0.001": 1.90, "0.01": 1.99, "0.05": 1.96,       # 0, 0.3, 0.5: doc
               "0.1": 1.80, "0.3": 1.67, "0.5": 1.88},                     # rest: plot
    # Final validation accuracy (%) against warmup ratio.
    "warmup": {"0": 96.86, "0.05": 97.61, "0.1": 97.10, "0.2": 97.68,      # 0.05, 0.2: doc
               "0.3": 97.43, "0.4": 97.21},                                # rest: plot
    # Age model: average screening mA on a 30% subsample.                 all doc
    "regions": {2: 0.7155, 3: 0.7327, 4: 0.7457, 5: 0.7432, 6: 0.7404, 7: 0.7420},
    "color_bins": {8: 0.7417, 16: 0.7457, 24: 0.7305, 32: 0.7220},
    "lbp_points": {8: 0.7455, 16: 0.7493, 24: 0.7486},
    "class_weight": {"none": 0.7432, "balanced": 0.7493},

    # Age data and the released age model on the test split.         all notebook
    "age_counts": [5448, 5804, 1840, 1149],
    "confusion": [[1770, 309, 87, 13],          # rows: true group, columns: predicted
                  [535, 1671, 96, 20],
                  [98, 90, 536, 12],
                  [21, 34, 15, 390]],
    "precision": [0.7302, 0.7942, 0.7302, 0.8966],
    "recall": [0.8123, 0.7196, 0.7283, 0.8478],
    "f1": [0.7691, 0.7551, 0.7293, 0.8715],
    "val_mA": [0.7996, 0.7823, 0.7897, 0.9108],
    "test_mA": [0.8107, 0.7888, 0.8312, 0.9102],
}

# ---------------------------------------------------------------------- style
INK = "#1b1f3b"
GREY = "#8a8fa8"
RULE = "#d9dbe6"
PICK = "#2a6fdc"        # the value that was selected, and the learned network
WARM = "#e0602a"        # the classical model

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": RULE,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "savefig.pad_inches": 0.02,
})


def tag(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", pad=6, fontweight="bold")


def dots(ax, labels, values, pick, fmt="{:.2f}", pad=0.12):
    """One dot per setting, joined by a thin line; the chosen setting is filled."""
    x = range(len(labels))
    ax.plot(x, values, color=GREY, lw=0.8, zorder=2)
    for i, (lab, v) in enumerate(zip(labels, values)):
        chosen = lab == pick
        ax.plot(i, v, "o", ms=4.6, mfc=PICK if chosen else "white",
                mec=PICK if chosen else INK, mew=0.9, zorder=3)
        if chosen:
            ax.annotate(fmt.format(v), (i, v), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", color=PICK, fontweight="bold", fontsize=7.5)
    ax.set_xticks(list(x), [str(l) for l in labels])
    lo, hi = min(values), max(values)
    span = hi - lo
    ax.set_ylim(lo - span * pad, hi + span * (pad + 0.22))
    ax.margins(x=0.12)


# -------------------------------------------------------------------- figures
def gender():
    fig = plt.figure(figsize=(7.16, 3.55))
    gs = fig.add_gridspec(2, 6, hspace=0.72, wspace=1.6,
                          left=0.07, right=0.995, top=0.93, bottom=0.115)

    ax = fig.add_subplot(gs[0, :3])
    colour = {"1e-4": WARM, "1e-5": PICK, "1e-7": GREY}
    names = {"1e-4": r"$10^{-4}$", "1e-5": r"$10^{-5}$ (chosen)", "1e-7": r"$10^{-7}$"}
    for key, ys in DATA["lr_curves"].items():
        ax.plot(range(5), ys, "-", color=colour[key], lw=1.2, marker="o", ms=3.6, mfc="white", mew=0.9)
    ax.annotate(names["1e-4"], (4, 0.992), xytext=(-4, 5), textcoords="offset points", ha="right",
                va="bottom", color=WARM, fontsize=7.5)
    ax.annotate(names["1e-5"], (4, 0.983), xytext=(-4, -6), textcoords="offset points", ha="right",
                va="top", color=PICK, fontsize=7.5)
    ax.annotate(names["1e-7"], (4, 0.804), xytext=(-4, 5), textcoords="offset points", ha="right",
                va="bottom", color=GREY, fontsize=7.5)
    ax.set_ylim(0.76, 1.03)
    ax.set_xticks(range(5))
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Validation accuracy")
    tag(ax, "a", "Backbone learning rate")

    ax = fig.add_subplot(gs[0, 3:])
    p = DATA["probe"]
    dots(ax, list(p), list(p.values()), 4)
    ax.set_xlabel("Linear probe epochs")
    ax.set_ylabel("Final val. accuracy (%)")
    tag(ax, "b", "Length of the linear probe")

    ax = fig.add_subplot(gs[1, :2])
    w = DATA["wd_acc"]
    dots(ax, list(w), list(w.values()), "0.3")
    ax.set_xlabel("Weight decay")
    ax.set_ylabel("Final val. accuracy (%)")
    tag(ax, "c", "Weight decay, accuracy")

    ax = fig.add_subplot(gs[1, 2:4])
    g = DATA["wd_gap"]
    dots(ax, list(g), list(g.values()), "0.3")
    ax.set_xlabel("Weight decay")
    ax.set_ylabel("Train minus val. (points)")
    tag(ax, "d", "Weight decay, gap")

    ax = fig.add_subplot(gs[1, 4:])
    u = DATA["warmup"]
    dots(ax, list(u), list(u.values()), "0.2")
    ax.set_xlabel("Warmup ratio")
    ax.set_ylabel("Final val. accuracy (%)")
    tag(ax, "e", "Warmup ratio")

    for a in fig.axes[2:]:
        a.tick_params(axis="x", labelsize=6.4)
    fig.savefig(OUT / "gender_sensitivity.pdf")
    plt.close(fig)


def age_sensitivity():
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95), sharey=True,
                             gridspec_kw={"width_ratios": [6, 4, 3, 2.9], "wspace": 0.12,
                                          "left": 0.075, "right": 0.985, "top": 0.86, "bottom": 0.225})
    panels = [
        ("regions", "Body regions", "Horizontal regions", 4),
        ("color_bins", "Colour bins", "Bins per channel", 16),
        ("lbp_points", "LBP points", "Sampling points", 16),
        ("class_weight", "Class weight", "SVM weighting", "balanced"),
    ]
    for ax, letter, (key, title, xlabel, pick) in zip(axes, "abcd", panels):
        d = DATA[key]
        dots(ax, list(d), list(d.values()), pick, fmt="{:.4f}")
        ax.set_xlabel(xlabel)
        tag(ax, letter, title)
    axes[3].margins(x=0.42)
    # One shared scale, so the panels show which setting matters most.
    axes[0].set_ylim(0.712, 0.756)
    axes[0].set_ylabel("Average mA (screening)")
    fig.savefig(OUT / "age_sensitivity.pdf")
    plt.close(fig)


def schedule():
    """Learning rate over Stage 1: each phase warms up, then decays on a cosine."""
    def curve(peak, epochs, warm=0.2, n=400):
        x = np.linspace(0, 1, n)
        y = np.where(x < warm, x / warm, 0.5 * (1 + np.cos(math.pi * (x - warm) / (1 - warm))))
        return x * epochs, peak * y

    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.9, bottom=0.235)
    xa, ya = curve(1e-3, 4)
    xb, yb = curve(1e-5, 10)
    ax.plot(xa, ya, color=WARM, lw=1.3)
    ax.plot(4 + xb, yb, color=PICK, lw=1.3)
    ax.axvline(4, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_ylim(2e-8, 3e-2)
    ax.set_xlim(0, 14)
    ax.set_xticks(range(0, 15, 2))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.text(2, 1.8e-3, "Phase A: head only", ha="center", va="bottom", color=WARM, fontsize=7.5)
    ax.text(9, 2.4e-5, "Phase B: whole network", ha="center", va="bottom", color=PICK, fontsize=7.5)
    fig.savefig(OUT / "lr_schedule.pdf")
    plt.close(fig)


def age_data():
    counts = DATA["age_counts"]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(3.45, 1.6))
    fig.subplots_adjust(left=0.2, right=0.9, top=0.97, bottom=0.25)
    y = range(len(AGES))
    ax.barh(y, counts, color=WARM, height=0.56)
    for i, n in enumerate(counts):
        ax.text(n + 90, i, f"{n:,}  ({100 * n / total:.1f}%)", va="center", fontsize=7.5)
    ax.set_yticks(list(y), AGES)
    ax.invert_yaxis()
    ax.set_xlim(0, 7900)
    ax.set_xlabel("Images in PETA")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    fig.savefig(OUT / "age_data.pdf")
    plt.close(fig)


def age_test():
    cm = np.array(DATA["confusion"], dtype=float)
    share = cm / cm.sum(axis=1, keepdims=True)
    fig = plt.figure(figsize=(7.16, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.25, 1.25], wspace=0.42,
                          left=0.085, right=0.995, top=0.88, bottom=0.2)

    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("ifas", ["#ffffff", WARM])
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    for i in range(4):
        for j in range(4):
            ax.text(j, i - 0.12, f"{100 * share[i, j]:.0f}%", ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if i == j else "normal",
                    color="white" if share[i, j] > 0.55 else INK)
            ax.text(j, i + 0.24, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=6,
                    color="white" if share[i, j] > 0.55 else GREY)
    short = ["16-30", "31-45", "46-60", "61+"]
    ax.set_xticks(range(4), short)
    ax.set_yticks(range(4), short)
    ax.set_xlabel("Predicted group")
    ax.set_ylabel("True group")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    tag(ax, "a", "Confusion on the test split")

    ax = fig.add_subplot(gs[1])
    x = np.arange(4)
    for off, key, label, fc in ((-0.22, "precision", "Precision", "white"), (0, "recall", "Recall", WARM),
                                (0.22, "f1", "F1", INK)):
        ax.plot(x + off, DATA[key], "o", ms=4.6, mfc=fc, mec=INK if fc != WARM else WARM, mew=0.9,
                label=label, zorder=3)
    for i in range(4):
        vals = [DATA[k][i] for k in ("precision", "recall", "f1")]
        ax.plot([i - 0.22, i + 0.22], [min(vals), min(vals)], alpha=0)
    ax.set_xticks(x, short)
    ax.set_ylim(0.66, 0.94)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left", frameon=False, ncol=3, handletextpad=0.1, columnspacing=0.8,
              borderaxespad=0.1)
    tag(ax, "b", "Four-way precision and recall")

    ax = fig.add_subplot(gs[2])
    for i in range(4):
        ax.plot([i, i], [DATA["val_mA"][i], DATA["test_mA"][i]], color=GREY, lw=0.8, zorder=2)
    ax.plot(x, DATA["val_mA"], "o", ms=4.6, mfc="white", mec=INK, mew=0.9, label="Validation", zorder=3)
    ax.plot(x, DATA["test_mA"], "o", ms=4.6, mfc=WARM, mec=WARM, mew=0.9, label="Test", zorder=3)
    ax.set_xticks(x, short)
    ax.set_ylim(0.76, 0.94)
    ax.set_xlabel("Age group")
    ax.set_ylabel("mA")
    ax.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.1, columnspacing=0.8,
              borderaxespad=0.1)
    tag(ax, "c", "One-vs-rest mA")

    fig.savefig(OUT / "age_test.pdf")
    plt.close(fig)


def runtime():
    path = OUT / "runtime.json"
    if not path.exists():
        print("runtime.json missing: run scripts/benchmark_stages.py first, skipping the runtime figure")
        return
    r = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        ("Detect people\n(per frame)", r["detector"]["ms_per_frame"], INK),
        ("Age: histograms\n(per crop)", r["age"]["ms_features_only"], WARM),
        ("Age: four SVMs\n(per crop)", r["age"]["ms_per_crop"] - r["age"]["ms_features_only"], WARM),
        ("Gender: ViT-B/16\n(per crop)", r["gender"]["ms_per_crop_by_batch"]["16"], PICK),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    fig.subplots_adjust(left=0.3, right=0.86, top=0.97, bottom=0.23)
    y = range(len(rows))
    ax.barh(y, [v for _, v, _ in rows], color=[c for _, _, c in rows], height=0.56)
    for i, (_, v, _) in enumerate(rows):
        ax.text(v * 1.12, i, f"{v:.0f} ms", va="center", fontsize=7.5)
    ax.set_yticks(list(y), [n for n, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(5, 3000)
    ax.set_xlabel("Median time (ms, log scale)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    fig.savefig(OUT / "runtime.pdf")
    plt.close(fig)


if __name__ == "__main__":
    gender()
    age_sensitivity()
    schedule()
    age_data()
    age_test()
    runtime()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.pdf"))))
