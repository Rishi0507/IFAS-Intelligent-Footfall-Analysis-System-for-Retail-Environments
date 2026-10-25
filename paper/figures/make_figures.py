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
    experiment  paper/results/rq1_gap.json, rq2_adaptation.json, rq3_pooling.json and
              rq4_zones.json, written by scripts/research/ (docs/research-plan.md).
              Each of these figures is drawn only once its results file exists.
"""
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent / "results"

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


# ------------------------------------------------------- experiment figures
def _results(name, results):
    path = Path(results) / name
    if not path.exists():
        print(f"{name} missing: run the research scripts first, skipping its figure")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(ax, x, ci, colour, label=None, marker="o"):
    """A dot with its 95% interval as a vertical bar."""
    v, lo, hi = ci["value"], ci["lo"], ci["hi"]
    ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt=marker, ms=4.6, color=colour,
                mfc=colour, mec=colour, elinewidth=0.9, capsize=2, label=label, zorder=3)


def rq1(results=RESULTS, out=OUT):
    doc = _results("rq1_gap.json", results)
    if not doc:
        return
    rows = [r for r in doc["rows"] if "spectrum" not in r]
    spectra = [r for r in doc["rows"] if "spectrum" in r]
    fig = plt.figure(figsize=(7.16, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.9, 1.0], wspace=0.36,
                          left=0.07, right=0.99, top=0.86, bottom=0.27)

    # (a) PETA under each degradation, worst level per family plus clean.
    ax = fig.add_subplot(gs[0])
    deg = [r for r in rows if r["domain"] == "peta-degraded" and "accuracy" in r]
    names = ["clean"] + sorted({r["condition"] for r in deg if r["condition"] != "clean"})
    for task, colour, off in (("gender", PICK, -0.12), ("age", WARM, 0.12)):
        for i, cond in enumerate(names):
            hit = [r for r in deg if r["task"] == task and r["condition"] == cond]
            if hit:
                _ci(ax, i + off, hit[0]["balanced_accuracy"], colour, task if i == 0 else None)
    ax.set_xticks(range(len(names)), [n.replace("_", " ").replace("@", " ") for n in names],
                  rotation=90, fontsize=5.6)
    ax.set_ylabel("Balanced accuracy")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    tag(ax, "a", "PETA under controlled degradation")

    # (b) clean accuracy per domain.
    ax = fig.add_subplot(gs[1])
    domains = [d for d in ("peta", "mars", "retail") if any(r["domain"] == d for r in rows)]
    for task, colour, off in (("gender", PICK, -0.12), ("age", WARM, 0.12)):
        for i, d in enumerate(domains):
            hit = [r for r in rows if r["domain"] == d and r["task"] == task
                   and r["condition"] == "clean" and "accuracy" in r]
            if hit:
                _ci(ax, i + off, hit[0]["balanced_accuracy"], colour)
    ax.set_xticks(range(len(domains)), [d.upper() if d != "retail" else "Retail" for d in domains])
    ax.margins(x=0.25)
    ax.set_ylabel("Balanced accuracy")
    tag(ax, "b", "Domain gap")

    # (c) radial power spectrum of each domain's clean crops.
    ax = fig.add_subplot(gs[2])
    colours = {"peta": PICK, "mars": GREY, "retail": WARM}
    for r in spectra:
        if r["condition"] != "clean":
            continue
        s = r["spectrum"]
        d = r["domain"] if r["domain"] != "peta-degraded" else "peta"
        ax.plot(s["freqs"], s["log_power"], color=colours.get(d, INK), lw=1.2,
                label=d.upper() if d != "retail" else "Retail")
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("log$_{10}$ power")
    ax.legend(loc="upper right", frameon=False)
    tag(ax, "c", "Detail in the crops")
    fig.savefig(out / "rq1_gap.pdf")
    plt.close(fig)


def rq2(results=RESULTS, out=OUT):
    doc = _results("rq2_adaptation.json", results)
    if not doc:
        return
    rows = doc["rows"]
    metrics = [("peta_test_accuracy", "PETA test", PICK), ("retail_person_accuracy", "Retail, per person", WARM)]
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.9, bottom=0.3)
    x = np.arange(len(rows))
    for k, (m, label, colour) in enumerate(metrics):
        off = (k - 0.5) * 0.24
        for i, r in enumerate(rows):
            if m not in r:
                continue
            ax.errorbar([i + off], [r[m]["mean"]], yerr=[[r[m]["sd"] or 0]], fmt="o", ms=4.6,
                        color=colour, elinewidth=0.9, capsize=2, label=label if i == 0 else None)
            ax.plot([i + off] * len(r[m]["values"]), r[m]["values"], ".", ms=2.5, color=colour, alpha=0.5)
    ax.set_xticks(x, [r["condition"] for r in rows], rotation=20, fontsize=6.6)
    ax.set_ylabel("Gender accuracy")
    ax.legend(loc="lower right", frameon=False, fontsize=6.6)
    fig.savefig(out / "rq2_adaptation.pdf")
    plt.close(fig)


def rq3(results=RESULTS, out=OUT):
    doc = _results("rq3_pooling.json", results)
    if not doc:
        return
    rows = doc["rows"]
    fig = plt.figure(figsize=(7.16, 2.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.3, left=0.07, right=0.99, top=0.86, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    colours = {"gender": PICK, "age": WARM}
    names = list(dict.fromkeys(s["strategy"] for r in rows for s in r["strategies"]))
    for r in rows:
        off = -0.12 if r["task"] == "gender" else 0.12
        for s in r["strategies"]:
            i = names.index(s["strategy"])
            _ci(ax, i + off, s["accuracy"], colours[r["task"]], r["task"] if i == 0 else None)
    ax.set_xticks(range(len(names)), names, fontsize=6.6)
    ax.set_ylabel("Per-person accuracy")
    ax.legend(loc="lower right", frameon=False)
    tag(ax, "a", "Pooling strategy")

    ax = fig.add_subplot(gs[1])
    styles = {"majority": "-", "mean": "--", "confidence": ":", "quality": "-.",
              "attention": (0, (1, 1.2))}
    for r in rows:
        ks = [b["k"] for b in r["by_k"]]
        for s, ls in styles.items():
            ys = [b.get(s) for b in r["by_k"]]
            if any(y is not None for y in ys):
                ax.plot(ks, ys, ls, color=colours[r["task"]], lw=1.1, label=f"{r['task']}, {s}")
    ax.set_xlabel("Frames pooled per person (k)")
    ax.set_ylabel("Per-person accuracy")
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=6)
    tag(ax, "b", "Accuracy against frames pooled")
    fig.savefig(out / "rq3_pooling.pdf")
    plt.close(fig)


def rq3_numbers(results=RESULTS, out=OUT):
    r"""paper/figures/rq3_numbers.tex: every RQ3 number the text quotes, as
    \rqthree{task}{key}. Missing numbers print as ?? in the PDF, so the text can
    be written before the data exists and nothing in it is typed by hand."""
    doc = _results("rq3_pooling.json", results)
    lines = ["% Written by make_figures.py from results/rq3_pooling.json. Do not edit."]
    if doc:
        def put(task, key, val):
            lines.append(f"\\expandafter\\def\\csname rq3/{task}/{key}\\endcsname{{{val}}}")

        def ci(c):
            if c is None or c.get("value") is None or c["value"] != c["value"]:
                return None
            return f"{c['value']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"

        def pval(p):
            return "$<0.001$" if p < 0.001 else f"{p:.3f}"

        for r in doc["rows"]:
            t = r["task"]
            put(t, "n", r["n_people"])
            put(t, "crops", r["n_crops"])
            for st in r["strategies"]:
                name = st["strategy"]
                if ci(st["accuracy"]):
                    put(t, name, ci(st["accuracy"]))
                    put(t, f"{name}-value", f"{st['accuracy']['value']:.2f}")
                mc = st.get("mcnemar_vs_majority")
                if mc:
                    put(t, f"p-{name}", pval(mc["p"]))
                    put(t, f"b-{name}", mc["b"])
                    put(t, f"c-{name}", mc["c"])
            for sub, vals in r.get("subsets", {}).items():
                put(t, f"{sub}-n", vals["n_people"])
                for name in ("majority", "quality", "attention"):
                    if vals.get(name) is not None:
                        put(t, f"{sub}-{name}", f"{vals[name]:.2f}")
            for name, coef in (r.get("quality_model_coef") or {}).items():
                put(t, f"coef-{name}", f"{round(coef, 2) + 0.0:+.2f}")     # + 0.0: no "-0.00"
            att = r.get("attention", {})
            rho = (att.get("vs_quality") or {}).get("mean_spearman")
            if rho is not None:
                put(t, "rho", f"{rho:.2f}")
            if att.get("fitted_on"):
                put(t, "attention-fitted", "cross-validation over people"
                    if "cross-validation" in att["fitted_on"] else "the MARS training tracklets")
            put(t, "fitted", "cross-validation over people" if "cross-validation"
                in r["quality_model_fitted_on"] else "MARS")
        lines.append(f"\\def\\rqthreeCommit{{{doc['meta'].get('commit', '')}}}")
    (out / "rq3_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rq4(results=RESULTS, out=OUT):
    doc = _results("rq4_zones.json", results)
    if not doc:
        return
    rows = doc["rows"]
    base = doc["meta"]["base_params"]
    fig = plt.figure(figsize=(7.16, 2.15))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.0, 0.95, 1.05], wspace=0.5,
                          left=0.06, right=0.995, top=0.85, bottom=0.25)

    # (a) flicker against the gap bridged, with and without dropping short stays.
    ax = fig.add_subplot(gs[0])
    sm = [r for r in rows if r["study"] == "smoothing"]
    offs = sorted({r["off_sec"] for r in sm})
    raw = sm[0]["raw_flips_per_min"]
    ax.axhline(raw, color=GREY, lw=0.8, ls="--", zorder=1)
    ax.annotate("raw", (offs[-1], raw), xytext=(0, 2), textcoords="offset points",
                ha="right", va="bottom", color=GREY, fontsize=6.6)
    for on, colour in ((0.0, WARM), (base["on_sec"], PICK)):
        ys = [next(r["smoothed_flips_per_min"] for r in sm if r["on_sec"] == on and r["off_sec"] == o)
              for o in offs]
        ax.plot(offs, ys, "-o", color=colour, lw=1.1, ms=3.2, mfc="white", mew=0.9,
                label=f"drop stays < {on:g} s" if on else "keep every stay")
        if on == base["on_sec"]:
            v = ys[offs.index(base["off_sec"])]
            ax.plot(base["off_sec"], v, "o", ms=4.2, color=PICK, zorder=4)
            ax.annotate(f"{v:.2f}", (base["off_sec"], v), xytext=(4, 4), textcoords="offset points",
                        color=PICK, fontweight="bold", fontsize=7)
    ax.set_yscale("log")
    ticks = [0.3, 1, 3, 10]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Gaps bridged up to (s)")
    ax.set_ylabel("State changes per zone-minute")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=False, fontsize=6.2,
              handlelength=1.4)
    tag(ax, "a", "Smoothing")

    # (b) agreement with the full sampling rate.
    ax = fig.add_subplot(gs[1])
    sa = sorted((r for r in rows if r["study"] == "sampling"), key=lambda r: r["sample_sec"])
    xs = [r["sample_sec"] for r in sa]
    ys = [100 * r["state_agreement"] for r in sa]
    ax.plot(xs, ys, "-o", color=PICK, lw=1.1, ms=3.2, mfc="white", mew=0.9)
    for x, y in zip(xs[1:], ys[1:]):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=6.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [f"{x:g}" for x in xs])
    ax.minorticks_off()
    ax.set_ylim(min(ys) - 3, 101.5)
    ax.set_xlabel("Seconds between samples")
    ax.set_ylabel("Same state as full rate (%)")
    tag(ax, "b", "Sampling")

    # (c) where a person "is": sightings that fall in no zone.
    ax = fig.add_subplot(gs[2])
    an = [r for r in rows if r["study"] == "anchor"]
    names = [r["anchor"] for r in an]
    vals = [100 * r["unassigned_share"] for r in an]
    for i, (n, v) in enumerate(zip(names, vals)):
        chosen = n == base["anchor"]
        ax.barh(i, v, height=0.56, color=PICK if chosen else GREY, zorder=2)
        ax.annotate(f"{v:.1f}%", (v, i), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=6.6, color=INK, fontweight="bold" if chosen else None)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Sightings in no zone (%)")
    tag(ax, "c", "Anchor")

    # (d) identity: tracks against people seated.
    ax = fig.add_subplot(gs[3])
    idr = next(r for r in rows if r["study"] == "identity")
    bars = [("Tracks", idr["tracks"], WARM), ("In view\n(peak)", idr["peak_people_in_frame"], PICK),
            ("Seats\n(peak)", idr["peak_occupied_zones"], PICK),
            ("Seats\n(end)", idr["occupied_at_end"], PICK)]
    for i, (_n, v, c) in enumerate(bars):
        ax.bar(i, v, width=0.62, color=c, zorder=2)
        ax.annotate(f"{v}", (i, v), xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=6.8)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=6.2)
    ax.set_ylim(0, idr["tracks"] * 1.15)
    ax.set_ylabel("Count")
    tag(ax, "d", "Identity")
    fig.savefig(out / "rq4_zones.pdf")
    plt.close(fig)


def rq4_numbers(results=RESULTS, out=OUT):
    r"""paper/figures/rq4_numbers.tex: every RQ4 number the text quotes, as
    \rqfour{key}. Missing numbers print as ?? in the PDF."""
    doc = _results("rq4_zones.json", results)
    lines = ["% Written by make_figures.py from results/rq4_zones.json. Do not edit."]
    if doc:
        def put(key, val):
            lines.append(f"\\expandafter\\def\\csname rq4/{key}\\endcsname{{{val}}}")

        m, rows = doc["meta"], doc["rows"]
        bp = m["base_params"]
        put("duration-min", f"{m['duration_sec'] / 60:.1f}")
        put("zones", m["zones"])
        put("zone-source", "proposed automatically" if m.get("zone_source") == "auto" else "drawn by hand")
        put("sample-sec", f"{m['sample_sec']:.1f}")
        for k in ("on_sec", "off_sec", "stick"):
            put(k.replace("_", "-"), f"{bp[k]:g}")
        base = next(r for r in rows if r["study"] == "smoothing"
                    and r["on_sec"] == bp["on_sec"] and r["off_sec"] == bp["off_sec"])
        put("raw-flicker", f"{base['raw_flips_per_min']:.2f}")
        put("flicker", f"{base['smoothed_flips_per_min']:.2f}")
        put("flicker-ratio", f"{base['raw_flips_per_min'] / base['smoothed_flips_per_min']:.0f}")
        put("utilisation", f"{100 * base['mean_utilization']:.0f}")
        put("stays", base["sessions"])
        put("median-stay", f"{base['median_session_sec']:.0f}")
        put("unassigned", f"{100 * base['unassigned_share']:.1f}")
        put("occupied", f"{base['mean_occupied']:.1f}")
        loose = next((r for r in rows if r["study"] == "stick" and r["stick"] == 0), None)
        if loose:
            put("raw-flicker-nostick", f"{loose['raw_flips_per_min']:.2f}")
            put("stick-cut", f"{100 * (1 - base['raw_flips_per_min'] / loose['raw_flips_per_min']):.0f}")
            put("occupied-nostick", f"{loose['mean_occupied']:.1f}")
        for r in rows:
            if r["study"] == "anchor":
                put(f"unassigned-{r['anchor']}", f"{100 * r['unassigned_share']:.1f}")
            if r["study"] == "sampling" and r["factor"] > 1:
                put(f"agree-{r['sample_sec']:g}", f"{100 * r['state_agreement']:.1f}")
                put(f"utilerr-{r['sample_sec']:g}", f"{100 * r['utilization_abs_err']:.1f}")
            if r["study"] == "proposal":
                put(f"proposal-{r['min_evidence_sec']:g}", r["zones"])
            if r["study"] == "identity":
                for k in ("tracks", "peak_people_in_frame", "peak_occupied_zones", "occupied_at_end",
                          "tracks_per_person", "tracks_rejoined"):
                    put(k.replace("_", "-"), r[k])
            if r["study"] == "labels" and r.get("base"):
                b = r["base"]
                put("lab-annotators", ", ".join(r["annotators"]))
                put("lab-items", b["items"])
                put("lab-unsure", b["unsure"])
                put("lab-moments", b["moments"])
                put("lab-base-rate", f"{100 * b['base_rate_taken']:.0f}")
                put("lab-kappa", f"{b['kappa']:.2f}")
                for k in ("accuracy", "precision", "recall", "f1"):
                    put(f"lab-{k}", f"{b[k]['value']:.2f} [{b[k]['lo']:.2f}, {b[k]['hi']:.2f}]")
            if r["study"] == "smoothing" and r.get("labels") and r["on_sec"] == bp["on_sec"]:
                put(f"lab-recall-off{r['off_sec']:g}", f"{r['labels']['recall']['value']:.2f}")
                put(f"lab-precision-off{r['off_sec']:g}", f"{r['labels']['precision']['value']:.2f}")
            if r["study"] == "next_free" and r.get("moments"):
                put("nf-moments", r["moments"])
                for k in ("longest_seated", "km_residual", "random"):
                    put(f"nf-{k.replace('_', '-')}", f"{r[k]:.3f}")
        lines.append(f"\\def\\rqfourCommit{{{m.get('commit', '')}}}")
    det = _results("rq4_detection.json", results)
    if det:
        lines.append(f"\\expandafter\\def\\csname rq4/det-frames\\endcsname{{{len(det['meta']['frames'])}}}")
        for r in det["rows"]:
            key = f"det/{r['model']}/{r['tiles']}"
            lines.append(f"\\expandafter\\def\\csname rq4/{key}/{r['conf']:g}\\endcsname{{{r['mean']:.1f}}}")
            lines.append(f"\\expandafter\\def\\csname rq4/{key}/sec\\endcsname{{{r['sec_per_frame']:.2f}}}")
    (out / "rq4_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def attr_numbers(results=RESULTS, out=OUT):
    r"""paper/figures/attr_numbers.tex: every number the held-out and CCTV-label
    section quotes, as \attr{key}, from attributes.json, label_check_before.json,
    identity_gap.json and rq1_gap.json. Missing numbers print as ??."""
    lines = ["% Written by make_figures.py from results/attributes.json, label_check_before.json,",
             "% identity_gap.json and rq1_gap.json. Do not edit."]

    def put(key, val):
        lines.append(f"\\expandafter\\def\\csname attr/{key}\\endcsname{{{val}}}")

    def f3(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "??"

    att = _results("attributes.json", results)
    if att:
        for r in att["rows"]:
            data = {"PETA test": "peta", "PETA CUHK test": "cuhk"}.get(r["data"], "room")
            key = f"{r['task']}/{data}/{r['model']}"
            put(f"{key}/acc", f3(r["accuracy"]))
            put(f"{key}/bal", f3(r["balanced"]))
            put(f"{key}/n", r["n"])
            for c in r.get("coverage") or []:
                th = f"{c['min_conf']:g}"
                put(f"{key}/kept{th}", f"{100 * c['kept']:.0f}" if c["kept"] is not None else "??")
                put(f"{key}/acc{th}", f"{100 * c['accuracy']:.1f}" if c["accuracy"] is not None else "??")
    before = _results("label_check_before.json", results)
    if before:
        for r in before["rows"]:
            a = r["accuracy"]
            put(f"before/{r['task']}", f"{a['value']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]")
            put(f"before/{r['task']}/n", r["people_labelled"])
    gap = _results("identity_gap.json", results)
    if gap:
        row = next(x for x in gap["rows"] if x["iou"] == 0.5)
        put("gap/same05", f"{100 * row['same_person_kept']:.1f}")
        put("gap/neighbour05", f"{100 * row['neighbour_pairs_at_or_above']:.1f}")
        put("gap/neighbour99", f"{gap['neighbours']['percentiles']['99']:.2f}")
        put("gap/neighbour03", f"{100 * next(x for x in gap['rows'] if x['iou'] == 0.3)['neighbour_pairs_at_or_above']:.0f}")
    rq1 = _results("rq1_gap.json", results)
    if rq1:
        for r in rq1["rows"]:
            if r.get("task") == "age" and r.get("balanced_accuracy"):
                put("age/cuhk/svm/bal", f3(r["balanced_accuracy"]["value"]))
    (out / "attr_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    gender()
    age_sensitivity()
    schedule()
    age_data()
    age_test()
    runtime()
    rq1()
    rq2()
    rq3()
    rq3_numbers()
    rq4()
    rq4_numbers()
    attr_numbers()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.pdf"))))
