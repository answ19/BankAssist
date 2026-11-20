"""Confidence, coverage and FAQ-gated eligibility from saved prediction files.

Replays the routing rule (confidence >= threshold AND an approved FAQ for the
predicted intent) on saved predictions; nothing is retrained or re-predicted.
Ollama wording is not run, so this measures reply eligibility, not answer quality.

Writes, for the chosen split name:
    figures/coverage_<split>.pdf/.png      coverage and accuracy against the threshold
    figures/reliability_<split>.pdf/.png   reliability diagram (10 equal-width bins)
    docs/coverage_<split>.md               tables behind the figures, with their labels

    python coverage_analysis.py
    python coverage_analysis.py --faq data/faq/faq_drafts.csv --threshold 0.90
    python coverage_analysis.py --split test \
        --predictions "Selected TF-IDF + LR=results/baseline_test_predictions.csv" \
        --predictions "DistilBERT=results/distilbert/test_predictions.csv"
"""

import argparse
import csv
import hashlib
import sys
from datetime import date
from pathlib import Path

import evaluation
import routing


ROOT = Path(__file__).resolve().parent
FAQ_PATH = ROOT / "data" / "faq" / "faq_drafts.csv"
FIGURE_DIR = ROOT / "figures"
DOCS_DIR = ROOT / "docs"
DEFAULT_PREDICTIONS = [
    ("Selected TF-IDF + LR", ROOT / "results" / "baseline_validation_predictions.csv"),
    ("DistilBERT", ROOT / "results" / "distilbert" / "validation_predictions.csv"),
]
TABLE_THRESHOLDS = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]
CURVE_THRESHOLDS = [round(0.30 + 0.01 * step, 2) for step in range(70)]  # 0.30 to 0.99

# Categorical slots 1-2 (blue, orange) and text inks; see the dataviz palette.
COLORS = ["#2a78d6", "#eb6834"]
MARKERS = ["o", "s"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DASHED = (0, (4, 2))


def read_predictions(path):
    """Return (true intents, predicted intents, confidence scores, queries) from a saved CSV."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = {"true_intent", "predicted_intent", "confidence_score"} - set(rows[0] if rows else {})
    if not rows or missing:
        raise ValueError("{} has no rows or lacks columns: {}".format(
            path, ", ".join(sorted(missing)) or "-"))
    return ([r["true_intent"] for r in rows], [r["predicted_intent"] for r in rows],
            [float(r["confidence_score"]) for r in rows], [r.get("query", "") for r in rows])


def read_faq_snapshot(path):
    """Approved intents plus a short content hash that identifies this FAQ version."""
    data = Path(path).read_bytes()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    approved = sorted(r["intent"] for r in rows if r["approved"].strip() == "1")
    return {"approved": approved, "total": len(rows),
            "sha256": hashlib.sha256(data).hexdigest()[:12]}


def eligibility(true, predicted, confidences, approved, threshold):
    """Confidence-only and FAQ-gated eligibility at one threshold."""
    approved = set(approved)
    confident = [i for i, score in enumerate(confidences) if score >= threshold]
    gated = [i for i in confident if predicted[i] in approved]

    def summary(indices):
        correct = sum(true[i] == predicted[i] for i in indices)
        return {"eligible": len(indices), "coverage": len(indices) / len(true),
                "correct": correct, "errors": len(indices) - correct,
                "accuracy": correct / len(indices) if indices else None}

    return {"threshold": threshold, "confidence_only": summary(confident),
            "faq_gated": summary(gated), "faq_gated_errors": [i for i in gated
                                                              if true[i] != predicted[i]]}


def analyse(models, approved, threshold):
    results = []
    for name, path in models:
        true, predicted, confidences, queries = read_predictions(path)
        correct = [t == p for t, p in zip(true, predicted)]
        results.append({
            "name": name, "path": path, "count": len(true),
            "accuracy": sum(correct) / len(correct),
            "calibration": evaluation.calibration_bins(confidences, correct),
            "table": [eligibility(true, predicted, confidences, approved, t)
                      for t in TABLE_THRESHOLDS],
            "curve": [eligibility(true, predicted, confidences, approved, t)
                      for t in CURVE_THRESHOLDS],
            "at_threshold": eligibility(true, predicted, confidences, approved, threshold),
            "rows": (true, predicted, confidences, queries),
        })
    return results


def pct(value):
    return "n/a" if value is None else "{:.2f}%".format(100 * value)


def caption(split, results, faq, threshold):
    sizes = ", ".join("{} n={:,}".format(r["name"], r["count"]) for r in results)
    return ("Split: {} ({}). Threshold {:.2f}. FAQ snapshot: {} of {} intents approved "
            "(faq_drafts.csv sha256 {}). Ollama wording checks: not run (eligibility "
            "replay only).".format(split, sizes, threshold, len(faq["approved"]),
                                   faq["total"], faq["sha256"]))


def build_report(split, results, faq, threshold, figures):
    lines = [
        "# Confidence and coverage: {} split".format(split), "",
        "Generated {} by `python coverage_analysis.py`. {}".format(
            date.today(), caption(split, results, faq, threshold)), "",
        "Approved intents: {}.".format(", ".join(faq["approved"]) or "none"), "",
        "Confidence-only = confidence at or above the threshold. FAQ-gated = also has an "
        "approved FAQ for the predicted intent, which is the application's rule for an "
        "automatic reply. Accuracy is intent accuracy within the eligible subset.", "",
        "## At the configured threshold ({:.2f})".format(threshold), "",
        "| Model | Accuracy (all) | ECE | Confidence-only eligible | Coverage | Accuracy | "
        "Errors | FAQ-gated eligible | Coverage | Correct | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        c, g = r["at_threshold"]["confidence_only"], r["at_threshold"]["faq_gated"]
        lines.append("| {} | {} | {:.4f} | {:,} | {} | {} | {} | {:,} | {} | {} | {} |".format(
            r["name"], pct(r["accuracy"]), r["calibration"]["ece"], c["eligible"],
            pct(c["coverage"]), pct(c["accuracy"]), c["errors"], g["eligible"],
            pct(g["coverage"]), g["correct"], g["errors"]))
    for r in results:
        lines += ["", "## {}: threshold sweep".format(r["name"]), "",
                  "| Threshold | Confidence-only eligible | Coverage | Accuracy | Errors | "
                  "FAQ-gated eligible | Coverage | Correct | Errors |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in r["table"]:
            c, g = row["confidence_only"], row["faq_gated"]
            lines.append("| {:.2f} | {:,} | {} | {} | {} | {:,} | {} | {} | {} |".format(
                row["threshold"], c["eligible"], pct(c["coverage"]), pct(c["accuracy"]),
                c["errors"], g["eligible"], pct(g["coverage"]), g["correct"], g["errors"]))
        true, predicted, _, queries = r["rows"]
        errors = r["at_threshold"]["faq_gated_errors"]
        lines += ["", "Incorrect FAQ-gated predictions at {:.2f} ({}):".format(
            threshold, len(errors)), ""]
        if errors:
            lines += ["| Query | True intent | Predicted intent |", "|---|---|---|"]
            lines += ["| {} | {} | {} |".format(queries[i].replace("|", "/"), true[i],
                                                 predicted[i]) for i in errors]
        else:
            lines.append("None.")
        lines += ["", "Reliability bins (10 equal-width bins):", "",
                  "| Bin | Queries | Mean confidence | Accuracy |", "|---|---:|---:|---:|"]
        lines += ["| {:.1f}-{:.1f} | {} | {} | {} |".format(
            b["low"], b["high"], b["count"], pct(b["mean_confidence"]), pct(b["accuracy"]))
            for b in r["calibration"]["bins"]]
    lines += ["", "## Figures", ""]
    lines += ["- `{}`".format(path.relative_to(ROOT).as_posix()) for path in figures]
    lines += ["", "Prediction files:", ""]
    lines += ["- {}: `{}`".format(r["name"], Path(r["path"]).resolve().relative_to(ROOT)
                                  .as_posix()) for r in results]
    return "\n".join(lines) + "\n"


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, width=0.6, length=3)


def plot_figures(split, results, faq, threshold, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelcolor": INK, "text.color": INK,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Figure 1: coverage (top) and accuracy within the eligible subset (bottom).
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.5, 4.1), sharex=True)
    xs = CURVE_THRESHOLDS
    for color, marker, r in zip(COLORS, MARKERS, results):
        conf = [100 * row["confidence_only"]["coverage"] for row in r["curve"]]
        gated = [100 * row["faq_gated"]["coverage"] for row in r["curve"]]
        acc = [100 * row["confidence_only"]["accuracy"] if row["confidence_only"]["accuracy"]
               is not None else float("nan") for row in r["curve"]]
        mark = {"marker": marker, "markevery": [xs.index(threshold)] if threshold in xs else [],
                "markersize": 4.5, "markeredgecolor": "white", "markeredgewidth": 0.8}
        top.plot(xs, conf, color=color, linewidth=1.4, **mark)
        top.plot(xs, gated, color=color, linewidth=1.4, linestyle=DASHED, **mark)
        bottom.plot(xs, acc, color=color, linewidth=1.4, label=r["name"], **mark)
    for ax in (top, bottom):
        style_axes(ax)
        ax.axvline(threshold, color=INK_MUTED, linewidth=0.8, linestyle=":")
    top.set_ylabel("Eligible queries (%)")
    top.set_ylim(0, 100)
    top.text(threshold, 98, " τ = {:.2f}".format(threshold), color=INK_MUTED,
             fontsize=7, va="top", ha="left")
    # Colour identifies the model, line style the rule, so the key stays short.
    from matplotlib.lines import Line2D
    key = [Line2D([], [], color=color, linewidth=1.4, marker=marker, markersize=4.5,
                  markeredgecolor="white", label=r["name"])
           for color, marker, r in zip(COLORS, MARKERS, results)]
    key += [Line2D([], [], color=INK_MUTED, linewidth=1.4, label="Confidence only"),
            Line2D([], [], color=INK_MUTED, linewidth=1.4, linestyle=DASHED,
                   label="With approved FAQ")]
    top.legend(handles=key, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.12),
               handlelength=2.6)
    bottom.set_ylabel("Intent accuracy when eligible (%)")
    bottom.set_xlabel("Confidence threshold τ")
    bottom.set_xlim(xs[0], xs[-1])
    bottom.legend(frameon=False, loc="lower right")
    fig.align_ylabels((top, bottom))
    fig.tight_layout(h_pad=0.8)
    saved += save(fig, out_dir / "coverage_{}".format(split))
    plt.close(fig)

    # Figure 2: reliability diagram.
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.8, linestyle=":",
            label="Perfect calibration")
    for color, marker, r in zip(COLORS, MARKERS, results):
        bins = r["calibration"]["bins"]
        ax.plot([b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins],
                color=color, linewidth=1.4, marker=marker, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.8,
                label="{} (ECE {:.4f})".format(r["name"], r["calibration"]["ece"]))
    style_axes(ax)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    saved += save(fig, out_dir / "reliability_{}".format(split))
    plt.close(fig)
    return saved


def save(fig, stem):
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=300, bbox_inches="tight", facecolor="white")
    return paths


def parse_prediction(value):
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("use NAME=PATH, got {!r}".format(value))
    return name.strip(), Path(path.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="validation", help="split name used in labels "
                        "and output file names (default: validation)")
    parser.add_argument("--predictions", action="append", type=parse_prediction,
                        help="NAME=PATH of a saved predictions CSV; repeat for each model "
                        "(default: the baseline and DistilBERT validation files)")
    parser.add_argument("--faq", type=Path, default=FAQ_PATH, help="FAQ CSV with approvals")
    parser.add_argument("--threshold", type=float, default=routing.CONFIDENCE_THRESHOLD)
    parser.add_argument("--no-figures", action="store_true", help="write the tables only")
    args = parser.parse_args(argv)
    models = args.predictions or DEFAULT_PREDICTIONS
    try:
        threshold = routing.check_probability("threshold", args.threshold)
        faq = read_faq_snapshot(args.faq)
        results = analyse(models, faq["approved"], threshold)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print("Coverage analysis failed: {}".format(error), file=sys.stderr)
        return 1
    figures = [] if args.no_figures else plot_figures(args.split, results, faq, threshold,
                                                      FIGURE_DIR)
    report_path = DOCS_DIR / "coverage_{}.md".format(args.split)
    report_path.write_text(build_report(args.split, results, faq, threshold, figures),
                           encoding="utf-8")
    print(caption(args.split, results, faq, threshold))
    for r in results:
        c, g = r["at_threshold"]["confidence_only"], r["at_threshold"]["faq_gated"]
        print("{}: confidence-only {:,} ({}, {} errors); with approved FAQ {:,} ({}, "
              "{} correct)".format(r["name"], c["eligible"], pct(c["coverage"]), c["errors"],
                                   g["eligible"], pct(g["coverage"]), g["correct"]))
    for path in figures + [report_path]:
        print("Saved: {}".format(path.relative_to(ROOT).as_posix()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
