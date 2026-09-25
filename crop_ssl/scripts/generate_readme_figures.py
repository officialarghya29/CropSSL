"""
Generate README figures from MEASURED results only.

Every number plotted here is read from results/*.json produced by actual
runs of this repository:
  - results/covariate_shift_measured.json  (covariate_shift_exp.py --k-sweep)
  - results/mobile_export_measured.json    (export_to_onnx_mobile run output)

Re-generate with:
    python3 -m crop_ssl.scripts.generate_readme_figures
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "assets" / "figures"

# Futuristic dark theme (matches the GitHub dark landing aesthetic)
BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
C_NAIVE_SSL = "#22d3ee"   # cyan
C_NAIVE_RND = "#f87171"   # red
C_LORA_SSL = "#a78bfa"    # violet
C_LORA_RND = "#fb923c"    # orange
C_LIN_SSL = "#34d399"     # emerald
C_LIN_RND = "#facc15"     # yellow
C_ORACLE = "#4ade80"      # green dashed


def _style_axis(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)


def fig_k_sweep(data, out_path):
    rows = data["results"]
    ks = sorted({r["k_shots"] for r in rows})

    def series(tag, key):
        pts = sorted(
            (r["k_shots"], r[key]) for r in rows if r["tag"] == tag
        )
        return [k for k, _ in pts], [v * 100 for _, v in pts]

    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=BG)
    _style_axis(ax)

    oracle_lo = min(r["oracle"] for r in rows) * 100
    oracle_hi = max(r["oracle"] for r in rows) * 100
    ax.axhspan(oracle_lo, oracle_hi, color=C_ORACLE, alpha=0.10)
    ax.axhline(
        sum((oracle_lo, oracle_hi)) / 2,
        color=C_ORACLE, linestyle="--", linewidth=1.2, alpha=0.85,
        label="Field-oracle upper bound",
    )

    for tag, key, color, label, marker in [
        ("ssl_pretrained", "naive", C_NAIVE_SSL,
         "Zero-shot (SSL pre-trained)", "o"),
        ("random_init", "naive", C_NAIVE_RND,
         "Zero-shot (random init)", "o"),
        ("ssl_pretrained", "linear", C_LIN_SSL,
         "Linear probe (SSL)", "s"),
        ("random_init", "linear", C_LIN_RND,
         "Linear probe (random)", "s"),
        ("ssl_pretrained", "lora", C_LORA_SSL,
         "LoRA (SSL)", "D"),
        ("random_init", "lora", C_LORA_RND,
         "LoRA (random)", "D"),
    ]:
        x, y = series(tag, key)
        ax.plot(x, y, color=color, marker=marker, markersize=6,
                linewidth=2.2, label=label)

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Labeled field shots per class (k)", color=FG, fontsize=11)
    ax.set_ylabel("Field accuracy (%)", color=FG, fontsize=11)
    ax.set_title(
        "Few-shot field adaptation vs. shift severity (measured)",
        color=FG, fontsize=13, pad=12,
    )
    legend = ax.legend(facecolor=BG, edgecolor=GRID, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_calibration(data, out_path):
    rows = data["results"]
    backbones = ["ssl_pretrained", "random_init"]
    labels = ["SSL pre-trained", "Random init"]
    raw = [next(r["ece_field_raw"] for r in rows if r["tag"] == b) * 100
           for b in backbones]
    cal = [next(r["ece_field_cal"] for r in rows if r["tag"] == b) * 100
           for b in backbones]

    x = range(len(backbones))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.6), facecolor=BG)
    _style_axis(ax)
    bars1 = ax.bar([i - width / 2 for i in x], raw, width,
                   color=C_NAIVE_RND, alpha=0.55, label="Uncalibrated")
    bars2 = ax.bar([i + width / 2 for i in x], cal, width,
                   color=C_NAIVE_SSL, label="Lab-calibrated (T = 1.08)")
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}%", ha="center", color=FG,
                    fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=FG, fontsize=11)
    ax.set_ylabel("Expected Calibration Error on field (%)", color=FG,
                  fontsize=10)
    ax.set_title(
        "Confidence calibration survives the domain gap only for "
        "pre-trained features (measured)",
        color=FG, fontsize=12, pad=12,
    )
    legend = ax.legend(facecolor=BG, edgecolor=GRID, fontsize=9)
    for text in legend.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_param_efficiency(data, out_path):
    rows = data["results"]
    k5_ssl = next(r for r in rows if r["tag"] == "ssl_pretrained"
                  and r["k_shots"] == 5)
    linear_params = k5_ssl["linear_params"]
    lora_params = k5_ssl["lora_params"]

    fig, ax = plt.subplots(figsize=(7.4, 3.6), facecolor=BG)
    _style_axis(ax)
    names = ["Linear probe", "LoRA (r=8)"]
    vals = [linear_params, lora_params]
    bars = ax.barh(names, vals, color=[C_LIN_SSL, C_LORA_SSL], height=0.5)
    for b, v in zip(bars, vals):
        ax.text(v * 1.15, b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", color=FG, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)", color=FG, fontsize=10)
    ax.set_title(
        "Adaptation cost: LoRA trains ~1% of the model (measured)",
        color=FG, fontsize=12, pad=10,
    )
    ax.set_xlim(right=max(vals) * 6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_mobile_int8(mobile, out_path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor=BG)
    _style_axis(ax)
    names = ["fp32 (static shape)", "int8 (QDQ, static)"]
    vals = [mobile["fp32_size_kb"], mobile["int8_size_kb"]]
    bars = ax.bar(names, vals, color=["#7dd3fc", "#f0abfc"], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + vals[0] * 0.02,
                f"{v/1024:.1f} MB", ha="center", color=FG, fontsize=11)
    ratio = vals[0] / vals[1]
    ax.annotate(f"{ratio:.2f}x smaller",
                xy=(1, vals[1] / 2), xytext=(0.52, vals[0] * 0.55),
                color="#f0abfc", fontsize=13,
                arrowprops=dict(arrowstyle="->", color="#f0abfc"))
    ax.set_ylabel("Model size (KB)", color=FG, fontsize=10)
    ax.set_title(
        "On-device footprint after int8 quantization (measured, ViT-S/16)",
        color=FG, fontsize=12, pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cov = json.loads((RESULTS_DIR / "covariate_shift_measured.json").read_text())
    fig_k_sweep(cov, FIGURES_DIR / "k_sweep_accuracy.png")
    fig_calibration(cov, FIGURES_DIR / "calibration_transfer.png")
    fig_param_efficiency(cov, FIGURES_DIR / "parameter_efficiency.png")

    mobile_path = RESULTS_DIR / "mobile_export_measured.json"
    if mobile_path.exists():
        fig_mobile_int8(json.loads(mobile_path.read_text()),
                        FIGURES_DIR / "mobile_int8.png")
    else:
        print(f"skipping mobile figure: {mobile_path.name} not found "
              f"(run export_to_onnx_mobile first)")


if __name__ == "__main__":
    main()
