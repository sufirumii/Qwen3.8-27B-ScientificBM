"""Builds the README graphs from benchmark/results/model_comparison.csv.

Numbers for the other 12 models are copied from the BioTool paper (Tables 1 and 2).
Numbers for ScientificBM come from our own run (see benchmark/run_benchmark.py).

Run:  python benchmark/make_charts.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "assets"
OUT.mkdir(exist_ok=True)

# Score if all 15 slow sequence-search samples that were not scored live had been perfect (see README).
UPPER_BOUND_OVERALL = 92.7

# ---- look & feel -----------------------------------------------------------
for f in font_manager.findSystemFonts():
    if "Inter-" in f or "Inter_" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B8BCC6", "axes.labelcolor": "#2B2F3A",
    "xtick.color": "#4A4F5C", "ytick.color": "#2B2F3A",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
INK, MUTED = "#1F2430", "#6B7280"
COL = {"Ours": "#F26419", "BioTool-FT": "#33658A", "Proprietary": "#6D7587", "Base open-source": "#B9C0CE"}
LEGEND = [Patch(color=COL["Ours"], label="Qwen3.8-27B-ScientificBM (ours)"),
          Patch(color=COL["BioTool-FT"], label="Small models fine-tuned on BioTool (paper)"),
          Patch(color=COL["Proprietary"], label="Frontier models, no fine-tuning (paper)"),
          Patch(color=COL["Base open-source"], label="Small open models, no fine-tuning (paper)")]
SOURCE = "Other models: BioTool paper (Gao et al., 2026), Tables 1-2. ScientificBM: single run, 3-bit GGUF, 1,408 test questions."


def load():
    rows = []
    for r in csv.DictReader(open(HERE / "results" / "model_comparison.csv")):
        for k in r:
            if k not in ("model", "group"):
                r[k] = float(r[k])
        rows.append(r)
    return rows


ROWS = load()
OURS = next(r for r in ROWS if r["group"] == "Ours")
SHORT = lambda n: n.replace("Qwen3.8-27B-ScientificBM (ours)", "ScientificBM (ours)").replace(" (BioTool-FT)", " + BioTool FT").replace(" (base)", " (base)")


def footer(fig, y=0.015):
    fig.text(0.01, y, SOURCE, fontsize=7.5, color=MUTED, ha="left")


def titles(fig, title, sub):
    fig.text(0.01, 0.965, title, fontsize=17, weight="bold", color=INK, va="top")
    fig.text(0.01, 0.915, sub, fontsize=10.5, color=MUTED, va="top")


# ---- 1. overall ranking -----------------------------------------------------
def chart_ranking():
    data = sorted(ROWS, key=lambda r: r["score_overall"])
    fig, ax = plt.subplots(figsize=(11, 7.4))
    fig.subplots_adjust(left=0.28, right=0.96, top=0.84, bottom=0.12)
    for i, r in enumerate(data):
        ours = r["group"] == "Ours"
        ax.barh(i, r["score_overall"], color=COL[r["group"]], height=0.68, zorder=3)
        if ours:
            ax.barh(i, UPPER_BOUND_OVERALL - r["score_overall"], left=r["score_overall"], height=0.68,
                    facecolor="#FBD3BD", edgecolor=COL["Ours"], hatch="////", linewidth=0.8, zorder=3)
            ax.text(UPPER_BOUND_OVERALL + 0.8, i, f"{r['score_overall']:.1f}  (up to {UPPER_BOUND_OVERALL})",
                    va="center", fontsize=10.5, weight="bold", color=COL["Ours"])
        else:
            ax.text(r["score_overall"] + 0.8, i, f"{r['score_overall']:.1f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(data)))
    labels = ax.set_yticklabels([SHORT(r["model"]) for r in data], fontsize=10.5)
    for lab, r in zip(labels, data):
        if r["group"] == "Ours":
            lab.set_weight("bold"); lab.set_color(COL["Ours"])
    ax.set_xlim(0, 108); ax.set_xlabel("BioTool Score (higher is better)", fontsize=10)
    ax.xaxis.grid(True, color="#ECEEF2", zorder=0); ax.set_axisbelow(True)
    ax.legend(handles=LEGEND, loc="lower right", frameon=False, fontsize=9)
    titles(fig, "ScientificBM beats every frontier model on biomedical tool calling",
           "Overall BioTool Score on the 1,408-question test set. Hatched part = score if the 15 unscored samples were all perfect.")
    footer(fig)
    fig.savefig(OUT / "01_overall_score_ranking.png", dpi=200); plt.close(fig)


# ---- 2. per database vs frontier --------------------------------------------
def chart_vs_frontier():
    names = ["Qwen3.8-27B-ScientificBM (ours)", "Claude 4.5 Sonnet", "Gemini 3 Pro", "GPT-5.1-Codex", "GPT-5.1"]
    cols = [COL["Ours"], "#4B5366", "#6D7587", "#9AA2B3", "#C3C8D3"]
    groups = [("NCBI", "score_ncbi"), ("UniProt", "score_uniprot"), ("Ensembl", "score_ensembl"), ("Overall", "score_overall")]
    by = {r["model"]: r for r in ROWS}
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.14)
    w = 0.16
    for j, (n, c) in enumerate(zip(names, cols)):
        xs = [i + (j - 2) * w for i in range(4)]
        vals = [by[n][k] for _, k in groups]
        bars = ax.bar(xs, vals, width=w * 0.92, color=c, label=SHORT(n), zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8,
                    color=COL["Ours"] if j == 0 else MUTED, weight="bold" if j == 0 else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels([g for g, _ in groups], fontsize=11)
    ax.set_ylim(0, 108); ax.set_ylabel("BioTool Score", fontsize=10)
    ax.yaxis.grid(True, color="#ECEEF2", zorder=0); ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.09), frameon=False, fontsize=9)
    titles(fig, "Ahead of every frontier model on every database",
           "BioTool Score by database. The narrowest lead is on UniProt (90.8 vs 87.8 for Gemini 3 Pro).")
    footer(fig)
    fig.savefig(OUT / "02_vs_frontier_by_database.png", dpi=200); plt.close(fig)


# ---- 3. exact match vs api success ------------------------------------------
def chart_scatter():
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.13)
    # label positions in data coordinates (x = API success, y = Exact Match); thin lines link label to point
    pos = {
        "GPT-5.1": (63.0, 14.0), "GPT-5.1-Codex": (66.6, 10.0), "Llama3.1-8B (base)": (67.8, 5.6),
        "Qwen3-8B (base)": (59.2, 9.5), "Qwen2.5-7B (base)": (76.5, 0.2), "Qwen3-4B (base)": (77.5, 9.0),
        "Gemini 3 Pro": (82.0, 11.0), "Claude 4.5 Sonnet": (88.5, 11.0),
        "Qwen3-8B (BioTool-FT)": (84.5, 24.5), "Qwen2.5-7B (BioTool-FT)": (84.5, 36.0),
        "Llama3.1-8B (BioTool-FT)": (96.0, 24.0), "Qwen3-4B (BioTool-FT)": (91.5, 43.5),
    }
    for r in ROWS:
        ours = r["group"] == "Ours"
        ax.scatter(r["as_overall"], r["em_overall"], s=330 if ours else 95, marker="*" if ours else "o",
                   color=COL[r["group"]], edgecolor="white", linewidth=1, zorder=4)
        label = SHORT(r["model"]).replace(" + BioTool FT", " (FT)")
        xy = (r["as_overall"], r["em_overall"])
        if ours:
            ax.annotate(label, xy, xytext=(84.0, 31.5), fontsize=10.5, weight="bold", color=COL["Ours"], va="center",
                        arrowprops=dict(arrowstyle="-", color=COL["Ours"], lw=1.0, shrinkA=2, shrinkB=6))
        else:
            ax.annotate(label, xy, xytext=pos[r["model"]], fontsize=8.8, color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color="#B8BCC6", lw=0.8, shrinkA=2, shrinkB=4))
    ax.set_xlim(58, 101); ax.set_ylim(-2, 47)
    ax.set_xlabel("API Success (%)  -  does the call run and return data?", fontsize=10)
    ax.set_ylabel("Exact Match (%)  -  is the call identical to the reference?", fontsize=10)
    ax.grid(True, color="#ECEEF2", zorder=0)
    ax.legend(handles=LEGEND, loc="upper left", frameon=False, fontsize=8.5)
    titles(fig, "Frontier models often pick the right tool, but rarely with the exact right settings",
           "Top-right is best. Fine-tuning on BioTool moves a model from the bottom-left to the top-right.")
    footer(fig)
    fig.savefig(OUT / "03_exact_match_vs_api_success.png", dpi=200); plt.close(fig)


# ---- 4. gap vs each model ----------------------------------------------------
def chart_gap():
    others = [r for r in ROWS if r["group"] != "Ours"]
    data = sorted(((r["model"], OURS["score_overall"] - r["score_overall"], r["group"]) for r in others), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.84, bottom=0.12)
    for i, (n, d, g) in enumerate(data):
        ax.barh(i, d, color="#2E9E6B" if d > 0 else "#D1495B", height=0.66, zorder=3)
        ax.text(d + (0.9 if d > 0 else -0.9), i, f"{d:+.1f}", va="center", ha="left" if d > 0 else "right", fontsize=10, weight="bold",
                color="#2E9E6B" if d > 0 else "#D1495B")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(data))); ax.set_yticklabels([SHORT(n) for n, _, _ in data], fontsize=10)
    ax.set_xlim(-14, 42); ax.set_xlabel("ScientificBM score minus the other model's score (points)", fontsize=10)
    ax.xaxis.grid(True, color="#ECEEF2", zorder=0); ax.set_axisbelow(True)
    titles(fig, "Ahead of 11 of the 12 models, behind only the Qwen3-4B fine-tune",
           "Overall BioTool Score difference. Green = ScientificBM is ahead. Uses the cautious score of 91.6.")
    footer(fig)
    fig.savefig(OUT / "04_gap_vs_each_model.png", dpi=200); plt.close(fig)


# ---- 5/6. simple pipeline diagrams ------------------------------------------
def pipeline(fname, title, sub, steps, accent):
    n = len(steps)
    fig, ax = plt.subplots(figsize=(14, 4.9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.9); ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.text(0.2, 4.55, title, fontsize=17, weight="bold", color=INK, va="center")
    ax.text(0.2, 4.1, sub, fontsize=10.5, color=MUTED, va="center")
    bw, gap = 2.3, 0.45
    x0 = (14 - (n * bw + (n - 1) * gap)) / 2
    for i, (head, body, note) in enumerate(steps):
        x = x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, 1.1), bw, 2.2, boxstyle="round,pad=0.02,rounding_size=0.18", fc="#F6F7F9", ec="#D5D9E0", lw=1.2))
        ax.add_patch(plt.Circle((x + 0.34, 3.0), 0.22, color=accent, zorder=5))
        ax.text(x + 0.34, 3.0, str(i + 1), color="white", ha="center", va="center", fontsize=10.5, weight="bold", zorder=6)
        ax.text(x + 0.72, 3.0, head, fontsize=11, weight="bold", color=INK, va="center")
        ax.text(x + 0.18, 2.62, body, fontsize=9.3, color="#374151", va="top", linespacing=1.45)
        ax.text(x + bw / 2, 0.72, note, fontsize=8.8, color=accent, ha="center", va="center", style="italic")
        if i < n - 1:
            ax.annotate("", xy=(x + bw + gap - 0.04, 2.2), xytext=(x + bw + 0.04, 2.2),
                        arrowprops=dict(arrowstyle="-|>", color="#9AA2B3", lw=1.6))
    fig.savefig(OUT / fname, dpi=200); plt.close(fig)


def chart_pipelines():
    pipeline("05_finetuning_strategy.png", "How we fine-tuned the model", "One A100 40GB GPU, one pass over the BioTool training questions.",
             [("Training data", "5,632 questions, each\npaired with the one\ncorrect tool call\n(BioTool train split)", "gxx27/BioTool"),
              ("Prepare", "Show the tools in the\nprompt, same format the\nmodel uses at run time.\nThinking switched off", "chat template"),
              ("Filter", "Drop any example\nlonger than 3,072\ntokens.\nNothing is cut short", "no truncation"),
              ("Train (QLoRA)", "Qwen3.8-27B kept in\n4-bit; only small LoRA\nadapters are trained\n(rank 16, 1 epoch)", "learns from the answer only"),
              ("Publish", "Merge the adapters into\nthe full model, save\nas 16-bit, upload to\nHugging Face", "Rumiii/Qwen3.8-27B-ScientificBM")],
             COL["BioTool-FT"])
    pipeline("06_benchmark_strategy.png", "How we tested it", "Same test set and same scoring code as the BioTool paper, so numbers are comparable.",
             [("Test questions", "1,408 questions the\nmodel never saw during\ntraining (held-out\ntest split)", "gxx27/BioTool"),
              ("Model answers", "Writes one tool call per\nquestion. Run as a 3-bit\nversion so it fits one\n16GB T4 (about 5 hours)", "greedy, no sampling"),
              ("Exact Match", "Is the call identical to\nthe reference call?\nSame tool, same\nsettings", "strict"),
              ("Run on real data", "Both calls are run on the\nlive NCBI, UniProt and\nEnsembl services", "API Success = it ran"),
              ("BioTool Score", "Compare what came back.\nA text-similarity model\n(MedCPT) rates how close\nthe two results are", "partial credit")],
             COL["Ours"])


if __name__ == "__main__":
    chart_ranking(); chart_vs_frontier(); chart_scatter(); chart_gap(); chart_pipelines()
    print("saved to", OUT)
