cv##import matplotlib.pyplot as plt
import numpy as np
#

# --------------------------------------------------
# DATA
# --------------------------------------------------

models = ["GPT", "Claude", "Grok", "Gemini"]

bleu1 = [0.1584, 0.1569, 0.1481, 0.1307]
bleu2 = [0.0290, 0.0228, 0.0170, 0.0153]
meteor = [0.2052, 0.1945, 0.1858, 0.1433]
rouge = [0.1374, 0.1326, 0.1357, 0.0907]

# --------------------------------------------------
# BAR POSITIONS
# --------------------------------------------------

x = np.arange(len(models))
width = 0.18

# --------------------------------------------------
# CREATE FIGURE
# --------------------------------------------------

plt.figure(figsize=(11, 6))

bar1 = plt.bar(
    x - 1.5 * width,
    bleu1,
    width,
    label="BLEU-1"
)

bar2 = plt.bar(
    x - 0.5 * width,
    bleu2,
    width,
    label="BLEU-2"
)

bar3 = plt.bar(
    x + 0.5 * width,
    meteor,
    width,
    label="METEOR-like"
)

bar4 = plt.bar(
    x + 1.5 * width,
    rouge,
    width,
    label="ROUGE-L"
)

# --------------------------------------------------
# VALUE LABELS
# --------------------------------------------------

def add_values(bars):
    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

add_values(bar1)
add_values(bar2)
add_values(bar3)
add_values(bar4)

# --------------------------------------------------
# GRAPH DESIGN
# --------------------------------------------------

plt.xticks(
    x,
    models,
    fontsize=12,
    fontweight="bold"
)

plt.xlabel(
    "Models",
    fontsize=12
)

plt.ylabel(
    "Score",
    fontsize=12
)

plt.title(
    "LLM Comparison for Dataset Augmentation",
    fontsize=16,
    fontweight="bold"
)

plt.ylim(0, 0.23)

plt.legend(
    title="Evaluation Metrics"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

# --------------------------------------------------
# SAVE IMAGE
# --------------------------------------------------

plt.savefig(
    "model_comparison.png",
    dpi=400,
    bbox_inches="tight"
)

# --------------------------------------------------
# DISPLAY IMAGE
# --------------------------------------------------

plt.show()

print("Plot generated successfully!")
print("Saved as: model_comparison.png")
