"""Draw the four diagrams requested in the task brief into images/.

Run from the project root:  python src/make_diagrams.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

IMG = Path(__file__).resolve().parent.parent / "images"
IMG.mkdir(exist_ok=True)

COLORS = ["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1", "#E15759", "#76B7B2", "#EDC948", "#9C755F"]


def box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="white", lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color="white", fontsize=fs, fontweight="bold", wrap=True)


def arrow(ax, p1, p2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                                 lw=1.8, color="#444", connectionstyle=f"arc3,rad={rad}"))


def lifecycle():
    steps = ["Problem\nDefinition", "Data\nCollection", "Data Cleaning\n& Prep", "Exploratory\nAnalysis",
             "Feature\nEngineering", "Model\nTraining", "Evaluation", "Deployment &\nMonitoring"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    xs = [0.3, 2.9, 5.5, 8.1]
    for i, s in enumerate(steps[:4]):
        box(ax, xs[i], 3.0, 2.4, 1.1, s, COLORS[i])
        if i < 3:
            arrow(ax, (xs[i] + 2.4, 3.55), (xs[i + 1], 3.55))
    for i, s in enumerate(steps[4:]):
        j = 3 - i
        box(ax, xs[j], 0.8, 2.4, 1.1, s, COLORS[4 + i])
        if i < 3:
            arrow(ax, (xs[j], 1.35), (xs[j - 1] + 2.4, 1.35))
    arrow(ax, (9.3, 3.0), (9.3, 1.9))
    ax.text(5.5, 0.12, "Iterate: monitor drift, retrain and improve", ha="center", fontsize=10, style="italic")
    ax.set_title("AI/ML Project Lifecycle", fontsize=14, fontweight="bold")
    fig.savefig(IMG / "diagram_ml_lifecycle.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def folder_structure():
    lines = [
        ("AI-ML-Internship/", ""),
        ("  data/", "raw and cleaned CSV datasets"),
        ("  notebooks/", "Jupyter notebooks (.ipynb)"),
        ("  src/", "reusable Python scripts"),
        ("  reports/", "PDF report and EDA write-ups"),
        ("  images/", "charts, diagrams, screenshots"),
        ("  models/", "saved models (used from Task 2)"),
        ("  requirements.txt", "pinned dependencies"),
        ("  README.md", "project documentation"),
        ("  .gitignore", "files excluded from Git (venv/ etc.)"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(lines) + 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 9.6, len(lines) + 0.4, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#F4F6F8", ec="#C9D1D9"))
    for i, (name, desc) in enumerate(lines):
        y = len(lines) - i
        is_dir = name.strip().endswith("/")
        ax.text(0.6, y, name, family="monospace", fontsize=12,
                fontweight="bold" if is_dir else "normal", color="#1F4E79" if is_dir else "#333", va="center")
        ax.text(5.0, y, desc, fontsize=10.5, color="#555", va="center")
    ax.set_title("Project Folder Structure", fontsize=14, fontweight="bold")
    fig.savefig(IMG / "diagram_folder_structure.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def eda_workflow():
    steps = ["Data\nLoading", "Cleaning", "Analysis", "Visualization", "Insights"]
    details = ["read_csv\nhead / info", "duplicates\nmissing values\nrename", "describe\nmean / median\ncorrelation",
               "histogram\nbox / scatter\nheatmap", "findings\nreport\nnext steps"]
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")
    for i, (s, d) in enumerate(zip(steps, details)):
        x = 0.2 + i * 2.35
        box(ax, x, 1.7, 1.9, 1.0, s, COLORS[i])
        ax.text(x + 0.95, 1.05, d, ha="center", va="center", fontsize=9, color="#333")
        if i < 4:
            arrow(ax, (x + 1.9, 2.2), (x + 2.35, 2.2))
    ax.set_title("Exploratory Data Analysis Workflow", fontsize=14, fontweight="bold")
    fig.savefig(IMG / "diagram_eda_workflow.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def git_workflow():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11); ax.set_ylim(-0.4, 4.4); ax.axis("off")
    boxes = [(0.2, "Working\nDirectory", "#4C78A8"), (3.1, "Staging Area\n(index)", "#F28E2B"),
             (6.0, "Local\nRepository", "#59A14F"), (8.9, "Remote\n(GitHub)", "#B07AA1")]
    for x, t, c in boxes:
        box(ax, x, 1.9, 1.9, 1.2, t, c)
    labels = ["git add", "git commit", "git push"]
    for i, l in enumerate(labels):
        x0 = boxes[i][0] + 1.9
        arrow(ax, (x0, 2.7), (boxes[i + 1][0], 2.7))
        ax.text((x0 + boxes[i + 1][0]) / 2, 2.95, l, ha="center", fontsize=8.5, family="monospace")
    arrow(ax, (9.85, 1.9), (1.15, 1.9), rad=-0.25)
    ax.text(5.5, -0.2, "git pull / git fetch  (sync remote changes back to local)", ha="center",
            fontsize=10, family="monospace")
    ax.text(5.5, 3.9, "git init  /  git clone  ->  start here", ha="center", fontsize=10, style="italic")
    ax.set_title("Git & GitHub Workflow", fontsize=14, fontweight="bold")
    fig.savefig(IMG / "diagram_git_workflow.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    lifecycle(); folder_structure(); eda_workflow(); git_workflow()
    print("Diagrams saved to", IMG)
