import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image

# Use Agg backend for clusters with no display
matplotlib.use("Agg")

# Absolute path to baseline results CSV
csv_path = "/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics_baseline/runs/train/exp/results.csv"

# Output directory
out_dir = "/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics_baseline/runs/train/exp"

df = pd.read_csv(csv_path)
sns.set_theme(style="whitegrid")

################################################################################
# 1. TRAINING LOSS PLOT
################################################################################
plt.figure(figsize=(12, 6))

sns.lineplot(data=df, x="epoch", y="train/box_loss", label="Box Loss")
sns.lineplot(data=df, x="epoch", y="train/cls_loss", label="Class Loss")
sns.lineplot(data=df, x="epoch", y="train/dfl_loss", label="DFL Loss")

plt.title("Training Loss Curves for Baseline Method", fontsize=18, weight="bold")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss Value", fontsize=14)
plt.legend(title="Loss Types")

plt.tight_layout()
plt.savefig(f"{out_dir}/loss_curve_baseline.png", dpi=300)
plt.close()


################################################################################
# 2. mAP50–95 PLOT
################################################################################
map_col = "metrics/mAP50-95(B)"

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="epoch", y=map_col, label="mAP50–95", color="darkgreen")

plt.title("mAP50–95 Curve for Baseline Method", fontsize=18, weight="bold")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("mAP50–95 Score", fontsize=14)
plt.legend(title="Metric")

plt.tight_layout()
plt.savefig(f"{out_dir}/map_curve_baseline.png", dpi=300)
plt.close()


################################################################################
# 3. PRECISION–RECALL CURVE (YOLO Saved Image)
################################################################################
pr_path = f"{out_dir}/PR_curve.png"

try:
    pr_img = np.array(Image.open(pr_path))

    plt.figure(figsize=(10, 8))
    plt.imshow(pr_img)
    plt.axis("off")
    plt.title("Precision–Recall Curve for Baseline Method", fontsize=18, weight="bold")

    plt.savefig(f"{out_dir}/PR_curve_titled.png", dpi=300, bbox_inches="tight")
    plt.close()

except FileNotFoundError:
    print("PR_curve.png not found — ensure YOLO validation was run.")


################################################################################
# 4. F1–CONFIDENCE CURVE
################################################################################
f1_path = f"{out_dir}/F1_curve.png"

try:
    f1_img = np.array(Image.open(f1_path))

    plt.figure(figsize=(10, 8))
    plt.imshow(f1_img)
    plt.axis("off")
    plt.title("F1–Confidence Curve for Baseline Method", fontsize=18, weight="bold")

    plt.savefig(f"{out_dir}/F1_curve_titled.png", dpi=300, bbox_inches="tight")
    plt.close()

except FileNotFoundError:
    print("F1_curve.png not found — ensure YOLO validation was run.")


################################################################################
# 5. CONFUSION MATRIX
################################################################################
cm_path = f"{out_dir}/confusion_matrix.png"

try:
    cm_img = np.array(Image.open(cm_path))

    plt.figure(figsize=(10, 10))
    plt.imshow(cm_img)
    plt.axis("off")
    plt.title("Confusion Matrix for Baseline Method", fontsize=18, weight="bold")

    plt.savefig(f"{out_dir}/confusion_matrix_titled.png", dpi=300, bbox_inches="tight")
    plt.close()

except FileNotFoundError:
    print("confusion_matrix.png not found — ensure YOLO validation was run.")

print("All plots generated with proper titles and axis labels.")
