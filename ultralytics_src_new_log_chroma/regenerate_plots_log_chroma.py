import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Use Agg backend for clusters with no display (HPC-safe)
matplotlib.use("Agg")

# Absolute path to log-chroma results CSV
csv_path = "/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics-main/runs/train_log_chroma/exp_log_chroma/results.csv"

# Output directory for saving figures
out_dir = "/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics-main/runs/train_log_chroma/exp_log_chroma"

df = pd.read_csv(csv_path)

sns.set_theme(style="whitegrid")

################################################################################
# 1. TRAINING LOSS PLOT (Log-Chroma YOLO)
################################################################################
plt.figure(figsize=(12, 6))

sns.lineplot(data=df, x="epoch", y="train/box_loss", label="Box Loss")
sns.lineplot(data=df, x="epoch", y="train/cls_loss", label="Class Loss")
sns.lineplot(data=df, x="epoch", y="train/dfl_loss", label="DFL Loss")

plt.title("YOLO Log-Chroma Training Loss", fontsize=18, weight="bold")
plt.suptitle("YOLOv11: Box, Class, and Distribution Focal Loss Across Epochs (Log-Chromacity Input)", fontsize=12)

plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Loss Value", fontsize=13)
plt.legend(title="Loss Types")

plt.tight_layout()
plt.savefig(f"{out_dir}/seaborn_loss.png", dpi=300)
plt.close()

################################################################################
# 2. mAP50-95 PLOT (Log-Chroma YOLO)
################################################################################
map_col = "metrics/mAP50-95(B)"

if map_col not in df.columns:
    print("ERROR: mAP50-95 column not found. Available columns:")
    print(df.columns)
else:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="epoch", y=map_col, label="mAP50–95", color="darkgreen")

    plt.title("YOLO Log-Chroma Validation Performance", fontsize=18, weight="bold")
    plt.suptitle("YOLOv11 — Mean Average Precision (IoU 0.50 to 0.95) Over Epochs", fontsize=12)

    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("mAP Score", fontsize=13)
    plt.ylim(0, max(df[map_col].max(), 0.1) * 1.1)
    plt.legend(title="Metric")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/seaborn_map.png", dpi=300)
    plt.close()

print("Done! Saved log-chroma seaborn plots with YOLOv11 titles.")
