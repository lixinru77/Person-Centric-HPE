import os
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. Load npz statistics
# =============================
npz_path = "./visualization/keypoint_heatmaps/SKL_full_validation_statistics.npz"
data = np.load(npz_path, allow_pickle=True)

print("Available keys:", data.files)

argmax_error = data["mean_arg_per_joint"]   # (14,)
skl_error = data["mean_skl_per_joint"]      # (14,)
joint_names = [str(x) for x in data["joint_names"]]

print("argmax_error:", argmax_error.shape)
print("skl_error:", skl_error.shape)
print("joint_names:", joint_names)

# =============================
# 2. Plot style
# =============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "DejaVu Serif"],

    "font.size": 13,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,

    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# =============================
# 3. Draw figure
# =============================
x = np.arange(len(joint_names))
bar_width = 0.36

fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.bar(
    x - bar_width / 2,
    argmax_error,
    width=bar_width,
    label="Argmax",
    edgecolor="black",
    linewidth=0.35
)

ax.bar(
    x + bar_width / 2,
    skl_error,
    width=bar_width,
    label="SKL",
    edgecolor="black",
    linewidth=0.35
)

# =============================
# 4. Axes settings
# =============================
ax.set_ylabel("Localization Error (pixel)", fontsize=19)
ax.set_xlabel("Keypoints", fontsize=19)

ax.set_xticks(x)
ax.set_xticklabels(
    joint_names,
    rotation=45,
    ha="right",
    fontsize=17   # 这里就是横坐标字体大小
)

y_max = max(float(np.max(argmax_error)), float(np.max(skl_error)))
ax.set_ylim(0, y_max * 1.12)

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
ax.set_axisbelow(True)

# =============================
# 5. Legend and border
# =============================
legend = ax.legend(
    loc="upper right",
    frameon=True,
    borderpad=0.4,
    labelspacing=0.35,
    handlelength=1.5
)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(0.7)

for spine in ax.spines.values():
    spine.set_linewidth(1.1)
    spine.set_color("black")

# =============================
# 6. Print summary
# =============================
argmax_mean = np.mean(argmax_error)
skl_mean = np.mean(skl_error)
improvement = (argmax_mean - skl_mean) / argmax_mean * 100

print(f"Argmax mean error: {argmax_mean:.4f} pixel")
print(f"SKL mean error: {skl_mean:.4f} pixel")
print(f"Relative improvement: {improvement:.2f}%")

# =============================
# 7. Save
# =============================
fig.subplots_adjust(
    left=0.08,
    right=0.985,
    bottom=0.30,
    top=0.95
)

save_dir = "./visualization/keypoint_heatmaps/"
os.makedirs(save_dir, exist_ok=True)

png_path = os.path.join(save_dir, "argmax_vs_skl_error_bar.png")
pdf_path = os.path.join(save_dir, "argmax_vs_skl_error_bar.pdf")

fig.savefig(png_path, bbox_inches="tight", dpi=600)
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print("Saved PNG:", png_path)
print("Saved PDF:", pdf_path)