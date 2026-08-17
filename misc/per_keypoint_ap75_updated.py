import os
import numpy as np
import matplotlib.pyplot as plt


# =============================
# 1. Data
# =============================
# 绘图顺序：按身体部位分组
# 注意：你的 Excel 表格顺序是：
# R_Hip, R_Knee, R_Ankle, L_Hip, L_Knee, L_Ankle, Neck, Head,
# L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist
#
# 这里为了画图美观，重新排列成：
# R_Hip, L_Hip, R_Knee, L_Knee, R_Ankle, L_Ankle, Neck, Head,
# R_Shoulder, L_Shoulder, R_Elbow, L_Elbow, R_Wrist, L_Wrist

keypoints = [
    "R_Hip", "L_Hip",
    "R_Knee", "L_Knee",
    "R_Ankle", "L_Ankle",
    "Neck", "Head",
    "R_Shoulder", "L_Shoulder",
    "R_Elbow", "L_Elbow",
    "R_Wrist", "L_Wrist"
]

base = [
    0.781, 0.775,
    0.668, 0.651,
    0.621, 0.607,
    0.736, 0.651,
    0.642, 0.641,
    0.359, 0.382,
    0.145, 0.204
]

mad = [
    0.786, 0.778,
    0.660, 0.647,
    0.594, 0.628,
    0.725, 0.660,
    0.636, 0.527,
    0.316, 0.348,
    0.144, 0.191
]

cctd = [
    0.987, 0.988,
    0.933, 0.934,
    0.936, 0.943,
    0.950, 0.930,
    0.919, 0.899,
    0.793, 0.800,
    0.560, 0.609
]

mad_cctd = [
    0.986, 0.987,
    0.933, 0.935,
    0.932, 0.942,
    0.956, 0.935,
    0.922, 0.911,
    0.784, 0.816,
    0.591, 0.627
]

full_model = [
    0.988, 0.988,
    0.960, 0.960,
    0.958, 0.969,
    0.970, 0.937,
    0.952, 0.929,
    0.803, 0.829,
    0.578, 0.562
]

x = np.arange(len(keypoints))


# =============================
# 2. Paper-style settings
# =============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times",
        "DejaVu Serif"
    ],
    "font.size": 11,

    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,

    "legend.fontsize": 10,

    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(7.2,3.8), dpi=300)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# =============================
# 3. Plot curves
# =============================
ax.plot(
    x, base,
    marker="o",
    linewidth=1.65,
    markersize=4.0,
    label="Base"
)

ax.plot(
    x, mad,
    marker="s",
    linewidth=1.65,
    markersize=4.0,
    label="+MAD"
)

ax.plot(
    x, cctd,
    marker="^",
    linewidth=1.75,
    markersize=4.2,
    label="+CCTD"
)

ax.plot(
    x, mad_cctd,
    marker="v",
    linewidth=1.75,
    markersize=4.2,
    label="MAD+CCTD"
)

ax.plot(
    x, full_model,
    marker="D",
    linewidth=1.85,
    markersize=4.0,
    label="MAD+CCTD+SKL"
)


# =============================
# 4. Axes
# =============================
ax.set_xlim(-0.4, len(keypoints) - 0.6)

# 关键修改：y 轴上限不要卡在 1.0，否则顶部类别文字会被挡住
ax.set_ylim(0.08, 1.1)

ax.set_xticks(x)
ax.set_xticklabels(keypoints, rotation=38, ha="right")
ax.set_yticks(np.arange(0.1, 1.01, 0.1))

ax.set_xlabel("Keypoints", fontsize=13)
ax.set_ylabel(r"Per-keypoint AP@0.75", fontsize=13)


# =============================
# 5. Grid and body-part separators
# =============================
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
ax.set_axisbelow(True)

# 身体部位分隔线
for pos in [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]:
    ax.axvline(pos, linestyle=":", linewidth=0.8, alpha=0.42)

# 身体部位标签
group_centers = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5]
group_labels = ["Hip", "Knee", "Ankle", "Head/Torso", "Shoulder", "Elbow", "Wrist"]

# 关键修改：放到 y=1.025，并且 y 轴上限设为 1.06，避免被顶部边框挡住
for center, label in zip(group_centers, group_labels):
    ax.text(
        center,
        1.025,
        label,
        ha="center",
        va="bottom",
        fontsize=12,
        clip_on=False
    )


# =============================
# 6. Spines and legend
# =============================
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

# 关键修改：图例缩小并放到左下角，避免遮挡 wrist/elbow 曲线
legend = ax.legend(
    loc="lower left",
    bbox_to_anchor=(0.018, 0.025),
    frameon=True,
    ncol=1,
    borderpad=0.32,
    labelspacing=0.22,
    handlelength=1.45,
    handletextpad=0.42,
    columnspacing=0.7
)

legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(0.65)
legend.get_frame().set_alpha(0.92)


# =============================
# 7. Layout
# =============================
# 关键修改：手动设置边距，比 tight_layout 更稳定
fig.subplots_adjust(
    left=0.072,
    right=0.992,
    bottom=0.265,
    top=0.900
)


# =============================
# 8. Save figure
# =============================
output_dir = "/mnt/newmy/MutliPHV-main/"
os.makedirs(output_dir, exist_ok=True)

png_path = os.path.join(output_dir, "per_keypoint_ap75_paper_fixed.png")
pdf_path = os.path.join(output_dir, "per_keypoint_ap75_paper_fixed.pdf")

fig.savefig(png_path, bbox_inches="tight", dpi=600)
fig.savefig(pdf_path, bbox_inches="tight")

plt.close(fig)

print("Saved PNG:", png_path)
print("Saved PDF:", pdf_path)