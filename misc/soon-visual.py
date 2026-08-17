import numpy as np

npz_path = "./visualization/keypoint_heatmaps/SKL_full_validation_statistics.npz"
data = np.load(npz_path, allow_pickle=True)

for name in ["gt_coords", "argmax_coords", "skl_coords"]:
    arr = data[name]
    print(name)
    print("  x min/max:", arr[:, 0].min(), arr[:, 0].max())
    print("  y min/max:", arr[:, 1].min(), arr[:, 1].max())
    print("  joint id min/max:", arr[:, 2].min(), arr[:, 2].max())

print("mean arg error:", data["argmax_error"].mean())
print("mean skl error:", data["skl_error"].mean())
print("mean_arg_per_joint:", data["mean_arg_per_joint"])
print("mean_skl_per_joint:", data["mean_skl_per_joint"])