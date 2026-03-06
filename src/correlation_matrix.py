import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load
df = pd.read_csv("feats_normalized.csv")

# Pick numeric cols
numeric = df.select_dtypes(include=[np.number]).copy()
# if 'dish_id' in numeric.columns:
# numeric = numeric.drop(columns=["edge_frac_mean", "area_frac_mean", "mean_visible_points", "fl_x", "k1", "extent_x", "extent_y", "extent_z", "camera_angle_x", "camera_angle_y", "median_track_length", "mean_view_angle_var", "mean_error", "median_error", "largest_comp_frac_mean", "area_frac_std", "bbox_aspect_ratio_mean", "bbox_aspect_ratio_std", "bbox_max_x", "bbox_max_y", "bbox_max_z", "bbox_min_x", "bbox_min_y", "bbox_min_z", "edge_frac_std", "entropy_std", "largest_comp_frac_std", "mean_visible_ratio", "num_components_std", "sharpness_std", "std_camera_center_dist", "std_error", "std_track_length", "std_view_angle_var", "std_visible_points", "std_visible_ratio"])
# numeric = numeric.drop(columns=["dish_id"])

# Drop cols with too many missing or zero variance
numeric = numeric.dropna(axis=1, thresh=0.5 * len(numeric))
numeric = numeric.loc[:, numeric.var() > 0]

# Optional: log1p transform for skewed features
for col in numeric.columns:
    if numeric[col].max() / (numeric[col].median() + 1e-6) > 100:  # heuristic
        numeric[col] = np.log1p(numeric[col])

# Compute correlation
corr = numeric.corr()

# Mask upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Plot
plt.figure(figsize=(14,12))
ax = sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.75},
    xticklabels=True,
    yticklabels=True
)
# Adjust ticks
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig("feats2_corr_matrix_aligned.png", dpi=300)
plt.show()
