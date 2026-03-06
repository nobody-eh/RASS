import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load
df = pd.read_csv("feats2.csv")

# Optionally, inspect what columns are numeric
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
print("Numeric columns:", numeric_cols)

# Optionally, drop columns you don’t want in the plot (too many, or redundant)
# For example, drop: id columns, volume if it's huge range, etc.
drop_cols = ['dish_id']  # if present
plot_cols = [c for c in numeric_cols if c not in drop_cols]

# If too many, select a subset
if len(plot_cols) > 10:
    # pick the most important ones, e.g.,
    plot_cols_small = ['point_density', 'mean_error', 'mean_visible_ratio',
                       'mean_camera_center_dist']
else:
    plot_cols_small = plot_cols

# Pairplot
sns.set(style="ticks", color_codes=True)
pair_grid = sns.pairplot(
    df[plot_cols_small],
)

# Save the figure
plt.tight_layout()
pair_grid.fig.suptitle("Pairwise relationships of selected features", y=1.02)
pair_grid.fig.savefig("feats2_pairplot.png", dpi=300)
plt.show()
