import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler, RobustScaler
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None


def _require_plot_libs():
    if plt is None or sns is None:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib and seaborn. "
            "Install them or run without plotting."
        )

def load_feats(csv_path):
    df = pd.read_csv(csv_path)
    return df

def inspect_features(df, id_col='dish_id'):
    """
    Compute summary statistics: num, mean, std, min, max, skew
    Return a DataFrame with these stats for numeric features
    """
    df_num = df.select_dtypes(include=[np.number]).copy()
    if id_col in df_num.columns:
        df_num = df_num.drop(columns=[id_col])
    desc = df_num.describe().transpose()
    desc['skew'] = df_num.skew().values
    # You might want also kurtosis etc.
    return desc

def select_log_transform(feat_stats, skew_thresh=1.0, range_ratio_thresh=100.0):
    """
    Decide which features to log1p transform.
    - skewed: absolute skew > skew_thresh
    - large range: max/min ratio > range_ratio_thresh
    Returns list of feature names.
    """
    to_log = []
    for feat, row in feat_stats.iterrows():
        # Skip non-positive minimums (we'll use log1p => need non-negative, so clip)
        if row['min'] < 0:
            # Could still log1p after shifting, but simpler to skip for now
            continue
        # Avoid dividing by zero
        if row['min'] == 0:
            ratio = np.inf
        else:
            ratio = row['max'] / (row['min'] + 1e-12)
        if (abs(row['skew']) > skew_thresh) or (ratio > range_ratio_thresh):
            to_log.append(feat)
    return to_log

def normalize_features(df, drop_cols=None, id_col='dish_id', skew_thresh=1.0, range_ratio_thresh=100.0, clip_std=3.0):
    """
    Normalize dataframe:
    - drop id column & optionally other columns
    - inspect stats
    - log1p on selected features
    - robust scale on skewed features
    - standard scale on others
    - clip values to ± clip_std
    Return normalized df, plus info about which features were log-transformed or robust scaled.
    """
    df_orig = df.copy()
    df_num = df_orig.select_dtypes(include=[np.number]).copy()
    # drop ID
    if id_col in df_num.columns:
        df_num = df_num.drop(columns=[id_col])
    # drop any additional cols
    if drop_cols:
        for c in drop_cols:
            if c in df_num.columns:
                df_num = df_num.drop(columns=[c])
    # fill missing
    df_num = df_num.fillna(df_num.mean())

    # compute stats
    stats = inspect_features(pd.concat([df_orig[[id_col]], df_num], axis=1), id_col=id_col)
    # select which to log
    log_feats = select_log_transform(stats, skew_thresh=skew_thresh, range_ratio_thresh=range_ratio_thresh)
    print("Features selected for log1p:", log_feats)

    # apply log1p
    df_trans = df_num.copy()
    for feat in log_feats:
        df_trans[feat] = np.log1p(df_trans[feat])

    # After log, recompute stats to decide scaling
    stats2 = df_trans.describe().transpose()
    stats2['skew'] = df_trans.skew().values
    # Decide which features to use RobustScaler vs StandardScaler
    # Let's use robust for features with high skew or large IQR/variance
    robust_feats = stats2[stats2['skew'].abs() > skew_thresh].index.tolist()
    standard_feats = [f for f in df_trans.columns if f not in robust_feats]

    print("Robust scaling features:", robust_feats)
    print("Standard scaling features:", standard_feats)

    # Apply scalers
    df_scaled = pd.DataFrame(index=df_trans.index, columns=df_trans.columns, dtype=float)

    if robust_feats:
        rs = RobustScaler()
        df_scaled[robust_feats] = rs.fit_transform(df_trans[robust_feats])
    if standard_feats:
        ss = StandardScaler()
        df_scaled[standard_feats] = ss.fit_transform(df_trans[standard_feats])

    # Clip extremes
    df_scaled = df_scaled.clip(lower=-clip_std, upper=clip_std)

    # Combine with non-numeric / id columns
    df_out = pd.concat([df_orig[[id_col]].reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

    return df_out, {
        'log_feats': log_feats,
        'robust_feats': robust_feats,
        'standard_feats': standard_feats,
        'stats_before': stats,
        'stats_after_log': stats2
    }

def plot_before_after(df_before, df_after, feats, id_col='dish_id', output_png="hist_normalization.png"):
    """
    Plot histograms of selected features before + after normalization for comparison.
    feats: list of feature names (subset)
    Each feature gets two subplots: before and after.
    """
    _require_plot_libs()

    # Count how many features
    m = len(feats)
    if m == 0:
        print("No features to plot.")
        return

    # We want 2 plots per feature (before + after).
    # Let's pick layout: 2 columns (before + after), and enough rows = number of features.
    ncols = 2
    nrows = m

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    for i, feat in enumerate(feats):
        if feat not in df_before.columns or feat not in df_after.columns:
            print(f"Feature {feat} not in both dataframes; skipping.")
            continue

        ax_before = axes[i][0]
        ax_after = axes[i][1]

        sns.histplot(df_before[feat].dropna(), bins=30, kde=True, ax=ax_before, color='blue')
        ax_before.set_title(f"Before: {feat}")

        sns.histplot(df_after[feat].dropna(), bins=30, kde=True, ax=ax_after, color='green')
        ax_after.set_title(f"After: {feat}")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Normalize feats.csv")
    parser.add_argument("csv_path", help="Path to feats.csv")
    parser.add_argument("--output_csv", default="feats_normalized.csv", help="Output normalized CSV")
    parser.add_argument("--drop_cols", nargs='*', default=None, help="Columns to drop before normalization (non-numeric or unwanted)")
    parser.add_argument("--id_col", default="dish_id", help="ID column name")
    parser.add_argument("--skew_thresh", type=float, default=1.0, help="Skewness threshold for choosing log and robust scaling")
    parser.add_argument("--range_ratio_thresh", type=float, default=100.0, help="Max/min ratio threshold for log transform")
    parser.add_argument("--clip_std", type=float, default=3.0, help="Clip scaled values to ± this many std devs")
    parser.add_argument("--plot_feats", nargs='*', default=None, help="List of features to plot before/after (if none, pick some automatically)")
    parser.add_argument("--plots_dir", default="plots", help="Directory to save plots")
    args = parser.parse_args()

    df = load_feats(args.csv_path)
    df_norm, info = normalize_features(df, drop_cols=args.drop_cols, id_col=args.id_col,
                                       skew_thresh=args.skew_thresh, range_ratio_thresh=args.range_ratio_thresh,
                                       clip_std=args.clip_std)

    # Save normalized CSV
    df_norm.to_csv(args.output_csv, index=False)
    print(f"Saved normalized data to {args.output_csv}")

    # Plot before/after if requested
    if args.plot_feats is not None:
        feats_to_plot = args.plot_feats
    else:
        # pick some features: maybe top 6 by variance
        stats = info['stats_before']
        # drop features with low variance
        high_var = stats.sort_values(by='std', ascending=False).head(6).index.tolist()
        feats_to_plot = high_var

    # Create plots directory if needed
    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir)

    plot_path = os.path.join(args.plots_dir, "before_after_hist.png")
    plot_before_after(df, df_norm, feats_to_plot, id_col=args.id_col, output_png=plot_path)

    print("Normalization info:")
    print("Log-transformed features:", info['log_feats'])
    print("Robust scaled features:", info['robust_feats'])
    print("Standard scaled features:", info['standard_feats'])

if __name__ == "__main__":
    main()
