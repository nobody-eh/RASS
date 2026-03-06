#!/usr/bin/env python3
"""
3D UMAP + clustering + interactive Plotly 3D scatter saved to a single HTML file.
Optional: embed first image per dish as base64 and show it on hover (overlay in top-right).
"""

import os
import base64
import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

import umap
import plotly.express as px
import plotly.io as pio

# Optional static plotting
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def pick_representatives_centroid(ids, X_scaled, labels, n_per_cluster=1):
    reps = []
    for c in np.unique(labels):
        mask = (labels == c)
        Xc = X_scaled[mask]
        ids_c = ids[mask]
        centroid = Xc.mean(axis=0)
        # compute Euclidean distance to centroid
        dists = np.linalg.norm(Xc - centroid, axis=1)
        idx_sorted = np.argsort(dists)
        selected = ids_c[idx_sorted[:n_per_cluster]]
        reps.extend(selected.tolist())
    return reps

def pick_representatives_medoid(ids, X_scaled, labels):
    from scipy.spatial.distance import cdist
    reps = []
    for c in np.unique(labels):
        mask = (labels == c)
        Xc = X_scaled[mask]
        ids_c = ids[mask]
        # compute pairwise distance matrix within the cluster
        if Xc.shape[0] == 0:
            continue
        # If only one element, that is the representative
        if Xc.shape[0] == 1:
            reps.append(ids_c[0])
            continue
        D = cdist(Xc, Xc, metric='euclidean')  # shape (n_c, n_c)
        sum_dists = D.sum(axis=1)  # sum of distances from each point to others in cluster
        medoid_idx = np.argmin(sum_dists)
        reps.append(ids_c[medoid_idx])
    return reps

# --------------------------
# Utility functions
# --------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'dish_id' not in df.columns:
        raise ValueError("CSV must include 'dish_id' column")
    return df

def preprocess_features(df, drop_cols=None):
    df2 = df.copy()
    if drop_cols:
        for c in drop_cols:
            if c in df2.columns:
                df2 = df2.drop(columns=[c])
    # Keep dish_id separately
    ids = df2['dish_id'].astype(str).values
    # features = df2.drop(columns=["dish_id", "edge_frac_mean", "area_frac_mean", "mean_visible_points", "fl_x", "k1", "extent_x", "extent_y", "extent_z", "camera_angle_x", "camera_angle_y", "median_track_length", "mean_view_angle_var", "mean_error", "median_error", "largest_comp_frac_mean", "area_frac_std", "bbox_aspect_ratio_mean", "bbox_aspect_ratio_std", "bbox_max_x", "bbox_max_y", "bbox_max_z", "bbox_min_x", "bbox_min_y", "bbox_min_z", "edge_frac_std", "entropy_std", "largest_comp_frac_std", "mean_visible_ratio", "num_components_std", "sharpness_std", "std_camera_center_dist", "std_error", "std_track_length", "std_view_angle_var", "std_visible_points", "std_visible_ratio"])

    features = df2.drop(columns=['dish_id'])
    # Fill NaNs
    features = features.fillna(features.mean())
    # Drop zero variance
    variances = features.var(axis=0)
    zero_var = variances[variances == 0.0].index.tolist()
    if zero_var:
        print("Dropping zero-variance features:", zero_var)
        features = features.drop(columns=zero_var)
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return ids, X_scaled, features.columns.tolist(), features

def choose_clusters(X_scaled, k_min=2, k_max=10):
    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=0)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"Silhouette score for k={k}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"-> Best k by silhouette: {best_k}")
    return best_k

def cluster_data(X_scaled, method='kmeans', n_clusters=5, **kwargs):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=0, **kwargs)
    elif method == 'agglomerative':
        linkage = kwargs.get('linkage', 'ward')
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    else:
        raise ValueError("Unsupported clustering method")
    labels = model.fit_predict(X_scaled)
    return labels, model

def embed_umap_3d(X_scaled, n_neighbors=15, min_dist=0.1, random_state=0):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=3, random_state=random_state)
    emb = reducer.fit_transform(X_scaled)
    return emb

# --------------------------
# Image helpers (first image in dish images folder)
# --------------------------
def get_first_image_path_for_dish(dish_id, base_images_dir):
    dish_img_dir = os.path.join(base_images_dir, str(dish_id), "images")
    if not os.path.isdir(dish_img_dir):
        return None
    files = [f for f in os.listdir(dish_img_dir)
             if os.path.isfile(os.path.join(dish_img_dir, f))
             and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not files:
        return None
    files = sorted(files)
    return os.path.join(dish_img_dir, files[0])

def image_to_data_uri(img_path, max_side=None):
    """
    Convert an image file to base64 data URI. Optional resizing (thumbnail) can be added here
    if you want to pre-generate thumbnails (not implemented to avoid PIL dependency).
    """
    if img_path is None or not os.path.isfile(img_path):
        return None
    ext = os.path.splitext(img_path)[1].lower()
    mime = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png'
    with open(img_path, 'rb') as f:
        b = f.read()
    b64 = base64.b64encode(b).decode('utf-8')
    return f"data:{mime};base64,{b64}"

# --------------------------
# Plot + HTML generation (3D)
# --------------------------
def build_and_save_3d(df, emb3d, labels, output_prefix, feature_x=None, feature_y=None,
                      include_images=False, base_images_dir=None, image_mapping_cache=None,
                      static_png=True):
    """
    emb3d: (N,3) embedding array
    labels: cluster labels (N,)
    include_images: if True, embed images as data URIs and inject JS to show overlay on hover
    """
    plot_df = pd.DataFrame({
        'UMAP1': emb3d[:,0],
        'UMAP2': emb3d[:,1],
        'UMAP3': emb3d[:,2],
        'dish_id': df['dish_id'].astype(str).values,
        'cluster': labels.astype(str),
    })
    # add optional hover columns
    for col in ('mean_error', 'point_density', 'mean_visible_ratio'):
        if col in df.columns:
            plot_df[col] = df[col].fillna(0).values

    # Build 3D figure with Plotly Express
    fig = px.scatter_3d(plot_df, x='UMAP1', y='UMAP2', z='UMAP3',
                        color='cluster',
                        hover_data=['dish_id'] + [c for c in ('mean_error','point_density','mean_visible_ratio') if c in plot_df.columns],
                        title="3D UMAP embedding colored by cluster")
    fig.update_traces(marker=dict(size=4, opacity=0.8))

    # Save a static 3D PNG (optional)
    if static_png and MATPLOTLIB_AVAILABLE:
        try:
            fig_png = plt.figure(figsize=(10,8))
            ax = fig_png.add_subplot(111, projection='3d')
            for c in np.unique(labels):
                mask = labels == c
                ax.scatter(emb3d[mask,0], emb3d[mask,1], emb3d[mask,2], label=f"Cluster {c}", s=10, alpha=0.6)
            ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_zlabel('UMAP3')
            ax.legend()
            png_path = f"{output_prefix}_3d_static.png"
            fig_png.savefig(png_path, dpi=300)
            plt.close(fig_png)
            print("Saved static 3D PNG to", png_path)
        except Exception as e:
            print("Could not save static PNG:", e)

    # If images are requested, build mapping from point index -> data URI
    post_script = None
    if include_images:
        if base_images_dir is None:
            raise ValueError("base_images_dir is required when include_images=True")
        # Build mapping (index -> data uri or null)
        datauri_map_entries = []
        for idx, did in enumerate(plot_df['dish_id'].values):
            img_path = get_first_image_path_for_dish(did, base_images_dir)
            #uri = image_to_data_uri(img_path) if img_path else None
            #if uri is None:
            #    datauri_map_entries.append(f"{idx}: null")
            # else:
            # escape double quotes in data uri (shouldn't have any) and safe JS string
            #uri_js = uri.replace("\n", "")
            datauri_map_entries.append(f"{idx}: \"{img_path}\"")
        mapping_js = "{\n" + ",\n".join(datauri_map_entries) + "\n}"

        # Build JS to overlay an <img> element inside the HTML when hovering
        # Use {plot_id} placeholder; write_html will replace it with actual div id.
        post_script = f"""
        (function() {{
            var dataUriMap = {mapping_js};
            var gd = document.getElementById('{{plot_id}}');
            // create a container div for the image overlay
            var overlay = document.createElement('div');
            overlay.style.position = 'absolute';
            overlay.style.top = '10px';
            overlay.style.right = '10px';
            overlay.style.zIndex = 1000;
            overlay.style.pointerEvents = 'none';
            overlay.style.maxWidth = '30%';
            overlay.style.maxHeight = '40%';
            overlay.id = 'hover-image-overlay';
            gd.parentElement.style.position = 'relative';
            gd.parentElement.appendChild(overlay);
            console.log(dataUriMap);
            gd.on('plotly_hover', function(eventdata) {{
                var pt = eventdata.points[0];
                console.log(pt);
                var idx = pt.pointNumber;
                var uri = dataUriMap[idx];
                console.log(idx);
                console.log(uri);
                if (!uri) {{
                    overlay.innerHTML = "<div style='padding:8px; color:#666;'>No image</div>";
                    return;
                }}
                overlay.innerHTML = "<img src='" + uri + "' style='max-width:100%; max-height:100%; border:1px solid #ccc; display:block;' />";
            }});

            gd.on('plotly_unhover', function(eventdata) {{
                overlay.innerHTML = "";
            }});
        }})();
        """

    # Write HTML to disk (self-contained)
    html_path = f"{output_prefix}_3d_umap.html"
    # Use write_html with post_script and {plot_id} placeholder (replaced automatically)
    pio.write_html(fig, file=html_path, full_html=True, include_plotlyjs='cdn', post_script=post_script)
    print("Saved interactive 3D HTML to", html_path)

    # Also save cluster mapping and cluster centroids profiles
    mapping_df = pd.DataFrame({'dish_id': plot_df['dish_id'].values, 'cluster': labels})
    mapping_csv = f"{output_prefix}_dish_cluster_mapping.csv"
    mapping_df.to_csv(mapping_csv, index=False)
    print("Saved dish->cluster mapping to", mapping_csv)

    # Cluster profiles: mean feature values per cluster (use original features if available)
    try:
        feats_for_profile = df.drop(columns=['dish_id']).copy()
        feats_for_profile['cluster'] = labels
        profile = feats_for_profile.groupby('cluster').mean()
        profile_csv = f"{output_prefix}_cluster_profiles.csv"
        profile.to_csv(profile_csv)
        print("Saved cluster profiles to", profile_csv)
    except Exception as e:
        print("Could not save cluster profiles:", e)


#---------------------------
# Feature Selection
#---------------------------
def read_ingp_csv(path):
    """
    Read a file such as ingp_oc.csv or ingp_fi.csv.
    Expected columns: dish_id, PSNR, MIN, MAX, SSIM, MIN, MAX, psnr_avgmse
    We'll rename columns to avoid duplicates.
    """
    df = pd.read_csv(path)
    # Example: if there are duplicate column names (MIN, MAX) for PSNR and SSIM, rename
    # Suppose CSV has columns: dish_id, PSNR, MIN, MAX, SSIM, MIN, MAX, psnr_avgmse
    # Those MIN, MAX need to be disambiguated, e.g. PSNR_MIN, PSNR_MAX, SSIM_MIN, SSIM_MAX
    # Let’s assume the CSV columns are already unambiguous; if not, rename here:

    # For safety, rename duplicates (you’ll need to adjust names according to file)
    # e.g.: df = df.rename(columns={'MIN_x': 'PSNR_MIN', 'MAX_x': 'PSNR_MAX', 'MIN_y': 'SSIM_MIN', 'MAX_y': 'SSIM_MAX'})

    return df

def compute_metrics_for_reps(reps, oc_df, fi_df):
    """
    Given list of representative dish_ids, and the two dataframes,
    compute for each metric the average and std over those reps.
    Returns a dict.
    """
    rows = []
    for df_name, df in [('oc', oc_df), ('fi', fi_df)]:
        # Filter only reps present in this csv
        df_sub = df[df['dish_id'].astype(str).isin([str(r) for r in reps])]
        if df_sub.empty:
            print(f"No matching dish_id in {df_name} CSV for the representatives.")
            continue

        # Define which metric columns to average & std
        # Modify the column names based on your CSV
        # Example columns:
        #   'PSNR', 'PSNR_MIN', 'PSNR_MAX', 'SSIM', 'SSIM_MIN', 'SSIM_MAX', 'psnr_avgmse'

        # You may need to inspect df_sub.columns to pick correct names
        metric_cols = []
        for col in ['PSNR', 'PSNR_MIN', 'PSNR_MAX', 'SSIM', 'SSIM_MIN', 'SSIM_MAX', 'psnr_avgmse']:
            if col in df_sub.columns:
                metric_cols.append(col)
            else:
                print(f"Warning: column {col} not found in {df_name} data")

        means = df_sub[metric_cols].mean()
        stds = df_sub[metric_cols].std(ddof=0)  # or ddof=1 if you want sample‐std

        # Store
        for m in metric_cols:
            rows.append({
                'csv': df_name,
                'metric': m,
                'mean': means[m],
                'std': stds[m],
                'count': len(df_sub)
            })

    result_df = pd.DataFrame(rows)
    return result_df


# --------------------------
# Main script
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Features CSV (must include dish_id)")
    parser.add_argument("--method", choices=['kmeans','agglomerative'], default='kmeans')
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters; if not provided silhouette search used")
    parser.add_argument("--drop_cols", nargs='*', default=None, help="Columns to drop before clustering")
    parser.add_argument("--output_prefix", default="clustered_scenes", help="Output file prefix")
    parser.add_argument("--include_images", action='store_true', help="Embed first image per dish in HTML and show on hover (makes HTML large)")
    parser.add_argument("--base_images_dir", default=None, help="Base path to images directories (required if include_images=True). Example: /media/.../360_4")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--k_min", type=int, default=2, help="Silhouette search min k")
    parser.add_argument("--k_max", type=int, default=10, help="Silhouette search max k")
    parser.add_argument("--n_per_cluster", type=int, default=4, help="Number of selected dishes per cluster")
    parser.add_argument("--rep_method", choices=['centroid','medoid'], default='centroid', help="How to pick representatives")
    args = parser.parse_args()

    df = load_data(args.csv_path)
    ids, X_scaled, feat_names, features_df = preprocess_features(df, drop_cols=args.drop_cols)

    if args.n_clusters is None:
        print("Searching best k by silhouette...")
        best_k = choose_clusters(X_scaled, k_min=args.k_min, k_max=args.k_max)
    else:
        best_k = args.n_clusters

    print("Clustering with k =", best_k)
    labels, model = cluster_data(X_scaled, method=args.method, n_clusters=best_k)

    print("Running 3D UMAP embedding...")
    emb3d = embed_umap_3d(X_scaled, n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    build_and_save_3d(df, emb3d, labels, args.output_prefix,
                      include_images=args.include_images, base_images_dir=args.base_images_dir)

    # ---------------------------
    # Pick representatives
    # ---------------------------
    if args.rep_method == 'centroid':
        reps = pick_representatives_centroid(ids, X_scaled, labels, n_per_cluster=args.n_per_cluster)
    else:
        reps = pick_representatives_medoid(ids, X_scaled, labels)

    reps_path = f"{args.output_prefix}_representatives_{args.rep_method}.txt"
    with open(reps_path, 'w') as f:
        for did in reps:
            f.write(str(did) + "\n")
    print("Representative dish_ids saved to:", reps_path)

    path_oc = 'ingp_oc.csv'
    path_fi = 'ingp_fi.csv'
    oc_df = read_ingp_csv(path_oc)
    fi_df = read_ingp_csv(path_fi)

    metrics_df = compute_metrics_for_reps(reps, oc_df, fi_df)

    # Save summary
    out_metrics_path = f"{args.output_prefix}_reps_ingp_metrics_summary.csv"
    metrics_df.to_csv(out_metrics_path, index=False)
    print("Saved summary metrics for representatives to", out_metrics_path)

    # Optionally, print them:
    print(metrics_df)

if __name__ == "__main__":
    main()
