"""
04_genre_region_analysis.py
============================
風格雷達圖 + 地區對應分析

資料來源：Kaggle Spotify Tracks Dataset
  https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
  欄位：track_genre (114種), popularity, 音頻特徵

圖表清單：
  10_genre_radar.png        — 各風格音頻特徵雷達圖 (前12大genre)
  11_region_radar.png       — 地區對應雷達圖 (Asia/Western/LatinAm/Africa)
  12_genre_heatmap.png      — 風格 × 特徵均值熱圖
  13_genre_cluster.png      — 風格聚類樹狀圖 (Dendrogram)
  14_region_boxplot.png     — 地區 × 特徵箱形圖 (genre-based proxy)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import kruskal
from math import pi

os.makedirs("outputs/figures", exist_ok=True)

FEATURES = ["danceability", "energy", "valence", "acousticness",
            "loudness_norm", "speechiness", "instrumentalness", "liveness"]

# ── Genre → Region mapping (基於文化地理，有真實根據) ──────────────────────
REGION_MAP = {
    # Asia
    "k-pop": "Asia", "j-pop": "Asia", "mandopop": "Asia", "cantopop": "Asia",
    "anime": "Asia", "j-rock": "Asia", "j-idol": "Asia", "korean": "Asia",
    "j-dance": "Asia", "j-electronic": "Asia",
    # Latin America
    "latin": "Latin America", "reggaeton": "Latin America", "salsa": "Latin America",
    "samba": "Latin America", "bossanova": "Latin America", "tango": "Latin America",
    "mpb": "Latin America", "pagode": "Latin America", "axe": "Latin America",
    "forró": "Latin America", "sertanejo": "Latin America",
    # Africa
    "afrobeat": "Africa", "afropop": "Africa",
    # Middle East
    "iranian": "Middle East", "turkish": "Middle East",
    # Western (default for rest)
}

def get_region(genre):
    g = str(genre).lower().strip()
    for key, region in REGION_MAP.items():
        if key in g:
            return region
    return "Western"  # default


# ── 0. Load / Simulate ───────────────────────────────────────────────────────
def load_or_simulate(path="data/dataset.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "track_genre" in df.columns:
            df = df.rename(columns={"track_genre": "genre"})
        print(f"Loaded real data: {len(df):,} tracks, {df['genre'].nunique()} genres")
    else:
        print("Simulating data with realistic genre distributions...")
        np.random.seed(42)
        n = 40000

        # 114個genre模擬，重點genre特徵有差異
        genre_profiles = {
            "pop":            dict(d=0.65, e=0.65, v=0.60, a=0.20, l=-7,  s=0.05, i=0.05, lv=0.15),
            "k-pop":          dict(d=0.72, e=0.78, v=0.68, a=0.08, l=-5,  s=0.05, i=0.03, lv=0.12),
            "j-pop":          dict(d=0.58, e=0.62, v=0.62, a=0.25, l=-7,  s=0.03, i=0.08, lv=0.13),
            "hip-hop":        dict(d=0.75, e=0.60, v=0.45, a=0.12, l=-6,  s=0.22, i=0.04, lv=0.14),
            "rap":            dict(d=0.73, e=0.62, v=0.42, a=0.10, l=-6,  s=0.28, i=0.03, lv=0.13),
            "latin":          dict(d=0.78, e=0.76, v=0.74, a=0.15, l=-6,  s=0.06, i=0.04, lv=0.16),
            "reggaeton":      dict(d=0.82, e=0.80, v=0.72, a=0.06, l=-5,  s=0.08, i=0.03, lv=0.15),
            "classical":      dict(d=0.28, e=0.22, v=0.38, a=0.88, l=-18, s=0.02, i=0.82, lv=0.10),
            "edm":            dict(d=0.65, e=0.90, v=0.50, a=0.05, l=-5,  s=0.03, i=0.12, lv=0.18),
            "rock":           dict(d=0.50, e=0.82, v=0.48, a=0.15, l=-6,  s=0.05, i=0.06, lv=0.18),
            "country":        dict(d=0.55, e=0.55, v=0.62, a=0.45, l=-8,  s=0.04, i=0.05, lv=0.12),
            "r-n-b":          dict(d=0.70, e=0.58, v=0.55, a=0.20, l=-7,  s=0.08, i=0.04, lv=0.13),
            "jazz":           dict(d=0.45, e=0.38, v=0.52, a=0.72, l=-14, s=0.04, i=0.35, lv=0.15),
            "afrobeat":       dict(d=0.80, e=0.78, v=0.72, a=0.18, l=-6,  s=0.07, i=0.05, lv=0.17),
            "bossanova":      dict(d=0.52, e=0.40, v=0.60, a=0.68, l=-13, s=0.03, i=0.28, lv=0.11),
            "anime":          dict(d=0.55, e=0.70, v=0.65, a=0.22, l=-7,  s=0.04, i=0.10, lv=0.14),
        }

        genres_list = list(genre_profiles.keys())
        weights = [0.12,0.08,0.06,0.08,0.06,0.07,0.05,0.05,0.06,0.07,0.05,0.06,0.04,0.04,0.04,0.07]
        weights = np.array(weights) / sum(weights)

        rows = []
        for _ in range(n):
            g = np.random.choice(genres_list, p=weights)
            p = genre_profiles[g]
            rows.append({
                "genre": g,
                "danceability":     np.clip(p["d"] + np.random.normal(0, 0.12), 0, 1),
                "energy":           np.clip(p["e"] + np.random.normal(0, 0.14), 0, 1),
                "valence":          np.clip(p["v"] + np.random.normal(0, 0.15), 0, 1),
                "acousticness":     np.clip(p["a"] + np.random.normal(0, 0.15), 0, 1),
                "loudness":         np.clip(p["l"] + np.random.normal(0, 2.5), -35, 0),
                "speechiness":      np.clip(p["s"] + np.random.normal(0, 0.05), 0, 1),
                "instrumentalness": np.clip(p["i"] + np.random.normal(0, 0.08), 0, 1),
                "liveness":         np.clip(p["lv"]+ np.random.normal(0, 0.06), 0, 1),
                "popularity":       int(np.clip(40 + np.random.normal(0, 20), 0, 100)),
            })
        df = pd.DataFrame(rows)

    # Normalize loudness to [0,1] for radar
    df["loudness_norm"] = (df["loudness"] - df["loudness"].min()) / \
                          (df["loudness"].max() - df["loudness"].min())
    # Add region
    df["region"] = df["genre"].apply(get_region)
    return df


# ── 1. Genre Radar Chart ──────────────────────────────────────────────────────
def plot_genre_radar(df, top_n=12):
    """
    各風格的音頻特徵雷達圖
    每個 genre 畫一個八邊形，形狀差異 = 風格「聲學指紋」
    """
    top_genres = df["genre"].value_counts().head(top_n).index.tolist()
    genre_means = df[df["genre"].isin(top_genres)].groupby("genre")[FEATURES].mean()

    N = len(FEATURES)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    labels_display = ["Danceability", "Energy", "Valence", "Acousticness",
                      "Loudness\n(norm)", "Speechiness", "Instrumental", "Liveness"]

    cols = 4
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4.5),
                             subplot_kw=dict(polar=True))
    fig.suptitle("Audio Feature Radar by Genre\n(Each axis = normalized mean score)",
                 fontsize=15, y=1.01)

    colors = plt.cm.tab20(np.linspace(0, 1, top_n))

    for idx, (genre, ax) in enumerate(zip(top_genres, axes.flat)):
        values = genre_means.loc[genre].tolist()
        values += values[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_display, size=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.5", "0.75"], size=6, color="gray")
        ax.grid(color="gray", alpha=0.3)

        c = colors[idx]
        ax.fill(angles, values, alpha=0.25, color=c)
        ax.plot(angles, values, color=c, linewidth=2)
        ax.set_title(genre.upper(), size=10, pad=12, color=c,
                     fontweight="bold")

    # Hide empty axes
    for ax in axes.flat[top_n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/figures/10_genre_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/10_genre_radar.png")


# ── 2. Region Radar Chart ─────────────────────────────────────────────────────
def plot_region_radar(df):
    """
    地區對應雷達圖
    用 genre 名稱推算地區 (有根據的 proxy，非隨機分配)
    """
    regions = df["region"].value_counts()
    regions = regions[regions >= 100].index.tolist()

    region_means = df.groupby("region")[FEATURES].mean()

    N = len(FEATURES)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    labels_display = ["Danceability", "Energy", "Valence", "Acousticness",
                      "Loudness\n(norm)", "Speechiness", "Instrumental", "Liveness"]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle("Audio Feature Radar by Region\n(Region derived from genre name — culturally grounded proxy)",
                 fontsize=12)

    region_colors = {
        "Western":      "#4c8bbe",
        "Asia":         "#e07b39",
        "Latin America":"#5ba55b",
        "Africa":       "#c25b5b",
        "Middle East":  "#9b6bbf",
    }

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_display, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.5", "0.75"], size=7, color="gray")
    ax.grid(color="gray", alpha=0.3)

    patches = []
    for region in regions:
        if region not in region_means.index:
            continue
        values = region_means.loc[region].tolist()
        values += values[:1]
        c = region_colors.get(region, "#888")
        ax.fill(angles, values, alpha=0.15, color=c)
        ax.plot(angles, values, color=c, linewidth=2.5, label=region)
        patches.append(mpatches.Patch(color=c, label=region))

    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.35, 1.15), fontsize=10)

    plt.tight_layout()
    plt.savefig("outputs/figures/11_region_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/11_region_radar.png")


# ── 3. Genre × Feature Heatmap ───────────────────────────────────────────────
def plot_genre_heatmap(df, top_n=16):
    """
    風格 × 特徵均值熱圖
    顏色深 = 該風格在該特徵上分數高
    """
    top_genres = df["genre"].value_counts().head(top_n).index.tolist()
    heatmap_features = ["danceability", "energy", "valence", "acousticness",
                        "speechiness", "instrumentalness", "liveness", "loudness_norm"]
    feat_labels = ["Danceability", "Energy", "Valence", "Acousticness",
                   "Speechiness", "Instrumental", "Liveness", "Loudness\n(norm)"]

    means = df[df["genre"].isin(top_genres)].groupby("genre")[heatmap_features].mean()
    means = means.loc[top_genres]

    # Z-score normalize per column for better contrast
    means_norm = (means - means.mean()) / means.std()

    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(means_norm.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)

    ax.set_xticks(range(len(feat_labels)))
    ax.set_xticklabels(feat_labels, fontsize=10)
    ax.set_yticks(range(len(top_genres)))
    ax.set_yticklabels([g.upper() for g in top_genres], fontsize=9)

    # Annotate raw mean values
    for i in range(len(top_genres)):
        for j in range(len(heatmap_features)):
            val = means.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black")

    plt.colorbar(im, ax=ax, label="Z-score (column normalized)")
    ax.set_title("Genre × Audio Feature Heatmap\n(Cell = raw mean, Color = z-score within feature)",
                 fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig("outputs/figures/12_genre_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/12_genre_heatmap.png")


# ── 4. Genre Dendrogram ───────────────────────────────────────────────────────
def plot_genre_dendrogram(df, top_n=16):
    """
    風格聚類樹狀圖
    聲學上相似的 genre 會聚在一起
    用 Ward linkage (最小化組內變異)
    """
    top_genres = df["genre"].value_counts().head(top_n).index.tolist()
    feat_cols = ["danceability", "energy", "valence", "acousticness",
                 "speechiness", "instrumentalness", "liveness", "loudness_norm"]
    means = df[df["genre"].isin(top_genres)].groupby("genre")[feat_cols].mean()
    means = means.loc[top_genres]

    Z = linkage(means.values, method="ward")

    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(Z, labels=[g.upper() for g in top_genres],
               ax=ax, color_threshold=0.7 * max(Z[:, 2]),
               leaf_rotation=30, leaf_font_size=10)

    ax.set_title("Genre Similarity Dendrogram (Ward Linkage)\n"
                 "(Genres that cluster together = similar audio fingerprint)",
                 fontsize=12)
    ax.set_ylabel("Distance (dissimilarity)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/13_genre_cluster.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/13_genre_cluster.png")


# ── 5. Region Boxplot + Kruskal-Wallis ────────────────────────────────────────
def plot_region_boxplot(df):
    """
    地區 × 特徵箱形圖 (genre-based region proxy)
    標注 Kruskal-Wallis H 統計量與顯著性
    """
    regions = df["region"].value_counts()
    regions = regions[regions >= 100].index.tolist()

    plot_feats = ["danceability", "energy", "valence",
                  "acousticness", "speechiness", "instrumentalness"]
    colors_map = {
        "Western":       "#4c8bbe",
        "Asia":          "#e07b39",
        "Latin America": "#5ba55b",
        "Africa":        "#c25b5b",
        "Middle East":   "#9b6bbf",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Audio Features by Region (Genre-based Proxy)\n"
                 "Region assigned from genre cultural origin — not random",
                 fontsize=13)

    for ax, feat in zip(axes.flat, plot_feats):
        data = [df[df["region"] == r][feat].dropna().values for r in regions]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=regions)
        for patch, region in zip(bp["boxes"], regions):
            patch.set_facecolor(colors_map.get(region, "#888"))
            patch.set_alpha(0.75)

        # Kruskal-Wallis
        stat, pval = kruskal(*data)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        ax.set_title(feat, fontsize=11)
        ax.text(0.97, 0.96, f"K-W H={stat:.1f} {sig}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/figures/14_region_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/14_region_boxplot.png")


# ── 6. Print region summary ───────────────────────────────────────────────────
def print_region_summary(df):
    print("\n=== Region Distribution (genre-based proxy) ===")
    rc = df["region"].value_counts()
    for region, cnt in rc.items():
        pct = cnt / len(df) * 100
        genres_in = df[df["region"] == region]["genre"].unique()[:5]
        print(f"  {region:<15} n={cnt:>6,} ({pct:>5.1f}%)  e.g. {list(genres_in)}")

    print("\n=== Region Mean Features ===")
    feat_cols = ["danceability", "energy", "valence", "acousticness", "speechiness"]
    print(df.groupby("region")[feat_cols].mean().round(3).to_string())


if __name__ == "__main__":
    df = load_or_simulate()

    print("\n[1/5] Genre radar charts ...")
    plot_genre_radar(df, top_n=12)

    print("[2/5] Region radar chart ...")
    plot_region_radar(df)

    print("[3/5] Genre × feature heatmap ...")
    plot_genre_heatmap(df, top_n=16)

    print("[4/5] Genre similarity dendrogram ...")
    plot_genre_dendrogram(df, top_n=16)

    print("[5/5] Region boxplot + Kruskal-Wallis ...")
    plot_region_boxplot(df)

    print_region_summary(df)

    print("\n✓ All charts saved to outputs/figures/")
    print("  10_genre_radar.png     — 各風格音頻特徵雷達圖")
    print("  11_region_radar.png    — 地區對比雷達圖")
    print("  12_genre_heatmap.png   — 風格 × 特徵熱圖")
    print("  13_genre_cluster.png   — 風格聲學相似度聚類")
    print("  14_region_boxplot.png  — 地區 × 特徵箱形圖 + K-W 檢定")
