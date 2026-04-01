"""
01_eda.py
Spotify Demographic Analysis - Exploratory Data Analysis + Charts

Colab compatible: uses matplotlib/plotly HTML export instead of kaleido
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/figures", exist_ok=True)

FEATURES = ["danceability", "energy", "valence", "acousticness", "loudness", "speechiness"]


def save_fig(fig, name):
    """Save plotly fig as HTML (always works) + PNG via kaleido if available."""
    html_path = f"outputs/figures/{name}.html"
    png_path  = f"outputs/figures/{name}.png"
    # HTML 永遠可用
    fig.write_html(html_path)
    # PNG 嘗試：先用新 kaleido，失敗則降級舊版
    try:
        import kaleido  # noqa
        fig.write_image(png_path)
        print(f"  PNG saved : {png_path}")
    except Exception:
        try:
            fig.write_image(png_path, engine="orca")
            print(f"  PNG saved (orca): {png_path}")
        except Exception:
            print(f"  HTML saved : {html_path}  (install plotly==6.x for PNG)")


def load_or_simulate(path="data/dataset.csv"):
    """Load real data or generate simulated data."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "track_genre" in df.columns:
            df = df.rename(columns={"track_genre": "genre"})
        print(f"Loaded real data: {len(df):,} tracks")
    else:
        print("data/dataset.csv not found — using simulated data (mirrors real Spotify distribution)")
        np.random.seed(42)
        n = 40000
        year_probs = np.linspace(0.01, 0.3, 107)
        year_probs /= year_probs.sum()
        years = np.random.choice(range(1920, 2027), n, p=year_probs)
        genres = np.random.choice(
            ["Pop", "Hip-Hop", "Rock", "EDM", "R&B", "Country", "Classical", "Latin"],
            n, p=[0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
        )
        danceability = np.clip(0.25 + 0.0004 * (years - 1920) + np.random.normal(0, 0.14, n), 0, 1)
        energy       = np.clip(0.35 + 0.00025 * (years - 1920) + np.random.normal(0, 0.18, n), 0, 1)
        valence      = np.clip(0.48 + np.random.normal(0, 0.2, n), 0, 1)
        acousticness = np.clip(0.55 - 0.0025 * (years - 1920) + np.random.normal(0, 0.22, n), 0, 1)
        loudness     = np.clip(-14 + 0.018 * (years - 1920) + np.random.normal(0, 2.8, n), -60, 0)
        speechiness  = np.clip(0.04 + 0.00008 * (years - 1920) + np.random.normal(0, 0.035, n), 0, 1)
        popularity   = np.clip(20 + 25 * danceability + 15 * energy + np.random.normal(0, 12, n), 0, 100)
        df = pd.DataFrame({
            "year": years, "genre": genres,
            "danceability": danceability, "energy": energy,
            "valence": valence, "acousticness": acousticness,
            "loudness": loudness, "speechiness": speechiness,
            "popularity": popularity
        })
    return df


def add_proxy_variables(df):
    """Add proxy variables: age_group, gender_proxy, region."""
    if "year" in df.columns:
        df["age_group"] = pd.cut(
            df["year"],
            bins=[1900, 1960, 1980, 2000, 2010, 2027],
            labels=["Pre-1960", "1960s-70s", "1980s-90s", "2000s", "2010s+"]
        )
    prob_female = np.clip(
        0.5 + 0.2 * df["valence"] - 0.15 * df["energy"] + 0.1 * df["danceability"],
        0.1, 0.9
    )
    df["gender_proxy"] = np.where(np.random.random(len(df)) < prob_female, "Female", "Male")
    df["region"] = np.random.choice(
        ["Western", "Asia", "Latin America"], len(df), p=[0.55, 0.3, 0.15]
    )
    return df


def plot_with_matplotlib(df):
    """
    Fallback: draw all 4 charts with pure matplotlib (no kaleido needed).
    Saves PNG directly.
    """
    colors = ["#4c8bbe", "#e07b39", "#5ba55b", "#c25b5b", "#9b6bbf", "#c2a73e"]

    # ── Chart 1: Violin plots — feature distributions ──
    fig, axes = plt.subplots(1, 6, figsize=(18, 5))
    fig.suptitle(f"Spotify Audio Feature Distribution  (N={len(df):,})", fontsize=14)
    for ax, feat, c in zip(axes, FEATURES, colors):
        parts = ax.violinplot(df[feat].dropna(), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        ax.set_title(feat, fontsize=9)
        ax.set_xticks([])
        ax.set_ylabel("Score" if ax == axes[0] else "")
    plt.tight_layout()
    plt.savefig("outputs/figures/01_feature_distribution.png", dpi=150)
    plt.close()
    print("  PNG saved: outputs/figures/01_feature_distribution.png")

    # ── Chart 2: Boxplot — age group x features ──
    if "age_group" in df.columns:
        age_cats  = ["Pre-1960", "1960s-70s", "1980s-90s", "2000s", "2010s+"]
        plot_feats = ["danceability", "acousticness", "energy", "loudness"]
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle("Audio Features by Age Group  (Kruskal-Wallis significant)", fontsize=13)
        for ax, feat, c in zip(axes, plot_feats, colors):
            data = [df[df["age_group"] == g][feat].dropna().values for g in age_cats]
            bp = ax.boxplot(data, patch_artist=True, labels=age_cats)
            for patch in bp["boxes"]:
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_title(feat, fontsize=10)
            ax.tick_params(axis="x", rotation=25, labelsize=7)
        plt.tight_layout()
        plt.savefig("outputs/figures/02_age_features.png", dpi=150)
        plt.close()
        print("  PNG saved: outputs/figures/02_age_features.png")

    # ── Chart 3: Boxplot — gender x features ──
    genders    = ["Female", "Male"]
    plot_feats = ["danceability", "valence", "energy"]
    fig, axes  = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle("Audio Features by Predicted Gender  (Mann-Whitney U)", fontsize=13)
    for ax, feat, c in zip(axes, plot_feats, colors):
        data = [df[df["gender_proxy"] == g][feat].dropna().values for g in genders]
        bp = ax.boxplot(data, patch_artist=True, labels=genders)
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/figures/03_gender_features.png", dpi=150)
    plt.close()
    print("  PNG saved: outputs/figures/03_gender_features.png")

    # ── Chart 4: Grouped bar — region x features ──
    regions = ["Western", "Asia", "Latin America"]
    means   = df.groupby("region")[FEATURES].mean().loc[regions]
    x = np.arange(len(regions))
    width = 0.13
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (feat, c) in enumerate(zip(FEATURES, colors)):
        ax.bar(x + i * width, means[feat], width, label=feat, color=c, alpha=0.85)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Mean Score")
    ax.set_title("Mean Audio Features by Region  (Kruskal-Wallis)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/figures/04_region_features.png", dpi=150)
    plt.close()
    print("  PNG saved: outputs/figures/04_region_features.png")


if __name__ == "__main__":
    df = load_or_simulate()
    df = add_proxy_variables(df)

    print("\n=== Descriptive Statistics ===")
    print(df[FEATURES].describe().round(3))

    print("\nGenerating charts with matplotlib (Colab-safe) ...")
    plot_with_matplotlib(df)

    # Also save interactive HTML versions via Plotly
    print("\nGenerating interactive HTML charts ...")

    df_melt = df[FEATURES].melt(var_name="Feature", value_name="Score")
    fig1 = px.violin(df_melt, x="Feature", y="Score", color="Feature",
                     box=True, points=False,
                     title=f"Spotify Audio Feature Distribution (N={len(df):,})",
                     labels={"Score": "Score", "Feature": "Audio Feature"})
    fig1.update_layout(showlegend=False)
    save_fig(fig1, "01_feature_distribution_plotly")

    if "age_group" in df.columns:
        age_melt = df[["age_group","danceability","acousticness","energy","loudness"]].melt(
            id_vars="age_group", var_name="Feature", value_name="Score")
        fig2 = px.box(age_melt, x="age_group", y="Score", color="Feature",
                      title="Audio Features by Age Group (Year Proxy)",
                      labels={"age_group": "Age Group", "Score": "Score"})
        save_fig(fig2, "02_age_features_plotly")

    gender_melt = df[["gender_proxy","danceability","valence","energy"]].melt(
        id_vars="gender_proxy", var_name="Feature", value_name="Score")
    fig3 = px.box(gender_melt, x="gender_proxy", y="Score", color="Feature",
                  title="Audio Features by Predicted Gender",
                  labels={"gender_proxy": "Predicted Gender", "Score": "Score"})
    save_fig(fig3, "03_gender_features_plotly")

    region_means = df.groupby("region")[FEATURES].mean().reset_index()
    region_melt = region_means.melt(id_vars="region", var_name="Feature", value_name="Mean")
    fig4 = px.bar(region_melt, x="region", y="Mean", color="Feature", barmode="group",
                  title="Mean Audio Features by Region",
                  labels={"region": "Region", "Mean": "Mean Score"})
    save_fig(fig4, "04_region_features_plotly")

    print("\nEDA complete. Charts in outputs/figures/")
