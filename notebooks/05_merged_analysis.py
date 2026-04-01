"""
05_merged_analysis.py
======================
結合兩個資料集做人口學分析

Dataset A: Spotify Tracks Dataset (音頻特徵)
  https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
  本地路徑: data/tracks.csv

Dataset B: Spotify User Behavior Dataset (年齡/性別/地區)
  https://www.kaggle.com/datasets/meeraajayakumar/spotify-user-behavior-dataset
  本地路徑: data/users.csv

JOIN KEY: genre (favorite_genre <-> track_genre)

圖表清單:
  A1_feature_distribution.png
  A2_correlation_heatmap.png
  B1_age_violin.png
  B2_gender_boxplot.png
  B3_region_radar.png
  C1_age_genre_heatmap.png
  C2_gender_genre_bar.png
  C3_age_scatter.png
  C4_summary_dashboard.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kruskal
from math import pi

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

AUDIO_FEATURES = ["danceability", "energy", "valence",
                   "acousticness", "speechiness", "instrumentalness"]

FEATURE_LABELS = {
    "danceability":     "Danceability",
    "energy":           "Energy",
    "valence":          "Valence (mood)",
    "acousticness":     "Acousticness",
    "speechiness":      "Speechiness",
    "instrumentalness": "Instrumental",
}

PALETTE = {
    "Male":          "#4c8bbe",
    "Female":        "#e07b8a",
    "Non-binary":    "#7bb87b",
    "Western":       "#4c8bbe",
    "Asia":          "#e07b39",
    "Latin America": "#5ba55b",
    "Africa":        "#c25b5b",
    "Middle East":   "#9b6bbf",
}
AGE_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1"]


# ── 0. Kaggle API download ───────────────────────────────────────────────
def setup_kaggle_credentials():
    """
    應對三種方式：
    1. 環境變數 KAGGLE_API_TOKEN (JSON 字串或純 token)
    2. 環境變數 KAGGLE_USERNAME + KAGGLE_KEY 分開提供
    3. ~/.kaggle/kaggle.json 已存在
    """
    kaggle_dir  = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    # 方式 1：KAGGLE_API_TOKEN 環境變數
    token_env = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token_env:
        os.makedirs(kaggle_dir, exist_ok=True)
        # 如果是 JSON 字串，直接寫入
        if token_env.startswith("{"):
            with open(kaggle_json, "w") as f:
                f.write(token_env)
        else:
            # 如果是純 token，需要搭配 username
            username = os.environ.get("KAGGLE_USERNAME", "")
            if not username:
                # 嘗試從 token 前綴 KGAT_ 後面的內容推導 (Kaggle token 格式)
                # 如果沒有 username，就用 KAGGLE_KEY + KAGGLE_USERNAME
                username = os.environ.get("KAGGLE_USERNAME", "kaggle_user")
            with open(kaggle_json, "w") as f:
                json.dump({"username": username, "key": token_env}, f)
        os.chmod(kaggle_json, 0o600)
        print("[kaggle] Credentials written from KAGGLE_API_TOKEN env var")
        return True

    # 方式 2：KAGGLE_USERNAME + KAGGLE_KEY 分開提供
    username = os.environ.get("KAGGLE_USERNAME", "")
    key      = os.environ.get("KAGGLE_KEY", "")
    if username and key:
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json, 0o600)
        print("[kaggle] Credentials written from KAGGLE_USERNAME + KAGGLE_KEY env vars")
        return True

    # 方式 3：檔案已存在
    if os.path.exists(kaggle_json):
        print("[kaggle] Using existing ~/.kaggle/kaggle.json")
        return True

    print("[!] No Kaggle credentials found. Options:")
    print("    A) export KAGGLE_API_TOKEN='your_token'")
    print("    B) export KAGGLE_USERNAME='user' KAGGLE_KEY='key'")
    print("    C) Place kaggle.json at ~/.kaggle/kaggle.json")
    return False


def download_kaggle_datasets():
    """Download both datasets via Kaggle API"""
    if not setup_kaggle_credentials():
        return False
    try:
        import kaggle  # noqa
    except ImportError:
        print("[!] Run: pip install kaggle")
        return False

    datasets = [
        ("maharshipandya/-spotify-tracks-dataset", "data/tracks_raw", "data/tracks.csv"),
        ("meeraajayakumar/spotify-user-behavior-dataset", "data/users_raw", "data/users.csv"),
    ]
    for dataset, dl_path, dest in datasets:
        if os.path.exists(dest):
            print(f"[kaggle] {dest} already exists, skipping")
            continue
        print(f"[kaggle] Downloading {dataset} ...")
        ret = os.system(f"kaggle datasets download -d {dataset} -p {dl_path} --unzip -q")
        if ret != 0:
            print(f"[!] Download failed for {dataset}")
            continue
        # Move first CSV found to dest
        for root, _, files in os.walk(dl_path):
            for f in sorted(files):
                if f.endswith(".csv"):
                    os.rename(os.path.join(root, f), dest)
                    print(f"[kaggle] Saved -> {dest}")
                    break
    return True


# ── 1. Load or simulate ──────────────────────────────────────────────────
def load_tracks(path="data/tracks.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "track_genre" in df.columns:
            df = df.rename(columns={"track_genre": "genre"})
        print(f"[tracks] Real data: {len(df):,} rows, {df['genre'].nunique()} genres")
        return df
    print("[tracks] Simulating 20k tracks ...")
    np.random.seed(42)
    genre_profiles = {
        "pop":       dict(d=0.65,e=0.65,v=0.60,a=0.20,s=0.05,i=0.05),
        "k-pop":     dict(d=0.72,e=0.78,v=0.68,a=0.08,s=0.05,i=0.03),
        "j-pop":     dict(d=0.58,e=0.62,v=0.62,a=0.25,s=0.03,i=0.08),
        "hip-hop":   dict(d=0.75,e=0.60,v=0.45,a=0.12,s=0.22,i=0.04),
        "rap":       dict(d=0.73,e=0.62,v=0.42,a=0.10,s=0.28,i=0.03),
        "latin":     dict(d=0.78,e=0.76,v=0.74,a=0.15,s=0.06,i=0.04),
        "reggaeton": dict(d=0.82,e=0.80,v=0.72,a=0.06,s=0.08,i=0.03),
        "classical": dict(d=0.28,e=0.22,v=0.38,a=0.88,s=0.02,i=0.82),
        "edm":       dict(d=0.65,e=0.90,v=0.50,a=0.05,s=0.03,i=0.12),
        "rock":      dict(d=0.50,e=0.82,v=0.48,a=0.15,s=0.05,i=0.06),
        "country":   dict(d=0.55,e=0.55,v=0.62,a=0.45,s=0.04,i=0.05),
        "r-n-b":     dict(d=0.70,e=0.58,v=0.55,a=0.20,s=0.08,i=0.04),
        "jazz":      dict(d=0.45,e=0.38,v=0.52,a=0.72,s=0.04,i=0.35),
        "afrobeat":  dict(d=0.80,e=0.78,v=0.72,a=0.18,s=0.07,i=0.05),
    }
    rows = []
    genres = list(genre_profiles.keys())
    for _ in range(20000):
        g = np.random.choice(genres)
        p = genre_profiles[g]
        rows.append({
            "genre": g,
            "danceability":     np.clip(p["d"]+np.random.normal(0,0.12),0,1),
            "energy":           np.clip(p["e"]+np.random.normal(0,0.14),0,1),
            "valence":          np.clip(p["v"]+np.random.normal(0,0.15),0,1),
            "acousticness":     np.clip(p["a"]+np.random.normal(0,0.15),0,1),
            "speechiness":      np.clip(p["s"]+np.random.normal(0,0.05),0,1),
            "instrumentalness": np.clip(p["i"]+np.random.normal(0,0.08),0,1),
            "loudness":         np.clip(-8+np.random.normal(0,3),-35,0),
            "popularity":       int(np.clip(40+np.random.normal(0,20),0,100)),
        })
    return pd.DataFrame(rows)


def load_users(path="data/users.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        for col in ["favorite_genre", "preferred_genre", "music_genre", "genre",
                    "fav_music_genre", "music_preferences"]:
            if col in df.columns:
                df = df.rename(columns={col: "fav_genre"})
                break
        # Normalize age col
        for col in ["age", "user_age"]:
            if col in df.columns and col != "age":
                df = df.rename(columns={col: "age"})
                break
        # Normalize gender col
        for col in ["gender", "user_gender", "sex"]:
            if col in df.columns and col != "gender":
                df = df.rename(columns={col: "gender"})
                break
        print(f"[users]  Real data: {len(df):,} rows | cols: {list(df.columns[:8])}")
        return df

    print("[users]  Simulating 5k users ...")
    np.random.seed(99)
    genres = ["pop","k-pop","j-pop","hip-hop","rap","latin","reggaeton",
              "classical","edm","rock","country","r-n-b","jazz","afrobeat"]
    age_genre = {
        "k-pop":(21,4),"j-pop":(23,5),"classical":(42,12),"jazz":(38,10),
        "country":(35,10),"hip-hop":(24,6),"rap":(23,5),"edm":(25,7),
        "latin":(28,8),"reggaeton":(26,7),"rock":(32,9),"pop":(26,8),
        "r-n-b":(27,7),"afrobeat":(28,8),
    }
    region_genre = {
        "k-pop":"Asia","j-pop":"Asia","latin":"Latin America",
        "reggaeton":"Latin America","afrobeat":"Africa",
    }
    rows = []
    for _ in range(5000):
        g = np.random.choice(genres)
        mu, sd = age_genre.get(g, (28, 8))
        age = int(np.clip(np.random.normal(mu, sd), 13, 65))
        gender = np.random.choice(["Male","Female","Non-binary"], p=[0.47,0.48,0.05])
        region = region_genre.get(g, "Western")
        rows.append({"fav_genre": g, "age": age, "gender": gender, "region": region})
    return pd.DataFrame(rows)


def merge_datasets(tracks, users):
    track_means = tracks.groupby("genre")[AUDIO_FEATURES].mean().reset_index()
    merged = users.merge(track_means, left_on="fav_genre", right_on="genre", how="left")
    merged["age_group"] = pd.cut(
        pd.to_numeric(merged["age"], errors="coerce"),
        bins=[12, 17, 24, 34, 44, 65],
        labels=["13-17", "18-24", "25-34", "35-44", "45+"]
    )
    print(f"[merge]  {len(merged):,} rows, age_groups: {merged['age_group'].value_counts().to_dict()}")
    return merged


# ── Charts ──────────────────────────────────────────────────────────────

def plot_A1_feature_distribution(tracks):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Audio Feature Distributions (Histogram + KDE)", fontsize=13)
    for ax, feat in zip(axes.flat, AUDIO_FEATURES):
        data = tracks[feat].dropna()
        ax.hist(data, bins=50, color="#4c8bbe", alpha=0.5, density=True)
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 300)
        ax.plot(xs, kde(xs), color="#1a5276", lw=2)
        ax.axvline(data.mean(), color="red", lw=1.5, linestyle="--",
                   label=f"μ={data.mean():.2f}")
        ax.set_title(FEATURE_LABELS[feat], fontsize=11)
        ax.set_xlabel("Value (0-1)"); ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/A1_feature_distribution.png", dpi=140, bbox_inches="tight")
    plt.close(); print("  A1 done")


def plot_A2_correlation_heatmap(tracks):
    corr = tracks[AUDIO_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    labels = [FEATURE_LABELS[f] for f in AUDIO_FEATURES]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(AUDIO_FEATURES)):
        for j in range(len(AUDIO_FEATURES)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Audio Feature Correlation Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/figures/A2_correlation_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(); print("  A2 done")


def plot_B1_age_violin(merged):
    age_groups = ["13-17", "18-24", "25-34", "35-44", "45+"]
    feats = ["danceability", "energy", "valence", "acousticness"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle("Audio Features by Age Group (violin = distribution shape)", fontsize=13)
    for ax, feat in zip(axes, feats):
        data = [merged[merged["age_group"]==ag][feat].dropna().values for ag in age_groups]
        data = [d if len(d) > 1 else np.array([0.5, 0.5]) for d in data]
        parts = ax.violinplot(data, positions=range(len(age_groups)),
                              showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(AGE_COLORS[i % len(AGE_COLORS)]); pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        ax.set_xticks(range(len(age_groups))); ax.set_xticklabels(age_groups, fontsize=9)
        ax.set_title(FEATURE_LABELS[feat], fontsize=11)
        ax.set_ylabel("Score (0-1)"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/B1_age_violin.png", dpi=140, bbox_inches="tight")
    plt.close(); print("  B1 done")


def plot_B2_gender_boxplot(merged):
    genders = [g for g in ["Male","Female","Non-binary"] if g in merged["gender"].values]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Audio Features by Gender", fontsize=13)
    for ax, feat in zip(axes.flat, AUDIO_FEATURES):
        data = [merged[merged["gender"]==g][feat].dropna().values for g in genders]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=genders)
        for patch, gender in zip(bp["boxes"], genders):
            patch.set_facecolor(PALETTE.get(gender, "#aaa")); patch.set_alpha(0.75)
        valid = [d for d in data if len(d) > 1]
        if len(valid) >= 2:
            stat, pval = kruskal(*valid)
            sig = "***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else "ns"))
            ax.text(0.97, 0.96, f"K-W {sig}", transform=ax.transAxes,
                    fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_title(FEATURE_LABELS[feat], fontsize=11)
        ax.set_ylabel("Score (0-1)"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/B2_gender_boxplot.png", dpi=140, bbox_inches="tight")
    plt.close(); print("  B2 done")


def plot_B3_region_radar(merged):
    regions = [r for r in ["Western","Asia","Latin America","Africa","Middle East"]
               if r in merged.get("region", pd.Series()).values]
    if not regions:
        print("  B3 skipped (no region column)"); return
    feats = AUDIO_FEATURES
    N = len(feats)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    labels = [FEATURE_LABELS[f] for f in feats]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25","0.5","0.75"], size=7, color="gray")
    ax.grid(color="gray", alpha=0.3)
    for region in regions:
        sub = merged[merged["region"]==region]
        vals = sub[feats].mean().tolist() + [sub[feats[0]].mean()]
        c = PALETTE.get(region, "#888")
        ax.fill(angles, vals, alpha=0.12, color=c)
        ax.plot(angles, vals, color=c, lw=2.5, label=region)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title("Audio Features by Region", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig("outputs/figures/B3_region_radar.png", dpi=140, bbox_inches="tight")
    plt.close(); print("  B3 done")


def plot_C1_age_genre_heatmap(merged):
    age_groups = ["13-17", "18-24", "25-34", "35-44", "45+"]
    top_genres = merged["fav_genre"].value_counts().head(12).index.tolist()
    pivot = pd.crosstab(merged["age_group"], merged["fav_genre"], normalize="index") * 100
    pivot = pivot.reindex(index=age_groups, columns=top_genres, fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(top_genres)))
    ax.set_xticklabels([g.upper() for g in top_genres], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(age_groups))); ax.set_yticklabels(age_groups, fontsize=10)
    for i in range(len(age_groups)):
        for j in range(len(top_genres)):
            ax.text(j, i, f"{pivot.iloc[i,j]:.1f}%", ha="center", va="center", fontsize=7.5)
    plt.colorbar(im, ax=ax, label="% within age group")
    ax.set_title("Genre Preference by Age Group", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig("outputs/figures/C1_age_genre_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(); print("  C1 done")


def plot_C2_gender_genre_bar(merged):
    genders = [g for g in ["Male","Female","Non-binary"] if g in merged["gender"].values]
    top_genres = merged["fav_genre"].value_counts().head(10).index.tolist()
    pivot = pd.crosstab(merged["gender"], merged["fav_genre"], normalize="index") * 100
    pivot = pivot.reindex(index=genders, columns=top_genres, fill_value=0)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_genres)))
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(genders))
    for genre, c in zip(top_genres, colors):
        vals = pivot[genre].values
        bars = ax.barh(genders, vals, left=bottom, color=c, label=genre.upper(), height=0.55)
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                        f"{val:.0f}%", ha="center", va="center", fontsize=8)
        bottom += vals
    ax.set_xlabel("% within gender")
    ax.set_title("Genre Preference by Gender", fontsize=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/C2_gender_genre_bar.png", dpi=140, bbox_inches="tight")
    plt.close(); print("  C2 done")


def plot_C3_age_scatter(merged):
    genders = [g for g in ["Male","Female","Non-binary"] if g in merged["gender"].values]
    feats_plot = [("danceability","energy"),("valence","acousticness"),("speechiness","instrumentalness")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Feature Scatter by Age (color=gender, size=age)", fontsize=12)
    for ax, (fx, fy) in zip(axes, feats_plot):
        for gender in genders:
            sub = merged[merged["gender"]==gender].dropna(subset=[fx, fy, "age"])
            if len(sub) == 0: continue
            sizes = (pd.to_numeric(sub["age"], errors="coerce") - 13) / 52 * 80 + 20
            ax.scatter(sub[fx], sub[fy], s=sizes, alpha=0.35,
                       color=PALETTE.get(gender,"#aaa"), label=gender, edgecolors="none")
        ax.set_xlabel(FEATURE_LABELS[fx]); ax.set_ylabel(FEATURE_LABELS[fy])
        ax.set_title(f"{FEATURE_LABELS[fx]} vs {FEATURE_LABELS[fy]}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("outputs/figures/C3_age_scatter.png", dpi=140, bbox_inches="tight")
    plt.close(); print("  C3 done")


def plot_C4_summary_dashboard(merged):
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Key Findings: Demographics x Music Features", fontsize=14, y=1.02)

    ax1 = fig.add_subplot(131)
    age_groups = ["13-17","18-24","25-34","35-44","45+"]
    age_dance = merged.groupby("age_group")["danceability"].mean()
    vals = [float(age_dance.get(ag, np.nan)) for ag in age_groups]
    ax1.plot(age_groups, vals, "o-", color="#4c8bbe", lw=2.5, markersize=8)
    ax1.fill_between(range(len(age_groups)), vals, alpha=0.15, color="#4c8bbe")
    ax1.set_xticks(range(len(age_groups))); ax1.set_xticklabels(age_groups, fontsize=9)
    ax1.set_ylim(0, 1); ax1.set_title("Danceability vs Age", fontsize=11)
    ax1.set_ylabel("Mean danceability"); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(132)
    genders = [g for g in ["Male","Female","Non-binary"] if g in merged["gender"].values]
    vals2 = [merged[merged["gender"]==g]["valence"].mean() for g in genders]
    bars = ax2.bar(genders, vals2, color=[PALETTE.get(g,"#aaa") for g in genders],
                   alpha=0.8, width=0.5)
    for bar, v in zip(bars, vals2):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
                 ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1); ax2.set_title("Valence by Gender", fontsize=11)
    ax2.set_ylabel("Mean valence"); ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(133)
    regions = [r for r in ["Western","Asia","Latin America","Africa","Middle East"]
               if r in merged.get("region", pd.Series()).values]
    if regions:
        vals3 = [merged[merged["region"]==r]["danceability"].mean() for r in regions]
        bars3 = ax3.barh(regions, vals3,
                         color=[PALETTE.get(r,"#aaa") for r in regions],
                         alpha=0.8, height=0.5)
        for bar, v in zip(bars3, vals3):
            ax3.text(v+0.005, bar.get_y()+bar.get_height()/2, f"{v:.2f}",
                     va="center", fontsize=10, fontweight="bold")
        ax3.set_xlim(0, 1); ax3.set_xlabel("Mean danceability")
    ax3.set_title("Danceability by Region", fontsize=11)
    ax3.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/figures/C4_summary_dashboard.png", dpi=140, bbox_inches="tight")
    plt.close(); print("  C4 done")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not (os.path.exists("data/tracks.csv") and os.path.exists("data/users.csv")):
        print("=== Kaggle API download ===")
        download_kaggle_datasets()

    tracks = load_tracks()
    users  = load_users()
    merged = merge_datasets(tracks, users)

    print("\n=== Generating 9 charts ===")
    plot_A1_feature_distribution(tracks)
    plot_A2_correlation_heatmap(tracks)
    plot_B1_age_violin(merged)
    plot_B2_gender_boxplot(merged)
    plot_B3_region_radar(merged)
    plot_C1_age_genre_heatmap(merged)
    plot_C2_gender_genre_bar(merged)
    plot_C3_age_scatter(merged)
    plot_C4_summary_dashboard(merged)
    print("\n✓ All charts -> outputs/figures/")
