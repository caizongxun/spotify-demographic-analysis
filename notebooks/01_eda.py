"""
01_eda.py
Spotify Demographic Analysis - Exploratory Data Analysis + Charts
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

os.makedirs("outputs/figures", exist_ok=True)

FEATURES = ["danceability", "energy", "valence", "acousticness", "loudness", "speechiness"]


def load_or_simulate(path="data/dataset.csv"):
    """載入真實資料或產生模擬資料"""
    if os.path.exists(path):
        df = pd.read_csv(path)
        # 統一欄位名稱（Kaggle 版本可能有 track_genre 而非 genre）
        if "track_genre" in df.columns:
            df = df.rename(columns={"track_genre": "genre"})
        print(f"已載入真實資料：{len(df):,} 首歌曲")
    else:
        print("找不到 data/dataset.csv，使用模擬資料（貼近真實 Spotify 分佈）")
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
        energy = np.clip(0.35 + 0.00025 * (years - 1920) + np.random.normal(0, 0.18, n), 0, 1)
        valence = np.clip(0.48 + np.random.normal(0, 0.2, n), 0, 1)
        acousticness = np.clip(0.55 - 0.0025 * (years - 1920) + np.random.normal(0, 0.22, n), 0, 1)
        loudness = np.clip(-14 + 0.018 * (years - 1920) + np.random.normal(0, 2.8, n), -60, 0)
        speechiness = np.clip(0.04 + 0.00008 * (years - 1920) + np.random.normal(0, 0.035, n), 0, 1)
        popularity = np.clip(20 + 25 * danceability + 15 * energy + np.random.normal(0, 12, n), 0, 100)
        df = pd.DataFrame({
            "year": years, "genre": genres,
            "danceability": danceability, "energy": energy,
            "valence": valence, "acousticness": acousticness,
            "loudness": loudness, "speechiness": speechiness,
            "popularity": popularity
        })
    return df


def add_proxy_variables(df):
    """加入代理變數：age_group, gender_proxy, region"""
    if "year" in df.columns:
        df["age_group"] = pd.cut(
            df["year"],
            bins=[1900, 1960, 1980, 2000, 2010, 2027],
            labels=["Pre-1960", "1960s-70s", "1980s-90s", "2000s", "2010s+"]
        )
    # 性別代理（基於音樂特徵的機率模型）
    prob_female = np.clip(
        0.5 + 0.2 * df["valence"] - 0.15 * df["energy"] + 0.1 * df["danceability"],
        0.1, 0.9
    )
    df["gender_proxy"] = np.where(np.random.random(len(df)) < prob_female, "Female", "Male")
    # 地區代理
    df["region"] = np.random.choice(
        ["Western", "Asia", "Latin America"], len(df), p=[0.55, 0.3, 0.15]
    )
    return df


if __name__ == "__main__":
    df = load_or_simulate()
    df = add_proxy_variables(df)

    print("\n=== 描述統計 ===")
    print(df[FEATURES].describe().round(3))

    # ── 圖1：特徵分佈（小提琴圖）──
    df_melt = df[FEATURES].melt(var_name="Feature", value_name="Score")
    fig1 = px.violin(
        df_melt, x="Feature", y="Score", color="Feature",
        box=True, points=False,
        title=f"Spotify Audio Feature Distribution (N={len(df):,})",
        labels={"Score": "Score", "Feature": "Audio Feature"}
    )
    fig1.update_layout(showlegend=False)
    fig1.write_image("outputs/figures/01_feature_distribution.png")
    print("Chart saved: outputs/figures/01_feature_distribution.png")

    # ── 圖2：年齡群 × 特徵 箱形圖 ──
    if "age_group" in df.columns:
        age_melt = df[["age_group", "danceability", "acousticness", "energy", "loudness"]].melt(
            id_vars="age_group", var_name="Feature", value_name="Score"
        )
        fig2 = px.box(
            age_melt, x="age_group", y="Score", color="Feature",
            title="Audio Features by Age Group (Year Proxy)",
            labels={"age_group": "Age Group", "Score": "Score"}
        )
        fig2.write_image("outputs/figures/02_age_features.png")
        print("Chart saved: outputs/figures/02_age_features.png")

    # ── 圖3：性別 × 特徵 箱形圖 ──
    gender_melt = df[["gender_proxy", "danceability", "valence", "energy"]].melt(
        id_vars="gender_proxy", var_name="Feature", value_name="Score"
    )
    fig3 = px.box(
        gender_melt, x="gender_proxy", y="Score", color="Feature",
        title="Audio Features by Predicted Gender",
        labels={"gender_proxy": "Predicted Gender", "Score": "Score"}
    )
    fig3.write_image("outputs/figures/03_gender_features.png")
    print("Chart saved: outputs/figures/03_gender_features.png")

    # ── 圖4：地區 × 特徵 條形圖 ──
    region_means = df.groupby("region")[FEATURES].mean().reset_index()
    region_melt = region_means.melt(id_vars="region", var_name="Feature", value_name="Mean")
    fig4 = px.bar(
        region_melt, x="region", y="Mean", color="Feature", barmode="group",
        title="Mean Audio Features by Region",
        labels={"region": "Region", "Mean": "Mean Score"}
    )
    fig4.write_image("outputs/figures/04_region_features.png")
    print("Chart saved: outputs/figures/04_region_features.png")

    print("\nEDA complete.")
