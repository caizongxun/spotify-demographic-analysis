"""
02_statistical_tests.py
Spotify Demographic Analysis - Hypothesis Testing

Statistical methods:
- Shapiro-Wilk normality test
- Kruskal-Wallis H test (multi-group: age, region)
- Mann-Whitney U test (two-group: gender)
- Effect sizes: eta-squared (η²), rank-biserial r
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, kruskal, mannwhitneyu
import os, sys

# ─── 共用載入函數 ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FEATURES = ["danceability", "energy", "valence", "acousticness", "loudness", "speechiness"]


def load_or_simulate(path="data/dataset.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "track_genre" in df.columns:
            df = df.rename(columns={"track_genre": "genre"})
        print(f"已載入真實資料：{len(df):,} 首歌曲")
    else:
        np.random.seed(42)
        n = 40000
        year_probs = np.linspace(0.01, 0.3, 107)
        year_probs /= year_probs.sum()
        years = np.random.choice(range(1920, 2027), n, p=year_probs)
        danceability = np.clip(0.25 + 0.0004 * (years - 1920) + np.random.normal(0, 0.14, n), 0, 1)
        energy = np.clip(0.35 + 0.00025 * (years - 1920) + np.random.normal(0, 0.18, n), 0, 1)
        valence = np.clip(0.48 + np.random.normal(0, 0.2, n), 0, 1)
        acousticness = np.clip(0.55 - 0.0025 * (years - 1920) + np.random.normal(0, 0.22, n), 0, 1)
        loudness = np.clip(-14 + 0.018 * (years - 1920) + np.random.normal(0, 2.8, n), -60, 0)
        speechiness = np.clip(0.04 + 0.00008 * (years - 1920) + np.random.normal(0, 0.035, n), 0, 1)
        df = pd.DataFrame({
            "year": years,
            "genre": np.random.choice(["Pop","Hip-Hop","Rock","EDM","R&B","Country","Classical","Latin"], n,
                                       p=[0.25,0.18,0.15,0.12,0.10,0.08,0.07,0.05]),
            "danceability": danceability, "energy": energy, "valence": valence,
            "acousticness": acousticness, "loudness": loudness, "speechiness": speechiness,
        })
        print("使用模擬資料")
    # 代理變數
    df["age_group"] = pd.cut(df["year"],
        bins=[1900,1960,1980,2000,2010,2027],
        labels=["Pre-1960","1960s-70s","1980s-90s","2000s","2010s+"])
    prob_f = np.clip(0.5+0.2*df["valence"]-0.15*df["energy"]+0.1*df["danceability"],0.1,0.9)
    df["gender_proxy"] = np.where(np.random.random(len(df)) < prob_f, "Female", "Male")
    df["region"] = np.random.choice(["Western","Asia","Latin America"],len(df),p=[0.55,0.3,0.15])
    return df


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


if __name__ == "__main__":
    df = load_or_simulate()

    # ── 1. Shapiro-Wilk 正態性檢定 ──
    print("=" * 55)
    print("1. Shapiro-Wilk Normality Test (n=500 sample)")
    print("=" * 55)
    sample = df[FEATURES].sample(500, random_state=42)
    for feat in FEATURES:
        W, p = shapiro(sample[feat])
        result = "Non-normal → use non-parametric" if p < 0.05 else "Normal → parametric OK"
        print(f"  {feat:<16} W={W:.4f}  p={p:.4f}  {result}")

    # ── 2. Kruskal-Wallis：年齡群 ──
    print("\n" + "=" * 55)
    print("2. Kruskal-Wallis H Test: Age Group × Features")
    print("=" * 55)
    age_cats = df["age_group"].cat.categories
    for feat in FEATURES:
        groups = [df[df["age_group"] == g][feat].dropna().values for g in age_cats]
        H, p = kruskal(*groups)
        n_t = sum(len(g) for g in groups)
        eta2 = max((H - len(groups) + 1) / (n_t - len(groups)), 0)
        print(f"  {feat:<16} H={H:>8.1f}  p={p:.4f}{sig_stars(p):<4}  η²={eta2:.3f}")

    # ── 3. Mann-Whitney U：性別 ──
    print("\n" + "=" * 55)
    print("3. Mann-Whitney U Test: Gender × Features")
    print("=" * 55)
    female = df[df["gender_proxy"] == "Female"]
    male = df[df["gender_proxy"] == "Male"]
    n1, n2 = len(female), len(male)
    for feat in FEATURES:
        U, p = mannwhitneyu(female[feat], male[feat], alternative="two-sided")
        z = (U - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        r = abs(z) / np.sqrt(n1 + n2)
        print(f"  {feat:<16} U={U:>12.0f}  Z={z:>6.2f}  p={p:.4f}{sig_stars(p):<4}  r={r:.3f}")

    # ── 4. Kruskal-Wallis：地區 ──
    print("\n" + "=" * 55)
    print("4. Kruskal-Wallis H Test: Region × Features")
    print("=" * 55)
    regions = ["Western", "Asia", "Latin America"]
    for feat in FEATURES:
        groups = [df[df["region"] == r][feat].dropna().values for r in regions]
        H, p = kruskal(*groups)
        n_t = sum(len(g) for g in groups)
        eta2 = max((H - len(groups) + 1) / (n_t - len(groups)), 0)
        print(f"  {feat:<16} H={H:>8.2f}  p={p:.4f}{sig_stars(p):<4}  η²={eta2:.3f}")

    print("\nAll statistical tests complete.")
