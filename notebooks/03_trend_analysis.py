"""
03_trend_analysis.py
====================
歷史趨勢分析 + 統計學概念視覺化

圖表清單：
  05_trend_lines.png        — 各特徵每10年均值折線圖 + 95% CI 誤差帶
  06_normal_overlay.png     — 實際分佈 vs 常態分佈疊加 (QQ-Plot)
  07_qq_plots.png           — 6 個特徵的 Q-Q Plot
  08_kendall_heatmap.png    — Kendall's τ 相關矩陣熱圖
  09_loudness_war.png       — Loudness War 專題 (趨勢 + CI + 標注)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import kendalltau, norm

os.makedirs("outputs/figures", exist_ok=True)

FEATURES = ["danceability", "energy", "valence", "acousticness", "loudness", "speechiness"]
COLORS   = ["#4c8bbe", "#e07b39", "#5ba55b", "#c25b5b", "#9b6bbf", "#c2a73e"]
PALETTE  = dict(zip(FEATURES, COLORS))

# ── 0. Load / Simulate ─────────────────────────────────────────────────────────
def load_or_simulate(path="data/dataset.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "track_genre" in df.columns:
            df = df.rename(columns={"track_genre": "genre"})
        print(f"Loaded real data: {len(df):,} tracks")
    else:
        print("Simulating data (mirrors real Spotify distribution)...")
        np.random.seed(42)
        n = 40000
        year_probs = np.linspace(0.01, 0.3, 107)
        year_probs /= year_probs.sum()
        years = np.random.choice(range(1920, 2027), n, p=year_probs)
        danceability = np.clip(0.25 + 0.0004 * (years - 1920) + np.random.normal(0, 0.14, n), 0, 1)
        energy       = np.clip(0.35 + 0.00025 * (years - 1920) + np.random.normal(0, 0.18, n), 0, 1)
        valence      = np.clip(0.48 + np.random.normal(0, 0.2, n), 0, 1)
        acousticness = np.clip(0.55 - 0.0025 * (years - 1920) + np.random.normal(0, 0.22, n), 0, 1)
        loudness     = np.clip(-14 + 0.018 * (years - 1920) + np.random.normal(0, 2.8, n), -60, 0)
        speechiness  = np.clip(0.04 + 0.00008 * (years - 1920) + np.random.normal(0, 0.035, n), 0, 1)
        popularity   = np.clip(20 + 25 * danceability + 15 * energy + np.random.normal(0, 12, n), 0, 100)
        df = pd.DataFrame({
            "year": years, "danceability": danceability, "energy": energy,
            "valence": valence, "acousticness": acousticness,
            "loudness": loudness, "speechiness": speechiness,
            "popularity": popularity
        })
    return df


# ── 1. 歷史趨勢折線圖 + 95% CI ────────────────────────────────────────────────
def plot_trend_lines(df):
    """
    每10年 bin 計算：
      - 均值 (mean)
      - 95% 信賴區間 = mean ± 1.96 * SEM
      - SEM = std / sqrt(n)
    統計學意義：
      若 CI 不重疊 → 兩個時期差異達 p<0.05 顯著水準
    """
    df2 = df.copy()
    df2["decade"] = (df2["year"] // 10 * 10).astype(int)
    grouped = df2.groupby("decade")[FEATURES]

    means = grouped.mean()
    sems  = grouped.sem()                 # Standard Error of the Mean
    ci95  = sems * 1.96                   # 95% CI half-width

    decades = means.index.values

    feats_to_plot = ["danceability", "energy", "acousticness", "loudness", "speechiness"]
    fig, axes = plt.subplots(len(feats_to_plot), 1, figsize=(12, 16), sharex=True)
    fig.suptitle("Audio Feature Trends by Decade\n(Line = Mean, Band = 95% CI)",
                 fontsize=14, y=1.01)

    for ax, feat in zip(axes, feats_to_plot):
        mu  = means[feat].values
        ci  = ci95[feat].values
        c   = PALETTE[feat]

        ax.fill_between(decades, mu - ci, mu + ci, alpha=0.25, color=c, label="95% CI")
        ax.plot(decades, mu, color=c, linewidth=2.5, marker="o", markersize=5, label="Mean")

        # Kendall's τ
        tau, pval = kendalltau(decades, mu)
        direction = "↑" if tau > 0 else "↓"
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        ax.set_ylabel(feat, fontsize=10)
        ax.text(0.02, 0.88, f"Kendall τ = {tau:.3f}  {sig} {direction}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Decade", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/figures/05_trend_lines.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/05_trend_lines.png")


# ── 2. 常態分佈疊加圖 ─────────────────────────────────────────────────────────
def plot_normal_overlay(df):
    """
    實際 KDE 分佈 vs 理論常態分佈 N(μ, σ²)
    統計學意義：
      - 如果兩條曲線吻合 → 近似常態 (parametric test 適用)
      - 偏離 → 非常態 (需用 non-parametric: Kruskal-Wallis, Mann-Whitney)
    同時顯示：Skewness (偏態) 和 Kurtosis (峰態)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Actual Distribution vs Normal Distribution N(μ,σ²)\n"
                 "(KDE = actual, dashed = theoretical normal)",
                 fontsize=13)

    for ax, feat, c in zip(axes.flat, FEATURES, COLORS):
        data = df[feat].dropna().values
        mu, sigma = data.mean(), data.std()
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)   # excess kurtosis (normal=0)

        # Actual KDE
        x_range = np.linspace(data.min(), data.max(), 300)
        kde = stats.gaussian_kde(data)
        ax.fill_between(x_range, kde(x_range), alpha=0.4, color=c)
        ax.plot(x_range, kde(x_range), color=c, linewidth=2, label="Actual KDE")

        # Theoretical Normal
        normal_y = norm.pdf(x_range, mu, sigma)
        ax.plot(x_range, normal_y, "k--", linewidth=1.8, label=f"N({mu:.2f}, {sigma:.2f}²)")

        # μ ± σ vertical lines
        ax.axvline(mu,         color="gray",  linestyle="-",  alpha=0.6, linewidth=1)
        ax.axvline(mu - sigma, color="gray",  linestyle=":",  alpha=0.5, linewidth=1)
        ax.axvline(mu + sigma, color="gray",  linestyle=":",  alpha=0.5, linewidth=1)

        # Shapiro-Wilk (sample 5000 for speed)
        sample = np.random.choice(data, min(5000, len(data)), replace=False)
        _, sw_p = stats.shapiro(sample)
        sw_str = f"S-W p {'< 0.001' if sw_p < 0.001 else f'= {sw_p:.3f}'}"

        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        info = f"μ={mu:.3f}  σ={sigma:.3f}\nSkew={skew:.2f}  Kurt={kurt:.2f}\n{sw_str}"
        ax.text(0.97, 0.95, info, transform=ax.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    plt.tight_layout()
    plt.savefig("outputs/figures/06_normal_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/06_normal_overlay.png")


# ── 3. Q-Q Plot ───────────────────────────────────────────────────────────────
def plot_qq(df):
    """
    Quantile-Quantile Plot：最直觀的常態性檢驗視覺化
    - 點在對角線上 = 常態分佈
    - S 型偏離 = 有偏態 (skewed)
    - 兩端翹起 = 厚尾 (heavy-tailed, leptokurtic)
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Q-Q Plot (Quantile-Quantile): Normality Check\n"
                 "(Points on diagonal = normal; deviation = non-normal)",
                 fontsize=13)

    for ax, feat, c in zip(axes.flat, FEATURES, COLORS):
        data   = df[feat].dropna().values
        sample = np.random.choice(data, min(3000, len(data)), replace=False)

        (osm, osr), (slope, intercept, r) = stats.probplot(sample, dist="norm")

        ax.scatter(osm, osr, alpha=0.3, s=6, color=c, label="Data")
        x_line = np.array([osm.min(), osm.max()])
        ax.plot(x_line, slope * x_line + intercept, "k-", linewidth=2, label="Normal line")

        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.legend(fontsize=8)
        ax.text(0.05, 0.92, f"R² = {r**2:.4f}", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/figures/07_qq_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/07_qq_plots.png")


# ── 4. Kendall's τ 相關矩陣 ───────────────────────────────────────────────────
def plot_kendall_heatmap(df):
    """
    Kendall's τ：非常態資料的相關係數
    - 比 Pearson r 更穩健：不假設常態、不受離群值影響
    - τ = +1 完全正相關 / τ = -1 完全負相關 / τ = 0 無相關
    """
    n = len(FEATURES)
    tau_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))

    for i, f1 in enumerate(FEATURES):
        for j, f2 in enumerate(FEATURES):
            if i == j:
                tau_matrix[i, j] = 1.0
            else:
                tau, pval = kendalltau(df[f1].dropna(), df[f2].dropna())
                tau_matrix[i, j]  = tau
                pval_matrix[i, j] = pval

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(tau_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(FEATURES, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(FEATURES, fontsize=10)

    for i in range(n):
        for j in range(n):
            val = tau_matrix[i, j]
            p   = pval_matrix[i, j]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            text_color = "white" if abs(val) > 0.4 else "black"
            ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Kendall's τ")
    ax.set_title("Kendall's τ Correlation Matrix\n(* p<0.05  ** p<0.01  *** p<0.001)",
                 fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig("outputs/figures/08_kendall_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/08_kendall_heatmap.png")


# ── 5. Loudness War 專題圖 ─────────────────────────────────────────────────────
def plot_loudness_war(df):
    """
    Loudness War：1990s 後製作人競相把歌曲混音得越來越響
    統計學：
      - 折線 = 10年均值
      - 帶狀 = ±1σ (68% 的歌落在此範圍) vs 95% CI
      - 標注關鍵事件：CD era / Streaming era
    """
    df2 = df.copy()
    df2["decade"] = (df2["year"] // 10 * 10).astype(int)
    g = df2.groupby("decade")["loudness"]
    mu    = g.mean()
    sigma = g.std()
    sem   = g.sem()
    n_dec = g.count()
    decades = mu.index.values

    fig, ax = plt.subplots(figsize=(13, 6))

    # ±1σ band (68% range)
    ax.fill_between(decades, mu - sigma, mu + sigma,
                    alpha=0.15, color="#9b6bbf", label="±1σ (68% of songs)")
    # 95% CI band
    ci95 = sem * 1.96
    ax.fill_between(decades, mu - ci95, mu + ci95,
                    alpha=0.4, color="#9b6bbf", label="95% CI of mean")
    # Mean line
    ax.plot(decades, mu, color="#6a2f9e", linewidth=2.5,
            marker="o", markersize=6, label="Mean loudness")

    # Annotations
    annotations = {
        1980: ("CD Era begins\n(digital mastering)", -0.5),
        2000: ("Loudness War\npeak", -0.5),
        2010: ("Streaming\nnormalization", -0.5),
    }
    for year, (label, offset) in annotations.items():
        if year in decades:
            y_val = mu.loc[year]
            ax.annotate(label, xy=(year, y_val),
                        xytext=(year + 1, y_val + offset + 2),
                        fontsize=8, color="#444",
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))

    tau, pval = kendalltau(decades, mu.values)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
    ax.text(0.03, 0.08,
            f"Kendall τ = {tau:.3f} {sig}\n(loudness increases over time)",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("Decade", fontsize=11)
    ax.set_ylabel("Loudness (dBFS)", fontsize=11)
    ax.set_title("The Loudness War: Mean Loudness by Decade\n"
                 "(Band = ±1σ shows individual song spread; CI shows mean uncertainty)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/09_loudness_war.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/figures/09_loudness_war.png")


# ── 6. Kendall τ Summary Table ────────────────────────────────────────────────
def print_kendall_summary(df):
    """
    對每個特徵計算與 year 的 Kendall τ，輸出摘要表
    """
    print("\n=== Kendall's τ: Feature vs Year ===")
    print(f"{'Feature':<15} {'τ':>8} {'p-value':>12} {'Significance':>14} {'Direction':>10}")
    print("-" * 65)
    for feat in FEATURES:
        sub = df[["year", feat]].dropna()
        tau, pval = kendalltau(sub["year"], sub[feat])
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        direction = "Increasing ↑" if tau > 0 else "Decreasing ↓"
        print(f"{feat:<15} {tau:>8.4f} {pval:>12.2e} {sig:>14} {direction:>10}")


if __name__ == "__main__":
    df = load_or_simulate()

    print("\n[1/5] Trend lines with 95% CI ...")
    plot_trend_lines(df)

    print("[2/5] Normal distribution overlay + Shapiro-Wilk ...")
    plot_normal_overlay(df)

    print("[3/5] Q-Q plots ...")
    plot_qq(df)

    print("[4/5] Kendall's τ heatmap ...")
    plot_kendall_heatmap(df)

    print("[5/5] Loudness War analysis ...")
    plot_loudness_war(df)

    print_kendall_summary(df)

    print("\n✓ All charts saved to outputs/figures/")
    print("  05_trend_lines.png      — 趨勢折線 + 95% CI")
    print("  06_normal_overlay.png   — 實際分佈 vs 常態分佈")
    print("  07_qq_plots.png         — Q-Q Plot 常態性檢驗")
    print("  08_kendall_heatmap.png  — Kendall τ 相關矩陣")
    print("  09_loudness_war.png     — Loudness War 專題")
