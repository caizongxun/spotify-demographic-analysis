"""
07_stat_analysis.py
統計推論分析 — 年齡 / 性別 / 地區 × 音樂特徵

統計方法鏈：
  [常態性]  Shapiro-Wilk → 決定後續用參數 or 無母數
  [年齡]    常態後用 One-way ANOVA + η²；非常態用 Kruskal-Wallis + ε²
            → Post-hoc Tukey HSD / Dunn → 推論哪兩個年齡層顯著不同
  [性別]    常態後用 Independent t-test + Cohen's d；非常態用 Mann-Whitney U
            → Point-biserial r → 推論性別對音樂特徵的效應大小
  [地區]    每洲視為群體，One-way ANOVA / K-W + η²
            → Dunn post-hoc → 推論哪兩洲間特徵差異顯著
  [常態分布]Z-score 標準化後驗證 95% 信賴區間覆蓋率是否符合理論 95%
            → 用於報告「音樂特徵服從常態假設的程度」

輸出：
  outputs/stat_report.txt  — 所有推論數字的純文字報告
  outputs/figures_zh/S1_常態檢定QQ圖.png
  outputs/figures_zh/S2_年齡ANOVA結果.png
  outputs/figures_zh/S3_性別Cohen_d.png
  outputs/figures_zh/S4_地區分組柱狀圖.png
  outputs/figures_zh/S5_地區顯著性矩陣.png
  outputs/figures_zh/S6_地區Eta平方.png
"""

import os, subprocess, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu, f_oneway, spearmanr
from itertools import combinations
warnings.filterwarnings('ignore')

# ── 字體 ─────────────────────────────────────────────────────────────────
subprocess.run(['apt-get', 'install', '-y', '-q', 'fonts-noto-cjk'], check=False)
_CJK = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
]
_font_path = next((p for p in _CJK if os.path.exists(p)), None)
if _font_path:
    fm.fontManager.addfont(_font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

def fp():
    return fm.FontProperties(fname=_font_path) if _font_path else None

# ── 路徑 ─────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
OUT      = os.path.join(BASE, 'outputs', 'figures_zh')
REPORT   = os.path.join(BASE, 'outputs', 'stat_report.txt')
os.makedirs(OUT, exist_ok=True)

TRACKS_CSV = os.path.join(DATA_DIR, 'tracks.csv')
USERS_CSV  = os.path.join(DATA_DIR, 'users.csv')
GLOBAL_CSV = os.path.join(DATA_DIR, 'global_top50.csv')

# ── 載入資料（同 06_chinese_charts.py 的邏輯）────────────────────────────
tracks = pd.read_csv(TRACKS_CSV)
if 'track_genre' in tracks.columns:
    tracks = tracks.rename(columns={'track_genre': 'genre'})

users = pd.read_csv(USERS_CSV)
users.columns = users.columns.str.lower().str.replace(' ', '_')
for col in ['favorite_genre', 'preferred_genre', 'music_genre',
            'fav_music_genre', 'music_preferences']:
    if col in users.columns:
        users = users.rename(columns={col: 'fav_genre'})
        break
if 'user_age' in users.columns:
    users = users.rename(columns={'user_age': 'age'})

def parse_age(s):
    def _one(v):
        v = str(v).strip()
        if '-' in v:
            p = v.split('-')
            try: return (float(p[0]) + float(p[1])) / 2
            except: return np.nan
        if v.endswith('+'):
            try: return float(v[:-1]) + 5
            except: return np.nan
        try: return float(v)
        except: return np.nan
    return s.apply(_one)

if 'age' in users.columns:
    users['age_num'] = parse_age(users['age'])
else:
    users['age_num'] = np.nan

FEATURES = ['danceability', 'energy', 'valence',
            'acousticness', 'speechiness', 'instrumentalness']
FEATURE_ZH = {
    'danceability':     '舞動度',
    'energy':           '能量',
    'valence':          '情緒正向度',
    'acousticness':     '原聲度',
    'speechiness':      '語音度',
    'instrumentalness': '純器樂度',
}
AGE_ORDER = ['13-17', '18-24', '25-34', '35-44', '45+']
AGE_ZH    = ['13-17歲', '18-24歲', '25-34歲', '35-44歲', '45歲以上']

track_means = tracks.groupby('genre')[FEATURES].mean().reset_index()
track_means['genre_lower'] = track_means['genre'].str.lower().str.strip()
users['fav_genre_lower'] = (
    users['fav_genre'].str.lower().str.strip()
    if 'fav_genre' in users.columns else ''
)
df = users.merge(track_means.drop(columns='genre'),
                 left_on='fav_genre_lower', right_on='genre_lower', how='left')
df.drop(columns=['fav_genre_lower', 'genre_lower'], inplace=True, errors='ignore')
df['age_group'] = pd.cut(
    df['age_num'], bins=[12, 17, 24, 34, 44, 100],
    labels=AGE_ORDER
)

# 地區資料
COUNTRY_TO_CONTINENT = {
    'US':'北美洲','CA':'北美洲','MX':'北美洲',
    'GB':'歐洲','DE':'歐洲','FR':'歐洲','IT':'歐洲','ES':'歐洲',
    'NL':'歐洲','SE':'歐洲','NO':'歐洲','DK':'歐洲','FI':'歐洲',
    'PL':'歐洲','BE':'歐洲','CH':'歐洲','AT':'歐洲','PT':'歐洲',
    'IE':'歐洲','CZ':'歐洲','HU':'歐洲','RO':'歐洲','GR':'歐洲',
    'TR':'歐洲','SK':'歐洲','HR':'歐洲',
    'BR':'南美洲','AR':'南美洲','CL':'南美洲','CO':'南美洲',
    'PE':'南美洲','VE':'南美洲','EC':'南美洲','BO':'南美洲',
    'UY':'南美洲','PY':'南美洲',
    'JP':'亞洲','KR':'亞洲','TW':'亞洲','HK':'亞洲','SG':'亞洲',
    'MY':'亞洲','TH':'亞洲','PH':'亞洲','ID':'亞洲','VN':'亞洲',
    'IN':'亞洲','PK':'亞洲','IL':'亞洲',
    'AU':'大洋洲','NZ':'大洋洲',
    'ZA':'非洲','NG':'非洲','EG':'非洲','MA':'非洲',
    'GT':'中美洲','SV':'中美洲','HN':'中美洲',
    'CR':'中美洲','PA':'中美洲','DO':'中美洲',
}
CONTINENT_COLORS = {
    '北美洲':'#4472C4','歐洲':'#ED7D31','南美洲':'#70AD47',
    '亞洲':'#FF4B4B','大洋洲':'#9B59B6','非洲':'#1ABC9C','中美洲':'#F39C12',
}

gdf = pd.read_csv(GLOBAL_CSV)
gdf.columns = gdf.columns.str.lower().str.strip().str.replace(' ', '_')
gdf['continent'] = gdf['country'].str.upper().str.strip().map(COUNTRY_TO_CONTINENT)
gdf = gdf[gdf['continent'].notna()]
continent_feats = [f for f in FEATURES if f in gdf.columns]

# ── 工具函數 ─────────────────────────────────────────────────────────────
lines = []  # 報告文字

def log(s=''):
    print(s)
    lines.append(s)

def sig_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def cohens_d(a, b):
    """兩組樣本的 Cohen's d"""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2)
                         / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

def eta_squared_kw(H, k, N):
    """Kruskal-Wallis η² = (H - k + 1) / (N - k)"""
    return max(0, (H - k + 1) / (N - k))

def eta_squared_anova(groups):
    """One-way ANOVA η² = SS_between / SS_total"""
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total   = sum(np.sum((g - grand_mean)**2) for g in groups)
    return ss_between / ss_total if ss_total > 0 else 0.0

def dunn_posthoc(data_dict):
    """
    Dunn's test (Bonferroni 校正)
    回傳 DataFrame: index/columns = 組名, 值 = 校正後 p 值
    """
    keys = list(data_dict.keys())
    all_vals = np.concatenate(list(data_dict.values()))
    N = len(all_vals)
    ranks = stats.rankdata(all_vals)
    rank_dict = {}
    idx = 0
    for k in keys:
        n = len(data_dict[k])
        rank_dict[k] = ranks[idx:idx+n]
        idx += n

    pairs = list(combinations(keys, 2))
    m = len(pairs)
    result = pd.DataFrame(np.nan, index=keys, columns=keys)
    for (i, j) in pairs:
        ni, nj = len(rank_dict[i]), len(rank_dict[j])
        ri, rj = rank_dict[i].mean(), rank_dict[j].mean()
        se = np.sqrt((N*(N+1)/12) * (1/ni + 1/nj))
        z = abs(ri - rj) / se if se > 0 else 0
        p = 2 * (1 - stats.norm.cdf(z))  # 雙尾
        p_adj = min(p * m, 1.0)           # Bonferroni 校正
        result.loc[i, j] = p_adj
        result.loc[j, i] = p_adj
    np.fill_diagonal(result.values, 1.0)
    return result


# ═══════════════════════════════════════════════════════════════
# S1. 常態性檢定 + Z-score 95% CI 覆蓋率驗證
# ═══════════════════════════════════════════════════════════════
def s1_normality():
    log('='*60)
    log('S1. 常態性檢定（Shapiro-Wilk）+ Z-score 95% CI 覆蓋率')
    log('='*60)
    log('理論基礎：若 X ~ N(μ, σ²)，則標準化後 Z = (X-μ)/σ ~ N(0,1)')
    log('          95% 的資料應落在 μ ± 1.96σ 範圍內')
    log()

    normal_feats = []  # 通過常態的特徵
    nonnormal_feats = []
    sw_results = {}

    for feat in FEATURES:
        data = tracks[feat].dropna().values
        # Shapiro-Wilk（樣本大時用子集）
        sample = data if len(data) <= 5000 else np.random.choice(data, 5000, replace=False)
        W, p = shapiro(sample)
        sw_results[feat] = (W, p)
        verdict = '通過(常態)' if p > 0.05 else '拒絕(非常態)'
        log(f'  {FEATURE_ZH[feat]:8s}  W={W:.4f}  p={p:.4e}  → {verdict}')

        # Z-score 95% CI 實際覆蓋率
        mu, sigma = data.mean(), data.std()
        within = np.sum(np.abs(data - mu) <= 1.96 * sigma) / len(data) * 100
        log(f'             μ={mu:.3f}  σ={sigma:.3f}  實際落在μ±1.96σ內: {within:.1f}%')
        log(f'             理論值: 95.0%  偏差: {within-95:.1f}%')

        if p > 0.05:
            normal_feats.append(feat)
        else:
            nonnormal_feats.append(feat)

    log()
    log(f'→ 推論：{len(normal_feats)} 個特徵通過常態（適用參數檢定）：{[FEATURE_ZH[f] for f in normal_feats]}')
    log(f'        {len(nonnormal_feats)} 個特徵拒絕常態（改用無母數檢定）：{[FEATURE_ZH[f] for f in nonnormal_feats]}')

    # Q-Q Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    F = fp()
    fig.suptitle('S1：各音樂特徵 Q-Q 圖（常態性檢定）', fontsize=14,
                 fontweight='bold', fontproperties=F)
    for ax, feat in zip(axes.flat, FEATURES):
        data = tracks[feat].dropna().values
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm')
        ax.scatter(osm, osr, s=4, alpha=0.3, color='steelblue')
        ax.plot([osm.min(), osm.max()],
                [slope*osm.min()+intercept, slope*osm.max()+intercept],
                color='red', lw=1.5)
        W, p = sw_results[feat]
        verdict = '常態' if p > 0.05 else '非常態'
        ax.set_title(f"{FEATURE_ZH[feat]}\nW={W:.3f}  p={p:.2e} [{verdict}]",
                     fontsize=10, fontproperties=F)
        ax.set_xlabel('理論分位數', fontsize=9, fontproperties=F)
        ax.set_ylabel('樣本分位數', fontsize=9, fontproperties=F)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'S1_常態檢定QQ圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log('已儲存：S1_常態檢定QQ圖.png')

    return normal_feats, nonnormal_feats


# ═══════════════════════════════════════════════════════════════
# S2. 年齡 × 音樂特徵：ANOVA / K-W + η² + Post-hoc
# ═══════════════════════════════════════════════════════════════
def s2_age(normal_feats, nonnormal_feats):
    log()
    log('='*60)
    log('S2. 年齡層 × 音樂特徵：ANOVA / Kruskal-Wallis + η² + Tukey')
    log('='*60)
    log('H₀：各年齡層在該音樂特徵的分布無顯著差異')
    log('H₁：至少一個年齡層存在顯著差異')
    log()

    results = {}  # feat -> {stat, p, eta2, method}
    for feat in FEATURES:
        groups = [
            df[df['age_group'] == ag][feat].dropna().values
            for ag in AGE_ORDER
        ]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue

        if feat in normal_feats:
            F_stat, p = f_oneway(*groups)
            eta2 = eta_squared_anova(groups)
            method = 'ANOVA'
            results[feat] = {'stat': F_stat, 'p': p, 'eta2': eta2, 'method': method}
            log(f'  {FEATURE_ZH[feat]:8s} [{method}]  F={F_stat:.3f}  p={p:.4e}  η²={eta2:.3f}  {sig_star(p)}')
        else:
            N = sum(len(g) for g in groups)
            k = len(groups)
            H, p = kruskal(*groups)
            eta2 = eta_squared_kw(H, k, N)
            method = 'K-W'
            results[feat] = {'stat': H, 'p': p, 'eta2': eta2, 'method': method}
            log(f'  {FEATURE_ZH[feat]:8s} [{method}]   H={H:.3f}  p={p:.4e}  η²={eta2:.3f}  {sig_star(p)}')

    sig_feats = [f for f, r in results.items() if r['p'] < 0.05]
    log()
    log(f'→ 推論：{[FEATURE_ZH[f] for f in sig_feats]} 在年齡層間存在顯著差異')
    log('        η² 越大表示年齡能解釋越多該特徵的變異')

    # 視覺化：η² 條形圖 + 顯著性
    fig, ax = plt.subplots(figsize=(10, 5))
    F = fp()
    feat_names = [FEATURE_ZH[f] for f in FEATURES if f in results]
    eta2_vals  = [results[f]['eta2'] for f in FEATURES if f in results]
    p_vals     = [results[f]['p']    for f in FEATURES if f in results]
    colors_bar = ['#c0392b' if p < 0.05 else '#7f8c8d' for p in p_vals]
    bars = ax.barh(feat_names, eta2_vals, color=colors_bar, alpha=0.85)
    for bar, p, eta2 in zip(bars, p_vals, eta2_vals):
        label = f'{sig_star(p)}  η²={eta2:.3f}'
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=10, fontproperties=F)
    ax.set_xlabel('Eta-squared (η²)  — 年齡層解釋的變異比例', fontsize=11, fontproperties=F)
    ax.set_title('S2：年齡層對各音樂特徵的效應大小（η²）\n紅色=顯著(p<0.05)  灰色=不顯著',
                 fontsize=12, fontproperties=F)
    ax.set_xlim(0, max(eta2_vals) * 1.35 if eta2_vals else 0.1)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(F)
    ax.axvline(0.01, color='orange', lw=1, linestyle='--', label='小效應(0.01)')
    ax.axvline(0.06, color='green',  lw=1, linestyle='--', label='中效應(0.06)')
    ax.axvline(0.14, color='purple', lw=1, linestyle='--', label='大效應(0.14)')
    ax.legend(fontsize=9, prop=F)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'S2_年齡ANOVA結果.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log('已儲存：S2_年齡ANOVA結果.png')
    return results


# ═══════════════════════════════════════════════════════════════
# S3. 性別 × 音樂特徵：t-test / Mann-Whitney U + Cohen's d
# ═══════════════════════════════════════════════════════════════
def s3_gender(normal_feats):
    log()
    log('='*60)
    log('S3. 性別 × 音樂特徵：Independent t-test / Mann-Whitney U + Cohen\'s d')
    log('='*60)
    log('H₀：男性與女性在該音樂特徵上的分布無顯著差異')
    log('H₁：兩性別存在顯著差異')
    log()
    if 'gender' not in df.columns:
        log('  找不到 gender 欄位，跳過 S3')
        return

    male   = df[df['gender'] == 'Male']
    female = df[df['gender'] == 'Female']
    if len(male) < 5 or len(female) < 5:
        log('  男/女樣本數不足，跳過 S3')
        return

    cohen_ds = {}
    for feat in FEATURES:
        m = male[feat].dropna().values
        f = female[feat].dropna().values
        if len(m) < 3 or len(f) < 3:
            continue

        if feat in normal_feats:
            stat, p = stats.ttest_ind(m, f)
            method = 't-test'
        else:
            stat, p = mannwhitneyu(m, f, alternative='two-sided')
            method = 'MWU'

        d = cohens_d(m, f)
        cohen_ds[feat] = d

        effect = '小' if abs(d) < 0.2 else ('中' if abs(d) < 0.5 else '大')
        log(f'  {FEATURE_ZH[feat]:8s} [{method}]  stat={stat:.3f}  p={p:.4e}  '
            f'd={d:+.3f}  效應={effect}  {sig_star(p)}')

    sig_feats = [f for f, d in cohen_ds.items()
                 if abs(d) >= 0.2]
    log()
    log(f'→ 推論：{[FEATURE_ZH[f] for f in sig_feats]} 的 Cohen\'s d ≥ 0.2，')
    log('        表示性別在這些特徵上有實際意義的差異（非僅統計顯著）')

    # 視覺化：Cohen's d 橫條圖
    fig, ax = plt.subplots(figsize=(10, 5))
    F = fp()
    feats_sorted = sorted(cohen_ds, key=lambda x: cohen_ds[x])
    d_vals = [cohen_ds[f] for f in feats_sorted]
    bar_colors = ['#c0392b' if d > 0 else '#2980b9' for d in d_vals]
    ax.barh([FEATURE_ZH[f] for f in feats_sorted], d_vals,
            color=bar_colors, alpha=0.85)
    ax.axvline(0,    color='black', lw=1)
    ax.axvline(0.2,  color='orange', lw=1, linestyle='--', label='小效應 |d|=0.2')
    ax.axvline(-0.2, color='orange', lw=1, linestyle='--')
    ax.axvline(0.5,  color='green',  lw=1, linestyle='--', label='中效應 |d|=0.5')
    ax.axvline(-0.5, color='green',  lw=1, linestyle='--')
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(F)
    ax.set_xlabel("Cohen's d  (正=男性>女性  負=女性>男性)", fontsize=11, fontproperties=F)
    ax.set_title("S3：性別對各音樂特徵的效應大小（Cohen's d）\n紅=男性偏高  藍=女性偏高",
                 fontsize=12, fontproperties=F)
    ax.legend(fontsize=9, prop=F)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'S3_性別Cohen_d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log('已儲存：S3_性別Cohen_d.png')


# ═══════════════════════════════════════════════════════════════
# S4. 地區 — 分組柱狀圖 + K-W 顯著性標記
# ═══════════════════════════════════════════════════════════════
def s4_region_bar():
    log()
    log('='*60)
    log('S4. 地區 × 音樂特徵：分組柱狀圖 + Kruskal-Wallis 顯著性')
    log('='*60)

    continents = [c for c in CONTINENT_COLORS if c != '全球'
                  and c in gdf['continent'].values]
    feat_means = gdf.groupby('continent')[continent_feats].mean()

    F = fp()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('S4：各洲別音樂特徵平均值（分組柱狀圖）', fontsize=14,
                 fontweight='bold', fontproperties=F)

    for ax, feat in zip(axes.flat, continent_feats):
        cont_list = [c for c in continents if c in feat_means.index]
        means = [feat_means.loc[c, feat] for c in cont_list]
        bar_colors = [CONTINENT_COLORS.get(c, '#888888') for c in cont_list]
        bars = ax.bar(cont_list, means, color=bar_colors, alpha=0.85, width=0.6)

        # K-W 檢定（每首歌為樣本）
        groups = [gdf[gdf['continent'] == c][feat].dropna().values
                  for c in cont_list]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            N = sum(len(g) for g in groups)
            k = len(groups)
            eta2 = eta_squared_kw(H, k, N)
            star = sig_star(p)
            title_str = (f'{FEATURE_ZH[feat]}\n'
                         f'K-W H={H:.1f}  p={p:.3e}  {star}  η²={eta2:.3f}')
        else:
            title_str = FEATURE_ZH[feat]

        ax.set_title(title_str, fontsize=10, fontproperties=F)
        ax.set_ylabel('平均值 (0-1)', fontsize=9, fontproperties=F)
        ax.set_ylim(0, min(max(means)*1.25, 1.0) if means else 1.0)
        ax.tick_params(axis='x', rotation=35)
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(F)
            tick.set_fontsize(9)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'S4_地區分組柱狀圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log('已儲存：S4_地區分組柱狀圖.png')


# ═══════════════════════════════════════════════════════════════
# S5. 地區 — Dunn Post-hoc 顯著性矩陣（以 danceability 為例）
# ═══════════════════════════════════════════════════════════════
def s5_region_posthoc():
    log()
    log('='*60)
    log('S5. 地區 Post-hoc Dunn\'s test（舞動度）')
    log('    目的：找出哪兩個洲之間差異最顯著')
    log('='*60)

    feat = 'danceability'
    continents = sorted(gdf['continent'].dropna().unique())
    data_dict = {
        c: gdf[gdf['continent'] == c][feat].dropna().values
        for c in continents if len(gdf[gdf['continent'] == c]) >= 3
    }
    if len(data_dict) < 2:
        log('  樣本不足，跳過 S5')
        return

    pmat = dunn_posthoc(data_dict)
    keys = list(data_dict.keys())

    # 轉成數值 heatmap
    pmat_num = pmat.loc[keys, keys].astype(float)
    sig_matrix = (pmat_num < 0.05).astype(int)  # 1=顯著 0=不顯著

    F = fp()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(pmat_num.values, cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Bonferroni 校正後 p 值')
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=40, fontproperties=F)
    ax.set_yticklabels(keys, fontproperties=F)
    ax.set_title(f'S5：洲別兩兩比較顯著性矩陣（{FEATURE_ZH[feat]}）\n'
                 f'綠色=不顯著  紅色=顯著差異(p<0.05)',
                 fontsize=12, fontproperties=F)

    for i in range(len(keys)):
        for j in range(len(keys)):
            p_val = pmat_num.iloc[i, j]
            star  = sig_star(p_val)
            color = 'white' if p_val < 0.3 else 'black'
            ax.text(j, i, f'{p_val:.2f}\n{star}',
                    ha='center', va='center', fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'S5_地區顯著性矩陣.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log('已儲存：S5_地區顯著性矩陣.png')

    # 推論文字
    sig_pairs = [(keys[i], keys[j])
                 for i, j in combinations(range(len(keys)), 2)
                 if pmat_num.iloc[i, j] < 0.05]
    log(f'→ 推論：在舞動度上，以下洲別對存在顯著差異（p<0.05）：')
    for a, b in sig_pairs:
        ma = data_dict[a].mean()
        mb = data_dict[b].mean()
        higher = a if ma > mb else b
        log(f'     {a} vs {b}  →  {higher} 的舞動度顯著較高')


# ═══════════════════════════════════════════════════════════════
# S6. 地區 — Eta² 條形圖（哪個特徵最能區分地區）
# ═══════════════════════════════════════════════════════════════
def s6_region_eta2():
    log()
    log('='*60)
    log('S6. 地區對各音樂特徵的解釋力（η²）')
    log('    目的：找出哪個音樂特徵最能區分不同洲別')
    log('='*60)

    eta2_dict = {}
    p_dict     = {}
    continents = sorted(gdf['continent'].dropna().unique())
    for feat in continent_feats:
        groups = [gdf[gdf['continent'] == c][feat].dropna().values
                  for c in continents]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        N = sum(len(g) for g in groups)
        k = len(groups)
        H, p = kruskal(*groups)
        eta2 = eta_squared_kw(H, k, N)
        eta2_dict[feat] = eta2
        p_dict[feat]    = p
        log(f'  {FEATURE_ZH[feat]:8s}  H={H:.2f}  p={p:.4e}  η²={eta2:.4f}  {sig_star(p)}')

    best_feat = max(eta2_dict, key=eta2_dict.get)
    log()
    log(f'→ 推論：{FEATURE_ZH[best_feat]} 的 η²={eta2_dict[best_feat]:.3f} 最高，')
    log('        表示地區因素對該特徵的變異解釋力最強，')
    log('        即「最能透過音樂特徵區分各洲音樂文化」')

    F = fp()
    fig, ax = plt.subplots(figsize=(10, 5))
    feats_sorted = sorted(eta2_dict, key=eta2_dict.get)
    eta2_vals = [eta2_dict[f] for f in feats_sorted]
    p_vals    = [p_dict[f]    for f in feats_sorted]
    bar_colors = ['#c0392b' if p < 0.05 else '#7f8c8d' for p in p_vals]
    bars = ax.barh([FEATURE_ZH[f] for f in feats_sorted], eta2_vals,
                   color=bar_colors, alpha=0.85)
    for bar, p, eta2 in zip(bars, p_vals, eta2_vals):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f'{sig_star(p)}  η²={eta2:.3f}', va='center',
                fontsize=10, fontproperties=F)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(F)
    ax.set_xlabel('Eta-squared (η²)  — 地區解釋的變異比例', fontsize=11, fontproperties=F)
    ax.set_title('S6：哪個音樂特徵最能區分不同地區？\n紅色=顯著(p<0.05)  灰=不顯著',
                 fontsize=12, fontproperties=F)
    ax.axvline(0.01, color='orange', lw=1, linestyle='--', label='小效應')
    ax.axvline(0.06, color='green',  lw=1, linestyle='--', label='中效應')
    ax.axvline(0.14, color='purple', lw=1, linestyle='--', label='大效應')
    ax.legend(fontsize=9, prop=F)
    ax.set_xlim(0, max(eta2_vals)*1.4 if eta2_vals else 0.1)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'S6_地區Eta平方.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log('已儲存：S6_地區Eta平方.png')


# ── 執行 ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)
    log('Spotify 音樂特徵統計推論報告')
    log(f'資料筆數：tracks={len(tracks)}  users={len(df)}  global={len(gdf)}')

    normal_feats, nonnormal_feats = s1_normality()
    s2_age(normal_feats, nonnormal_feats)
    s3_gender(normal_feats)
    s4_region_bar()
    s5_region_posthoc()
    s6_region_eta2()

    # 儲存報告
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n統計報告已儲存：{REPORT}')
    print('全部完成！')
