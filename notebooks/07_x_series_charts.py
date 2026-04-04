"""
07_x_series_charts.py
X 系列圖表 —— 修正第 6、7、9、10、11 頁文不對圖問題

檔名規則：  X{N}_p{pages}_{content}.png

圖表清單:
  X1_p6_主要特徵直方圖KDE_舞動度能量情緒.png       → 第6頁
  X2_p7_右偏特徵直方圖KDE_原聲度語音度器樂度.png   → 第7頁
  X3_p9_年齡折線圖含95CI誤差線.png                 → 第9頁
  X4_p10_性別Genre比例圖含信賴區間.png             → 第10頁
  X5_p11_地區效應量_eta2_橫向柱狀圖.png            → 第11頁
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import gaussian_kde

# ── 字體 ──────────────────────────────────────────────────────────────────
subprocess.run(['apt-get', 'install', '-y', '-q', 'fonts-noto-cjk'], check=False)
_CJK = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
]
_font_path = next((p for p in _CJK if os.path.exists(p)), None)
if _font_path:
    fm.fontManager.addfont(_font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

def fp():
    return fm.FontProperties(fname=_font_path) if _font_path else None

# ── 路徑 ──────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE, 'data')
OUT       = os.path.join(BASE, 'outputs', 'figures_zh')
os.makedirs(OUT, exist_ok=True)
TRACKS_CSV = os.path.join(DATA_DIR, 'tracks.csv')
USERS_CSV  = os.path.join(DATA_DIR, 'users.csv')
GLOBAL_CSV = os.path.join(DATA_DIR, 'global_top50.csv')

# ── 資料載入 ──────────────────────────────────────────────────────────────
tracks = pd.read_csv(TRACKS_CSV)
if 'track_genre' in tracks.columns:
    tracks = tracks.rename(columns={'track_genre': 'genre'})

users = pd.read_csv(USERS_CSV)
users.columns = users.columns.str.lower().str.replace(' ', '_')
for col in ['favorite_genre', 'preferred_genre', 'music_genre',
            'fav_music_genre', 'music_preferences']:
    if col in users.columns:
        users = users.rename(columns={col: 'fav_genre'}); break
if 'user_age' in users.columns:
    users = users.rename(columns={'user_age': 'age'})

def parse_age(s):
    def _one(v):
        v = str(v).strip()
        if '-' in v:
            p = v.split('-')
            try: return (float(p[0]) + float(p[1])) / 2
            except: return np.nan
        if v.endswith('+'): return float(v[:-1]) + 5 if v[:-1].replace('.','').isdigit() else np.nan
        try: return float(v)
        except: return np.nan
    return s.apply(_one)

users['age_num'] = parse_age(users['age']) if 'age' in users.columns else np.nan

AUDIO = ['danceability', 'energy', 'valence',
         'acousticness', 'speechiness', 'instrumentalness']
FEATURE_ZH = {
    'danceability':     '舞動度',
    'energy':           '能量',
    'valence':          '情緒正向度',
    'acousticness':     '原聲度',
    'speechiness':      '語音度',
    'instrumentalness': '純器樂度',
}

track_means = tracks.groupby('genre')[AUDIO].mean().reset_index()
track_means['gl'] = track_means['genre'].str.lower().str.strip()
users['gl'] = users['fav_genre'].str.lower().str.strip() if 'fav_genre' in users.columns else ''
df = users.merge(track_means.drop(columns='genre'), on='gl', how='left').drop(columns='gl', errors='ignore')
df['age_group'] = pd.cut(df['age_num'], bins=[12,17,24,34,44,100],
                         labels=['13-17','18-24','25-34','35-44','45+'])

AGE_ORDER = ['13-17','18-24','25-34','35-44','45+']
AGE_ZH    = ['13-17歲','18-24歲','25-34歲','35-44歲','45歲以上']
GENDER_ZH = {'Male':'男性','Female':'女性','Non-binary':'非二元性別'}

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
    'JP':'亞洲','KR':'亞洲','TW':'亞洲','HK':'亞洲',
    'SG':'亞洲','MY':'亞洲','TH':'亞洲','PH':'亞洲',
    'ID':'亞洲','VN':'亞洲','IN':'亞洲','PK':'亞洲','IL':'亞洲',
    'AU':'大洋洲','NZ':'大洋洲',
    'ZA':'非洲','NG':'非洲','EG':'非洲','MA':'非洲',
    'GT':'中美洲','SV':'中美洲','HN':'中美洲',
    'CR':'中美洲','PA':'中美洲','DO':'中美洲',
}


# ========================================================================
# X1  p6  主要特徵直方圖+KDE  舞動度/能量/情緒正向度
# ========================================================================
def plot_x1_p6_main_features():
    feats = ['danceability', 'energy', 'valence']
    COLOR = '#4c8bbe'
    MEAN_COLORS = {'danceability':'#e74c3c', 'energy':'#e67e22', 'valence':'#8e44ad'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('主要音樂特徵分布與點估計（舞動度 / 能量 / 情緒正向度）',
                 fontsize=14, fontweight='bold', fontproperties=fp())

    summaries = []
    for ax, feat in zip(axes, feats):
        data = tracks[feat].dropna()
        mn = data.mean()
        summaries.append(f'{FEATURE_ZH[feat]}: 平均={mn:.3f}')

        ax.hist(data, bins=60, color=COLOR, alpha=0.45, density=True, label='分布')
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 400)
        ax.plot(xs, kde(xs), color='#1a5276', lw=2.2, label='KDE')
        ax.axvline(mn, color=MEAN_COLORS[feat], lw=2, linestyle='--',
                   label=f'平均 = {mn:.2f}')
        ax.set_title(FEATURE_ZH[feat], fontsize=13, fontproperties=fp())
        ax.set_xlabel('數値 (0–1)', fontsize=10, fontproperties=fp())
        ax.set_ylabel('機率密度', fontsize=10, fontproperties=fp())
        ax.legend(fontsize=9, prop=fp())
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = 'X1_p6_主要特徵直方圖KDE_舞動度能量情緒.png'
    plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n已儲存：{fname}')
    print('數字摘要 (X1 第6頁):')
    for s in summaries:
        print('  ', s)


# ========================================================================
# X2  p7  右偏特徵直方圖+KDE  原聲度/語音度/純器樂度
# ========================================================================
def plot_x2_p7_skewed_features():
    feats = ['acousticness', 'speechiness', 'instrumentalness']
    COLOR = '#e8a838'
    MEAN_COLORS = {'acousticness':'#2980b9', 'speechiness':'#27ae60', 'instrumentalness':'#c0392b'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('極端右偏音樂特徵分布（原聲度 / 語音度 / 純器樂度）',
                 fontsize=14, fontweight='bold', fontproperties=fp())

    summaries = []
    for ax, feat in zip(axes, feats):
        data = tracks[feat].dropna()
        mn = data.mean()
        from scipy.stats import skew
        sk = skew(data)
        summaries.append(f'{FEATURE_ZH[feat]}: 平均={mn:.3f}  偏態={sk:.2f}')

        ax.hist(data, bins=60, color=COLOR, alpha=0.45, density=True, label='分布')
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 400)
        ax.plot(xs, kde(xs), color='#7d3c00', lw=2.2, label='KDE')
        ax.axvline(mn, color=MEAN_COLORS[feat], lw=2, linestyle='--',
                   label=f'平均 = {mn:.2f}')
        ax.set_title(FEATURE_ZH[feat], fontsize=13, fontproperties=fp())
        ax.set_xlabel('數値 (0–1)', fontsize=10, fontproperties=fp())
        ax.set_ylabel('機率密度', fontsize=10, fontproperties=fp())
        ax.legend(fontsize=9, prop=fp())
        ax.grid(alpha=0.3)
        # 只標偏態係數，讓觀眾直接看到右偏的程度
        ax.text(0.97, 0.95, f'偏態係數 = {sk:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                fontproperties=fp(),
                bbox=dict(boxstyle='round,pad=0.4', fc='#fff9e6', ec='#e8a838', alpha=0.9))

    plt.tight_layout()
    fname = 'X2_p7_右偏特徵直方圖KDE_原聲度語音度器樂度.png'
    plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n已儲存：{fname}')
    print('數字摘要 (X2 第7頁):')
    for s in summaries:
        print('  ', s)


# ========================================================================
# X3  p9  年齡折線圖 + 95% CI 誤差線  (全 6 特徵)
# ========================================================================
def plot_x3_p9_age_ci():
    tmp = df.copy()
    tmp['age_group'] = pd.Categorical(tmp['age_group'], categories=AGE_ORDER, ordered=True)
    stats = (
        tmp.groupby('age_group', observed=True)[AUDIO]
        .agg(['mean','std','count'])
        .reindex(AGE_ORDER)
    )
    Z = 1.96

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('各年齡層音樂特徵平均値 ± 95% 信賴區間',
                 fontsize=14, fontweight='bold', fontproperties=fp())

    summaries = []
    for ax, feat in zip(axes.flatten(), AUDIO):
        means  = stats[feat]['mean'].values.astype(float)
        stds   = stats[feat]['std'].values.astype(float)
        counts = stats[feat]['count'].values.astype(float)
        ci     = np.where(counts > 1, Z * stds / np.sqrt(counts), np.nan)

        valid  = ~np.isnan(means)
        xp     = [AGE_ZH[i] for i in range(len(AGE_ZH)) if valid[i]]
        yp     = means[valid]; cp = ci[valid]
        xi     = list(range(len(xp)))
        if len(yp) == 0:
            ax.set_title(FEATURE_ZH[feat]+' (no data)', fontsize=11); continue

        ax.plot(xi, yp, marker='o', lw=2.5, color='steelblue', ms=7, zorder=3)
        ax.fill_between(xi, yp, alpha=0.10, color='steelblue')
        ax.errorbar(xi, yp, yerr=cp, fmt='none', color='steelblue',
                    capsize=5, capthick=1.8, elinewidth=1.8, zorder=4, label='95% CI')
        for i, (x, v, c) in enumerate(zip(xi, yp, cp)):
            ax.annotate(f'{v:.2f}', (x, v), xytext=(0,10),
                        textcoords='offset points', ha='center', fontsize=8.5, color='#1a3a5c')

        ax.set_xticks(xi)
        ax.set_xticklabels(xp, rotation=30, fontsize=10, fontproperties=fp())
        ax.set_title(FEATURE_ZH[feat], fontsize=13, fontproperties=fp())
        ax.set_xlabel('年齡層', fontsize=10, fontproperties=fp())
        ax.set_ylabel('平均値 (0–1)', fontsize=10, fontproperties=fp())
        ax.set_ylim(max(0, np.nanmin(yp-cp)-0.05), min(1.05, np.nanmax(yp+cp)+0.08))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, prop=fp())
        summaries.append(f'{FEATURE_ZH[feat]}: 平均範圍 [{np.nanmin(yp):.2f}–{np.nanmax(yp):.2f}]  CI幅={np.nanmean(cp):.3f}')

    fig.text(0.5, -0.01,
             'CI 公式： x̄ ± 1.96·(s/√n)，誤差線不重疊表示兩組差異具統計意義',
             ha='center', fontsize=9, color='gray', fontproperties=fp())
    plt.tight_layout()
    fname = 'X3_p9_年齡折線圖含95CI誤差線.png'
    plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n已儲存：{fname}')
    print('數字摘要 (X3 第9頁):')
    for s in summaries:
        print('  ', s)


# ========================================================================
# X4  p10  性別 Genre 比例堆疊柱狀圖
# ========================================================================
def plot_x4_p10_gender_genre_ci():
    if 'gender' not in df.columns or 'fav_genre' not in df.columns:
        print('無 gender / fav_genre 欄位，跳過 X4')
        return

    gender_order = [g for g in ['Male','Female','Non-binary'] if g in df['gender'].values]
    ct  = pd.crosstab(df['gender'], df['fav_genre'])
    ct  = ct.loc[[g for g in gender_order if g in ct.index]]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    ns = ct.sum(axis=1)
    cmap = plt.get_cmap('tab20')
    genres = ct_pct.columns.tolist()
    y_labels = [GENDER_ZH.get(g, g) for g in ct_pct.index]
    n_gender = len(y_labels)

    fig, ax = plt.subplots(figsize=(14, max(4, n_gender * 1.8)))
    fig.suptitle('性別 × 音樂類型占比（堆疊柱狀圖）',
                 fontsize=14, fontweight='bold', fontproperties=fp())

    bottom = np.zeros(n_gender)
    for i, genre in enumerate(genres):
        vals = ct_pct[genre].values / 100
        pcts = vals * 100
        bars = ax.barh(y_labels, pcts, left=bottom * 100,
                       color=cmap(i / max(len(genres), 1)), label=genre, alpha=0.85)
        for bar, pct in zip(bars, pcts):
            if pct >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{pct:.0f}%', ha='center', va='center', fontsize=8)
        bottom += vals

    ax.set_xlabel('百分比 (%)', fontsize=11, fontproperties=fp())
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(fp())
    ax.legend(loc='lower right', bbox_to_anchor=(1.18, 0), fontsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    fig.text(0.5, -0.04,
             '比例信賴區間公式： p̅ ± 1.96·√[p̅(1−p̅)/n]',
             ha='center', fontsize=9, color='gray', fontproperties=fp())
    plt.tight_layout()
    fname = 'X4_p10_性別Genre比例圖含信賴區間.png'
    plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n已儲存：{fname}')
    print('數字摘要 (X4 第10頁):')
    for g in gender_order:
        if g not in ct_pct.index: continue
        top = ct_pct.loc[g].idxmax()
        top_pct = ct_pct.loc[g].max()
        n = int(ns[g])
        p = top_pct / 100
        ci_val = 1.96 * np.sqrt(p*(1-p)/n) * 100
        print(f'  {GENDER_ZH.get(g,g)}: 最高Genre={top} ({top_pct:.1f}%)  95%CI=[{top_pct-ci_val:.1f}%, {top_pct+ci_val:.1f}%]  n={n}')


# ========================================================================
# X5  p11  地區 η² 橫向柱狀圖 (效應量)
# ========================================================================
def plot_x5_p11_eta2_barchart():
    if not os.path.exists(GLOBAL_CSV):
        print('找不到 global_top50.csv，跳過 X5')
        return

    from scipy.stats import kruskal

    gdf = pd.read_csv(GLOBAL_CSV)
    gdf.columns = gdf.columns.str.lower().str.strip().str.replace(' ', '_')
    country_col = next((c for c in ['country','region','market','country_code','code']
                        if c in gdf.columns), None)
    if not country_col:
        print('找不到 country 欄位，跳過 X5'); return

    gdf['continent'] = gdf[country_col].str.upper().str.strip().map(COUNTRY_TO_CONTINENT)
    gdf = gdf[gdf['continent'].notna() & (gdf['continent'] != '全球')]
    feats = [f for f in AUDIO if f in gdf.columns]

    eta2_vals = {}
    H_vals    = {}
    for feat in feats:
        groups = [g[feat].dropna().values for _, g in gdf.groupby('continent')]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2: continue
        H, p = kruskal(*groups)
        N = gdf[feat].notna().sum()
        k = len(groups)
        eta2 = (H - k + 1) / (N - k)
        eta2_vals[feat] = max(eta2, 0)
        H_vals[feat]    = H

    if not eta2_vals:
        print('無法計算 η²，跳過 X5'); return

    feat_labels = [FEATURE_ZH[f] for f in feats if f in eta2_vals]
    eta2_list   = [eta2_vals[f] for f in feats if f in eta2_vals]
    H_list      = [H_vals[f]    for f in feats if f in eta2_vals]
    colors = ['#e74c3c' if v == max(eta2_list) else '#5b9bd5' for v in eta2_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('地理區域對音樂特徵的效應量（η²，Kruskal-Wallis）',
                 fontsize=14, fontweight='bold', fontproperties=fp())

    bars = ax.barh(feat_labels, eta2_list, color=colors, alpha=0.85, height=0.55)
    for bar, v, H in zip(bars, eta2_list, H_list):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'η²={v:.4f}  H={H:.1f}', va='center', fontsize=10,
                fontproperties=fp())

    ax.set_xlabel('η² (效應量)', fontsize=11, fontproperties=fp())
    ax.set_xlim(0, max(eta2_list) * 1.45)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(fp())
        tick.set_fontsize(12)
    ax.axvline(0.01, color='gray', lw=1, linestyle=':', alpha=0.7)
    ax.axvline(0.06, color='gray', lw=1, linestyle=':', alpha=0.7)
    ax.text(0.01, -0.8, '小效應(0.01)', fontsize=8, color='gray', fontproperties=fp())
    ax.text(0.06, -0.8, '中效應(0.06)', fontsize=8, color='gray', fontproperties=fp())
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    fig.text(0.5, -0.05,
             'η² 公式： (H − k + 1) / (N − k)，紅色 = 最強解釋力',
             ha='center', fontsize=9, color='gray', fontproperties=fp())
    plt.tight_layout()
    fname = 'X5_p11_地區效應量_eta2_橫向柱狀圖.png'
    plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n已儲存：{fname}')
    print('數字摘要 (X5 第11頁):')
    sorted_feats = sorted(eta2_vals.items(), key=lambda x: -x[1])
    for feat, v in sorted_feats:
        print(f'  {FEATURE_ZH[feat]}: η²={v:.4f}  H={H_vals[feat]:.2f}  (效應大小: {"large" if v>0.14 else "medium" if v>0.06 else "small"})')


# ── 執行 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== X 系列圖表產生開始 ===')
    plot_x1_p6_main_features()     # X1  第6頁
    plot_x2_p7_skewed_features()   # X2  第7頁
    plot_x3_p9_age_ci()            # X3  第9頁
    plot_x4_p10_gender_genre_ci()  # X4  第10頁
    plot_x5_p11_eta2_barchart()    # X5  第11頁
    print('\n=== 全部完成 —— 圖表存放於 outputs/figures_zh/ ===')
    print('''
檔名說明：
  X1_p6_主要特徵直方圖KDE_舞動度能量情緒.png       → 第6頁  舞動度/能量/情緒正向度
  X2_p7_右偏特徵直方圖KDE_原聲度語音度器樂度.png   → 第7頁  原聲度/語音度/純器樂度 (右偏)
  X3_p9_年齡折線圖含95CI誤差線.png                 → 第9頁  年齡 x 95% 信賴區間
  X4_p10_性別Genre比例圖含信賴區間.png             → 第10頁 性別 x Genre 比例
  X5_p11_地區效應量_eta2_橫向柱狀圖.png            → 第11頁 效應量 eta2
''')
