"""
06_chinese_charts.py
產生中文版圖表
圖表清單:
  Z0 音樂特徵分布（Histogram + KDE 中文版）
  Z1 年齡 x 折線圖 + 95% 信賴區間誤差線
  Z2 年齡 x Strip+Box 圖 (小提琴替代)
  Z3 性別 x 箱形圖
  Z4 性別 x Genre 堆疊柱狀圖
  Z5 訂閱方案 x 雷達圖
  Z6 地區 x 音樂特徵雷達圖（kevinam 72 國 Top50 資料集）
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

# ── 0. 中文字體（直接指定 .ttc 路徑，不依賴快取） ──────────────────────────
subprocess.run(['apt-get', 'install', '-y', '-q', 'fonts-noto-cjk'], check=False)

_CJK_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
]
_font_path = next((p for p in _CJK_CANDIDATES if os.path.exists(p)), None)
if _font_path:
    _fp = fm.FontProperties(fname=_font_path)
    plt.rcParams['font.family'] = _fp.get_name()
    fm.fontManager.addfont(_font_path)
    print(f'字體載入成功: {_font_path}  name={_fp.get_name()}')
else:
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    print('警告: 找不到 NotoSansCJK-Regular.ttc，嘗試字體名稱模式')
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 路徑 ─────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE, 'data')
OUT        = os.path.join(BASE, 'outputs', 'figures_zh')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
TRACKS_CSV  = os.path.join(DATA_DIR, 'tracks.csv')
USERS_CSV   = os.path.join(DATA_DIR, 'users.csv')
GLOBAL_CSV  = os.path.join(DATA_DIR, 'global_top50.csv')

# ── 2. 下載 ─────────────────────────────────────────────────────────────────
def _find_file(folder, exts):
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if any(f.lower().endswith(e) for e in exts):
                return os.path.join(root, f)
    return None

def download_if_missing():
    need = [
        ('maharshipandya/-spotify-tracks-dataset',
         os.path.join(DATA_DIR, 'tracks_raw'), TRACKS_CSV),
        ('meeraajayakumar/spotify-user-behavior-dataset',
         os.path.join(DATA_DIR, 'users_raw'), USERS_CSV),
        ('kevinam/spotify-global-top-50-song-data-oct-18-nov-19',
         os.path.join(DATA_DIR, 'global_raw'), GLOBAL_CSV),
    ]
    missing = [(s, d, t) for s, d, t in need if not os.path.exists(t)]
    if not missing:
        print('資料已存在，跳過下載')
        return
    subprocess.run(['pip', 'install', '-q', 'kaggle', 'openpyxl'], check=True)
    for slug, dl_dir, dest in missing:
        os.makedirs(dl_dir, exist_ok=True)
        subprocess.run(['kaggle', 'datasets', 'download', '-d', slug,
                        '-p', dl_dir, '--unzip'], check=True)
        all_files = []
        for root, _, files in os.walk(dl_dir):
            for f in files:
                all_files.append(os.path.relpath(os.path.join(root, f), dl_dir))
        print(f'  解壓後檔案: {all_files}')
        found = _find_file(dl_dir, ['.csv'])
        if found:
            import shutil
            shutil.copy(found, dest)
        else:
            found_x = _find_file(dl_dir, ['.xlsx', '.xls'])
            if found_x:
                pd.read_excel(found_x, engine='openpyxl').to_csv(dest, index=False)
            else:
                raise FileNotFoundError(f'在 {dl_dir} 找不到可用檔案: {all_files}')
        print(f'  完成 -> {dest}')

download_if_missing()

# ── 3. 載入資料 ───────────────────────────────────────────────────────────
tracks = pd.read_csv(TRACKS_CSV)
if 'track_genre' in tracks.columns:
    tracks = tracks.rename(columns={'track_genre': 'genre'})

users = pd.read_csv(USERS_CSV)
users.columns = users.columns.str.lower().str.replace(' ', '_')
for col in ['favorite_genre', 'preferred_genre', 'music_genre', 'fav_music_genre',
            'music_preferences']:
    if col in users.columns:
        users = users.rename(columns={col: 'fav_genre'})
        break
if 'user_age' in users.columns:
    users = users.rename(columns={'user_age': 'age'})

def parse_age(series):
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
    return series.apply(_one)

if 'age' in users.columns:
    users['age_num'] = parse_age(users['age'])
else:
    users['age_num'] = np.nan

# ── 4. Merge audio features ──────────────────────────────────────────────────
AUDIO_FEATURES = ['danceability', 'energy', 'valence',
                  'acousticness', 'speechiness', 'instrumentalness']

track_means = tracks.groupby('genre')[AUDIO_FEATURES].mean().reset_index()
track_means['genre_lower'] = track_means['genre'].str.lower().str.strip()
users['fav_genre_lower'] = users['fav_genre'].str.lower().str.strip() if 'fav_genre' in users.columns else ''
df = users.merge(track_means.drop(columns='genre'),
                 left_on='fav_genre_lower', right_on='genre_lower', how='left')
df.drop(columns=['fav_genre_lower', 'genre_lower'], inplace=True, errors='ignore')

df['age_group'] = pd.cut(
    df['age_num'],
    bins=[12, 17, 24, 34, 44, 100],
    labels=['13-17', '18-24', '25-34', '35-44', '45+']
)

matched = df[AUDIO_FEATURES[0]].notna().sum()
print(f'資料 {len(df)} 筆，genre merge 配對 {matched}/{len(df)}')
print(f'age_group: {df["age_group"].value_counts().sort_index().to_dict()}')

# ── 標籤對照表 ────────────────────────────────────────────────────────────
FEATURE_ZH = {
    'danceability':     '舞動度',
    'energy':           '能量',
    'valence':          '情緒正向度',
    'acousticness':     '原聲度',
    'speechiness':      '語音度',
    'instrumentalness': '純器樂度',
}
FEATURES   = list(FEATURE_ZH.keys())
AGE_ORDER  = ['13-17', '18-24', '25-34', '35-44', '45+']
AGE_ZH     = ['13-17歲', '18-24歲', '25-34歲', '35-44歲', '45歲以上']
GENDER_ZH  = {'Male': '男性', 'Female': '女性', 'Non-binary': '非二元性別'}
AGE_COLORS = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#b07aa1']

# 國家代碼 → 洲別對照
COUNTRY_TO_CONTINENT = {
    'US': '北美洲', 'CA': '北美洲', 'MX': '北美洲',
    'GB': '歐洲', 'DE': '歐洲', 'FR': '歐洲', 'IT': '歐洲',
    'ES': '歐洲', 'NL': '歐洲', 'SE': '歐洲', 'NO': '歐洲',
    'DK': '歐洲', 'FI': '歐洲', 'PL': '歐洲', 'BE': '歐洲',
    'CH': '歐洲', 'AT': '歐洲', 'PT': '歐洲', 'IE': '歐洲',
    'CZ': '歐洲', 'HU': '歐洲', 'RO': '歐洲', 'GR': '歐洲',
    'TR': '歐洲', 'SK': '歐洲', 'HR': '歐洲',
    'BR': '南美洲', 'AR': '南美洲', 'CL': '南美洲', 'CO': '南美洲',
    'PE': '南美洲', 'VE': '南美洲', 'EC': '南美洲', 'BO': '南美洲',
    'UY': '南美洲', 'PY': '南美洲',
    'JP': '亞洲', 'KR': '亞洲', 'TW': '亞洲', 'HK': '亞洲',
    'SG': '亞洲', 'MY': '亞洲', 'TH': '亞洲', 'PH': '亞洲',
    'ID': '亞洲', 'VN': '亞洲', 'IN': '亞洲', 'PK': '亞洲',
    'IL': '亞洲',
    'AU': '大洋洲', 'NZ': '大洋洲',
    'ZA': '非洲', 'NG': '非洲', 'EG': '非洲', 'MA': '非洲',
    'GT': '中美洲', 'SV': '中美洲', 'HN': '中美洲',
    'CR': '中美洲', 'PA': '中美洲', 'DO': '中美洲',
    'global': '全球',
}


# ── 共用：設字體到 figure ────────────────────────────────────────────────
def _set_font(fig):
    if _font_path:
        fp = fm.FontProperties(fname=_font_path)
        for ax in fig.get_axes():
            for text in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
                text.set_fontproperties(fp)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontproperties(fp)


# ── Z0. 音樂特徵分布 ──────────────────────────────────────────────────────
def plot_feature_distribution():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('音樂特徵分布（直方圖 + KDE）', fontsize=15, fontweight='bold',
                 fontproperties=fm.FontProperties(fname=_font_path) if _font_path else None)
    for ax, feat in zip(axes.flat, FEATURES):
        data = tracks[feat].dropna()
        ax.hist(data, bins=50, color='#4c8bbe', alpha=0.5, density=True)
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 300)
        ax.plot(xs, kde(xs), color='#1a5276', lw=2)
        ax.axvline(data.mean(), color='red', lw=1.5, linestyle='--',
                   label=f'平均={data.mean():.2f}')
        fp = fm.FontProperties(fname=_font_path) if _font_path else None
        ax.set_title(FEATURE_ZH[feat], fontsize=13, fontproperties=fp)
        ax.set_xlabel('數値 (0-1)', fontsize=10, fontproperties=fp)
        ax.set_ylabel('機率密度', fontsize=10, fontproperties=fp)
        ax.legend(fontsize=9, prop=fp)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z0_音樂特徵分布.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z0_音樂特徵分布.png')


# ── Z1. 年齡 x 折線圖 + 95% 信賴區間誤差線 ──────────────────────────────
def plot_age_line():
    """
    折線圖加入 95% 信賴區間誤差線
    公式：CI = x̄ ± 1.96 * (s / √n)
    誤差線不重疊 → 兩組平均值差異具統計意義
    """
    tmp = df.copy()
    tmp['age_group'] = pd.Categorical(tmp['age_group'], categories=AGE_ORDER, ordered=True)

    # 計算每個年齡層每個特徵的 mean / std / n / CI
    stats_df = (
        tmp.groupby('age_group', observed=True)[FEATURES]
        .agg(['mean', 'std', 'count'])
        .reindex(AGE_ORDER)
    )

    fp = fm.FontProperties(fname=_font_path) if _font_path else None
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('各年齡層音樂特徵平均值 ± 95% 信賴區間', fontsize=15,
                 fontweight='bold', fontproperties=fp)

    Z = 1.96  # 95% 信賴水準對應的 z 值

    for ax, feat in zip(axes.flatten(), FEATURES):
        means  = stats_df[feat]['mean'].values.astype(float)
        stds   = stats_df[feat]['std'].values.astype(float)
        counts = stats_df[feat]['count'].values.astype(float)

        # CI = 1.96 * s / √n
        ci = np.where(counts > 1, Z * stds / np.sqrt(counts), np.nan)

        valid = ~np.isnan(means)
        x_p   = [AGE_ZH[i] for i in range(len(AGE_ZH)) if valid[i]]
        y_p   = means[valid]
        ci_p  = ci[valid]
        x_idx = list(range(len(x_p)))

        if len(y_p) == 0:
            ax.set_title(FEATURE_ZH[feat] + ' (no data)', fontsize=11)
            continue

        # 折線
        ax.plot(x_idx, y_p, marker='o', linewidth=2.5,
                color='steelblue', markersize=7, zorder=3)
        ax.fill_between(x_idx, y_p, alpha=0.10, color='steelblue')

        # 誤差線（95% CI）
        ax.errorbar(x_idx, y_p, yerr=ci_p,
                    fmt='none', color='steelblue', capsize=5,
                    capthick=1.8, elinewidth=1.8, zorder=4,
                    label='95% CI')

        # 標示平均值
        for xi, v, c in zip(x_idx, y_p, ci_p):
            offset = c + 0.025 if not np.isnan(c) else 0.025
            ax.annotate(f'{v:.2f}', (xi, v),
                        textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=8.5,
                        color='#1a3a5c')

        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_p, rotation=30, fontsize=10)
        if fp:
            for tick in ax.get_xticklabels():
                tick.set_fontproperties(fp)

        ax.set_title(FEATURE_ZH[feat], fontsize=13, fontproperties=fp)
        ax.set_xlabel('年齡層', fontsize=10, fontproperties=fp)
        ax.set_ylabel('平均值 (0-1)', fontsize=10, fontproperties=fp)
        ax.set_ylim(max(0, np.nanmin(y_p - ci_p) - 0.05),
                    min(1.05, np.nanmax(y_p + ci_p) + 0.08))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, prop=fp)

    # 圖底說明
    fig.text(0.5, -0.01,
             '誤差線 = 95% 信賴區間（CI = x̄ ± 1.96·s/√n）  '
             '誤差線不重疊表示兩組差異具統計意義',
             ha='center', fontsize=9, color='gray', fontproperties=fp)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z1_年齡折線圖_CI.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z1_年齡折線圖_CI.png')


# ── Z2. 年齡 x Strip + Box 圖 ────────────────────────────────────────────
def plot_age_strip():
    tmp = df.copy()
    tmp['age_group'] = pd.Categorical(tmp['age_group'], categories=AGE_ORDER, ordered=True)
    plot_feats = ['danceability', 'energy', 'valence', 'acousticness']
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fp = fm.FontProperties(fname=_font_path) if _font_path else None
    fig.suptitle('各年齡層音樂特徵分布（抒點 + 箱形圖）', fontsize=15, fontweight='bold',
                 fontproperties=fp)
    np.random.seed(42)
    for ax, feat in zip(axes, plot_feats):
        for i, ag in enumerate(AGE_ORDER):
            sub = tmp[tmp['age_group'] == ag][feat].dropna().values
            if len(sub) == 0:
                continue
            jitter = np.random.uniform(-0.18, 0.18, len(sub))
            ax.scatter(np.full(len(sub), i) + jitter, sub,
                       alpha=0.45, s=18, color=AGE_COLORS[i], zorder=2)
            q1, med, q3 = np.percentile(sub, [25, 50, 75])
            ax.plot([i - 0.2, i + 0.2], [med, med], color='black', lw=2.5, zorder=3)
            rect = plt.Rectangle((i - 0.2, q1), 0.4, q3 - q1,
                                  fill=False, edgecolor='black', lw=1.5, zorder=3)
            ax.add_patch(rect)
            ax.scatter([i], [sub.mean()], marker='D', s=40,
                       color='white', edgecolor='black', lw=1.5, zorder=4)
        ax.set_xticks(range(len(AGE_ORDER)))
        ax.set_xticklabels(AGE_ZH, rotation=30, fontsize=10,
                           fontproperties=fp)
        ax.set_title(FEATURE_ZH[feat], fontsize=13, fontproperties=fp)
        ax.set_ylabel('數值 (0-1)', fontsize=10, fontproperties=fp)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=7, label='個別資料點'),
        Line2D([0], [0], color='black', lw=2.5, label='中位數'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=7, label='平均値'),
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=9, prop=fp)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z2_年齡抒點箱形圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z2_年齡抒點箱形圖.png')


# ── Z3. 性別 x 箱形圖 ─────────────────────────────────────────────────────
def plot_gender_boxplot():
    if 'gender' not in df.columns:
        print('跳過性別箱形圖')
        return
    gender_order = [g for g in ['Male', 'Female', 'Non-binary'] if g in df['gender'].values]
    if not gender_order:
        print('性別欄位沒有匹配選項，跳過')
        return
    colors = ['#5B9BD5', '#FF9AA2', '#B5EAD7']
    fp = fm.FontProperties(fname=_font_path) if _font_path else None
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('各性別音樂特徵分布（箱形圖）', fontsize=15, fontweight='bold',
                 fontproperties=fp)
    for ax, feat in zip(axes.flatten(), FEATURES):
        data_by_g = [df[df['gender'] == g][feat].dropna().values for g in gender_order]
        if all(len(d) == 0 for d in data_by_g):
            ax.set_title(FEATURE_ZH[feat] + ' (no data)', fontsize=11)
            continue
        bp = ax.boxplot(data_by_g, patch_artist=True,
                        medianprops=dict(color='darkorange', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        valid = [d for d in data_by_g if len(d) > 1 and np.std(d) > 0]
        if len(valid) >= 2:
            try:
                from scipy.stats import kruskal
                _, p = kruskal(*valid)
                sig = 'ns' if p > 0.05 else ('*' if p > 0.01 else '**')
                ax.text(0.97, 0.97, f'K-W {sig}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'))
            except ValueError:
                pass
        ax.set_xticks(range(1, len(gender_order) + 1))
        ax.set_xticklabels([GENDER_ZH.get(g, g) for g in gender_order], fontsize=11,
                           fontproperties=fp)
        ax.set_title(FEATURE_ZH[feat], fontsize=13, fontproperties=fp)
        ax.set_ylabel('數值 (0-1)', fontsize=10, fontproperties=fp)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z3_性別箱形圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z3_性別箱形圖.png')


# ── Z4. 性別 x Genre 堆疊柱狀圖 ─────────────────────────────────────────
def plot_gender_genre_bar():
    genre_col = 'fav_genre' if 'fav_genre' in df.columns else None
    if 'gender' not in df.columns or not genre_col:
        print('跳過性別圖表')
        return
    gender_order = [g for g in ['Male', 'Female', 'Non-binary'] if g in df['gender'].values]
    ct = pd.crosstab(df['gender'], df[genre_col])
    ct = ct.loc[[g for g in gender_order if g in ct.index]]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fp = fm.FontProperties(fname=_font_path) if _font_path else None
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle('各性別音樂類型偏好（堆疊柱狀圖）', fontsize=15, fontweight='bold',
                 fontproperties=fp)
    bottom = np.zeros(len(ct_pct))
    cmap = plt.get_cmap('tab20')
    genres = ct_pct.columns.tolist()
    y_labels = [GENDER_ZH.get(g, g) for g in ct_pct.index]
    for i, genre in enumerate(genres):
        vals = ct_pct[genre].values
        bars = ax.barh(y_labels, vals, left=bottom,
                       color=cmap(i / max(len(genres), 1)), label=genre, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.0f}%', ha='center', va='center', fontsize=8.5)
        bottom += vals
    ax.set_xlabel('百分比 (%)', fontsize=11, fontproperties=fp)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(fp)
    ax.legend(loc='lower right', bbox_to_anchor=(1.18, 0), fontsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z4_性別Genre堆疊柱狀圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z4_性別Genre堆疊柱狀圖.png')


# ── Z5. 訂閱方案 x 雷達圖 ────────────────────────────────────────────────
def plot_subscription_radar():
    group_col = None
    for candidate in ['spotify_subscription_plan', 'spotify_usage_period',
                      'spotify_listening_device']:
        if candidate in df.columns:
            group_col = candidate
            break
    if not group_col:
        print('找不到適合雷達圖的分組欄位，跳過')
        return
    groups = df[group_col].dropna().unique().tolist()[:6]
    radar_feats = FEATURES
    labels_zh = [FEATURE_ZH[f] for f in radar_feats]
    N = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    group_means = df.groupby(group_col)[radar_feats].mean()
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FF4B4B', '#9B59B6', '#1ABC9C']
    col_zh = {
        'spotify_subscription_plan': '訂閱方案',
        'spotify_usage_period':      '使用時間',
        'spotify_listening_device':  '載具類型',
    }
    title_label = col_zh.get(group_col, group_col)
    fp = fm.FontProperties(fname=_font_path) if _font_path else None
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle(f'各{title_label}音樂特徵輪廓（雷達圖）',
                 fontsize=15, fontweight='bold', y=1.02, fontproperties=fp)
    plotted = 0
    for group, color in zip(groups, colors):
        if group not in group_means.index:
            continue
        row = group_means.loc[group, radar_feats]
        if row.isna().all():
            continue
        values = row.tolist() + [row.iloc[0]]
        ax.plot(angles, values, linewidth=2.5, color=color, label=str(group))
        ax.fill(angles, values, alpha=0.12, color=color)
        plotted += 1
    if plotted == 0:
        print('雷達圖沒有有效分組，跳過')
        plt.close()
        return
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_zh, fontsize=12,
                      fontproperties=fp)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=9)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'Z5_{title_label}雷達圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已儲存：Z5_{title_label}雷達圖.png')


# ── Z6. 地區 x 音樂特徵雷達圖（kevinam 72 國 Top50） ────────────────────
def plot_region_radar():
    if not os.path.exists(GLOBAL_CSV):
        print('找不到 global_top50.csv，跳過 Z6')
        return

    gdf = pd.read_csv(GLOBAL_CSV)
    print(f'global_top50 欄位: {list(gdf.columns[:15])}')
    gdf.columns = gdf.columns.str.lower().str.strip().str.replace(' ', '_')

    country_col = None
    for c in ['country', 'region', 'market', 'country_code', 'code']:
        if c in gdf.columns:
            country_col = c
            break
    if not country_col:
        for c in gdf.columns:
            if gdf[c].dtype == object and gdf[c].str.len().median() <= 4:
                country_col = c
                print(f'  猜測 country 欄位: {country_col}')
                break
    if not country_col:
        print(f'找不到國家欄位，現有欄位: {list(gdf.columns)}')
        return

    radar_feats = [f for f in FEATURES if f in gdf.columns]
    if len(radar_feats) < 3:
        print(f'音頻特徵欄位不足: {radar_feats}，跳過 Z6')
        return

    gdf['continent'] = gdf[country_col].str.upper().str.strip().map(COUNTRY_TO_CONTINENT)
    unmapped = gdf[gdf['continent'].isna()][country_col].unique()
    if len(unmapped) > 0:
        print(f'  未映射國家代碼（前10）: {unmapped[:10]}')
    gdf = gdf[gdf['continent'].notna()]

    continent_means = gdf.groupby('continent')[radar_feats].mean()
    continents = [c for c in continent_means.index.tolist() if c != '全球']
    if len(continents) < 2:
        print(f'洲別分組不足 ({continents})，跳過 Z6')
        return

    labels_zh = [FEATURE_ZH.get(f, f) for f in radar_feats]
    N = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    CONTINENT_COLORS = {
        '北美洲': '#4472C4',
        '歐洲':   '#ED7D31',
        '南美洲': '#70AD47',
        '亞洲':   '#FF4B4B',
        '大洋洲': '#9B59B6',
        '非洲':   '#1ABC9C',
        '中美洲': '#F39C12',
    }

    fp = fm.FontProperties(fname=_font_path) if _font_path else None
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle('各洲別 Spotify Top50 音樂特徵輪廓（雷達圖）',
                 fontsize=15, fontweight='bold', y=1.02, fontproperties=fp)

    for cont in continents:
        if cont not in continent_means.index:
            continue
        row = continent_means.loc[cont, radar_feats]
        if row.isna().all():
            continue
        values = row.tolist() + [row.iloc[0]]
        color = CONTINENT_COLORS.get(cont, '#888888')
        ax.plot(angles, values, linewidth=2.5, color=color, label=cont)
        ax.fill(angles, values, alpha=0.10, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels_zh, fontsize=12,
                      fontproperties=fp)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=9)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.18), fontsize=11,
              prop=fp)

    country_counts = gdf.groupby('continent')[country_col].nunique()
    note = '  '.join([f'{c}({country_counts.get(c,0)}國)' for c in continents])
    fig.text(0.5, -0.02, note, ha='center', fontsize=9, color='gray',
             fontproperties=fp)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z6_地區音樂特徵雷達圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z6_地區音樂特徵雷達圖.png')


# ── 執行 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('開始生成中文圖表...')
    plot_feature_distribution()   # Z0
    plot_age_line()               # Z1（含 95% CI 誤差線）
    plot_age_strip()              # Z2
    plot_gender_boxplot()         # Z3
    plot_gender_genre_bar()       # Z4
    plot_subscription_radar()     # Z5
    plot_region_radar()           # Z6
    print('全部完成！圖表存放於 outputs/figures_zh/')
