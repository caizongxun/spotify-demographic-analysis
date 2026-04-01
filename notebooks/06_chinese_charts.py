"""
06_chinese_charts.py
產生中文版圖表
圖表清單:
  Z0 音樂特徵分布（Histogram + KDE 中文版）
  Z1 年齡 x 折線圖
  Z2 年齡 x Strip+Box 圖 (小提琴替代)
  Z3 性別 x 箱形圖
  Z4 性別 x Genre 堆疊柱狀圖
  Z5 訂閱方案 x 雷達圖 (地區替代)
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

# ── 0. 中文字體 ──────────────────────────────────────────────────────────────
subprocess.run(['apt-get', 'install', '-y', '-q', 'fonts-noto-cjk'], check=False)
fm._load_fontmanager(try_read_cache=False)
cjk_fonts = [f for f in fm.findSystemFonts() if 'NotoSansCJK' in f or 'NotoSerifCJK' in f]
if cjk_fonts:
    plt.rcParams['font.family'] = fm.FontProperties(fname=cjk_fonts[0]).get_name()
else:
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 路徑 ─────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE, 'data')
OUT        = os.path.join(BASE, 'outputs', 'figures_zh')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
TRACKS_CSV = os.path.join(DATA_DIR, 'tracks.csv')
USERS_CSV  = os.path.join(DATA_DIR, 'users.csv')

# ── 2. 下載 ─────────────────────────────────────────────────────────────────
def _find_file(folder, exts):
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if any(f.lower().endswith(e) for e in exts):
                return os.path.join(root, f)
    return None

def download_if_missing():
    if os.path.exists(TRACKS_CSV) and os.path.exists(USERS_CSV):
        print('資料已存在，跳過下載')
        return
    subprocess.run(['pip', 'install', '-q', 'kaggle', 'openpyxl'], check=True)
    for slug, dl_dir, dest in [
        ('maharshipandya/-spotify-tracks-dataset',
         os.path.join(DATA_DIR, 'tracks_raw'), TRACKS_CSV),
        ('meeraajayakumar/spotify-user-behavior-dataset',
         os.path.join(DATA_DIR, 'users_raw'),  USERS_CSV),
    ]:
        if os.path.exists(dest):
            continue
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
            os.rename(found, dest)
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
print(f'users 欄位: {list(users.columns)}')

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


# ── Z0. 音樂特徵分布（Histogram + KDE 中文版） ───────────────────────────────
def plot_feature_distribution():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('音樂特徵分布（直方圖 + KDE）', fontsize=15, fontweight='bold')
    for ax, feat in zip(axes.flat, FEATURES):
        data = tracks[feat].dropna()
        ax.hist(data, bins=50, color='#4c8bbe', alpha=0.5, density=True)
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 300)
        ax.plot(xs, kde(xs), color='#1a5276', lw=2)
        ax.axvline(data.mean(), color='red', lw=1.5, linestyle='--',
                   label=f'平均={data.mean():.2f}')
        ax.set_title(FEATURE_ZH[feat], fontsize=13)
        ax.set_xlabel('數値 (0-1)', fontsize=10)
        ax.set_ylabel('機率密度', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z0_音樂特徵分布.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z0_音樂特徵分布.png')


# ── Z1. 年齡 x 折線圖 ────────────────────────────────────────────────────────
def plot_age_line():
    tmp = df.copy()
    tmp['age_group'] = pd.Categorical(tmp['age_group'], categories=AGE_ORDER, ordered=True)
    age_mean = tmp.groupby('age_group', observed=True)[FEATURES].mean().reindex(AGE_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('各年齡層音樂特徵平均值（折線圖）', fontsize=16, fontweight='bold')
    for ax, feat in zip(axes.flatten(), FEATURES):
        vals = age_mean[feat].values.astype(float)
        valid = ~np.isnan(vals)
        x_p = [AGE_ZH[i] for i in range(len(AGE_ZH)) if valid[i]]
        y_p = vals[valid]
        if len(y_p) == 0:
            ax.set_title(FEATURE_ZH[feat] + ' (no data)', fontsize=11)
            continue
        ax.plot(x_p, y_p, marker='o', linewidth=2.5, color='steelblue', markersize=7)
        ax.fill_between(x_p, y_p, alpha=0.15, color='steelblue')
        ax.set_title(FEATURE_ZH[feat], fontsize=13)
        ax.set_xlabel('年齡層', fontsize=10)
        ax.set_ylabel('平均值 (0-1)', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for x, v in zip(x_p, y_p):
            ax.annotate(f'{v:.2f}', (x, v), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z1_年齡折線圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z1_年齡折線圖.png')


# ── Z2. 年齡 x Strip + Box 圖（小提琴圖替代） ───────────────────────────────
def plot_age_strip():
    """
    不用 seaborn，純 matplotlib 實作:
      - 第一層：简化算法抄點圖 (strip)
      - 第二層：箱形圖覆蓋，顯示 IQR 與中位數
    """
    tmp = df.copy()
    tmp['age_group'] = pd.Categorical(tmp['age_group'], categories=AGE_ORDER, ordered=True)
    plot_feats = ['danceability', 'energy', 'valence', 'acousticness']

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle('各年齡層音樂特徵分布（抒點 + 箱形圖）', fontsize=15, fontweight='bold')

    np.random.seed(42)
    for ax, feat in zip(axes, plot_feats):
        for i, ag in enumerate(AGE_ORDER):
            sub = tmp[tmp['age_group'] == ag][feat].dropna().values
            if len(sub) == 0:
                continue
            # jitter
            jitter = np.random.uniform(-0.18, 0.18, len(sub))
            ax.scatter(np.full(len(sub), i) + jitter, sub,
                       alpha=0.45, s=18, color=AGE_COLORS[i], zorder=2)
            # box
            q1, med, q3 = np.percentile(sub, [25, 50, 75])
            ax.plot([i - 0.2, i + 0.2], [med, med], color='black', lw=2.5, zorder=3)
            rect = plt.Rectangle((i - 0.2, q1), 0.4, q3 - q1,
                                  fill=False, edgecolor='black', lw=1.5, zorder=3)
            ax.add_patch(rect)
            # mean dot
            ax.scatter([i], [sub.mean()], marker='D', s=40,
                       color='white', edgecolor='black', lw=1.5, zorder=4)

        ax.set_xticks(range(len(AGE_ORDER)))
        ax.set_xticklabels(AGE_ZH, rotation=30, fontsize=10)
        ax.set_title(FEATURE_ZH[feat], fontsize=13)
        ax.set_ylabel('數值 (0-1)', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # 圖例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=7, label='個別資料點'),
        Line2D([0], [0], color='black', lw=2.5, label='中位數'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=7, label='平均値'),
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=9)

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
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('各性別音樂特徵分布（箱形圖）', fontsize=15, fontweight='bold')
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
                _, p = stats.kruskal(*valid)
                sig = 'ns' if p > 0.05 else ('*' if p > 0.01 else '**')
                ax.text(0.97, 0.97, f'K-W {sig}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'))
            except ValueError:
                pass
        ax.set_xticks(range(1, len(gender_order) + 1))
        ax.set_xticklabels([GENDER_ZH.get(g, g) for g in gender_order], fontsize=11)
        ax.set_title(FEATURE_ZH[feat], fontsize=13)
        ax.set_ylabel('數值 (0-1)', fontsize=10)
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
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle('各性別音樂類型偏好（堆疊柱狀圖）', fontsize=15, fontweight='bold')
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
    ax.set_xlabel('百分比 (%)', fontsize=11)
    ax.legend(loc='lower right', bbox_to_anchor=(1.18, 0), fontsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z4_性別Genre堆疊柱狀圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z4_性別Genre堆疊柱狀圖.png')


# ── Z5. 訂閱方案 x 雷達圖（地區替代） ──────────────────────────────────
def plot_subscription_radar():
    """
    users 沒有 region，用 spotify_subscription_plan 作為分組發雷達圖。
    若欄位不存在，就用 spotify_usage_period。
    """
    group_col = None
    for candidate in ['spotify_subscription_plan', 'spotify_usage_period',
                      'spotify_listening_device']:
        if candidate in df.columns:
            group_col = candidate
            break
    if not group_col:
        print('找不到適合雷達圖的分組欄位，跳過')
        return

    groups = df[group_col].dropna().unique().tolist()
    # 限制最多 6 組避免色彩太擠
    groups = groups[:6]

    radar_feats = FEATURES
    labels_zh = [FEATURE_ZH[f] for f in radar_feats]
    N = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    group_means = df.groupby(group_col)[radar_feats].mean()
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FF4B4B', '#9B59B6', '#1ABC9C']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    col_zh = {
        'spotify_subscription_plan': '訂閱方案',
        'spotify_usage_period':      '使用時間',
        'spotify_listening_device':  '載具類型',
    }
    title_label = col_zh.get(group_col, group_col)
    fig.suptitle(f'各{title_label}音樂特徵輪廓（雷達圖）',
                 fontsize=15, fontweight='bold', y=1.02)

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

    ax.set_thetagrids(np.degrees(angles[:-1]), labels_zh, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=9)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'Z5_{title_label}雷達圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已儲存：Z5_{title_label}雷達圖.png')


# ── 執行 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('開始生成中文圖表...')
    plot_feature_distribution()   # Z0
    plot_age_line()               # Z1
    plot_age_strip()              # Z2 抒點+箱形圖
    plot_gender_boxplot()         # Z3
    plot_gender_genre_bar()       # Z4
    plot_subscription_radar()     # Z5 訂閱方案雷達圖
    print('全部完成！圖表存放於 outputs/figures_zh/')
