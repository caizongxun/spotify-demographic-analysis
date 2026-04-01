"""
06_chinese_charts.py
產生中文版圖表：年齡、性別、地區
圖表類型：折線圖、小提琴圖、箱形圖、堆疊柱狀圖、雷達圖
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

# ── 0. 中文字體設定（Colab 環境）──────────────────────────────────────────
subprocess.run(['apt-get', 'install', '-y', '-q', 'fonts-noto-cjk'], check=False)
fm._load_fontmanager(try_read_cache=False)
cjk_fonts = [f for f in fm.findSystemFonts() if 'NotoSansCJK' in f or 'NotoSerifCJK' in f]
if cjk_fonts:
    plt.rcParams['font.family'] = fm.FontProperties(fname=cjk_fonts[0]).get_name()
else:
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 路徑設定 ─────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE, 'data')
OUT        = os.path.join(BASE, 'outputs', 'figures_zh')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

TRACKS_CSV = os.path.join(DATA_DIR, 'tracks.csv')
USERS_CSV  = os.path.join(DATA_DIR, 'users.csv')

# ── 2. 自動下載 Kaggle 資料 ───────────────────────────────────────────
def download_if_missing():
    if os.path.exists(TRACKS_CSV) and os.path.exists(USERS_CSV):
        print('資料檔案已存在，跳過下載')
        return

    subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)

    # 下載兩個 dataset
    for slug in [
        'maharshipandya/-spotify-tracks-dataset',
        'meeraajayakumar/spotify-user-behavior-dataset',
    ]:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', slug, '-p', DATA_DIR, '--unzip'],
            check=True
        )

    # 印出解壓後所有 csv
    all_csv = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f'解壓後所有檔案: {all_csv}')

    # 先處理 tracks：常見名稱候選清單
    TRACK_NAMES = {'dataset.csv', 'tracks.csv', 'spotify_tracks.csv',
                   'spotify-tracks.csv', 'SpotifyFeatures.csv'}
    if not os.path.exists(TRACKS_CSV):
        for name in all_csv:
            if name in TRACK_NAMES:
                src = os.path.join(DATA_DIR, name)
                os.rename(src, TRACKS_CSV)
                print(f'重命名 {name} -> tracks.csv')
                all_csv.remove(name)
                break

    # 再處理 users：常見名稱候選清單
    USER_NAMES = {'Spotify_Behavior_Dataset.csv', 'users.csv',
                  'spotify_user_behavior_dataset.csv',
                  'Spotify_User_Behavior_Dataset.csv',
                  'spotify-user-behavior-dataset.csv'}
    if not os.path.exists(USERS_CSV):
        matched = False
        for name in all_csv:
            if name in USER_NAMES:
                src = os.path.join(DATA_DIR, name)
                os.rename(src, USERS_CSV)
                print(f'重命名 {name} -> users.csv')
                matched = True
                break
        # 候選清單都沒配對：把剩下第一個不是 tracks.csv 的檔當 users
        if not matched:
            remaining = [f for f in os.listdir(DATA_DIR)
                         if f.endswith('.csv') and f != 'tracks.csv']
            print(f'候選清單未配對，剩餘檔案: {remaining}')
            if remaining:
                src = os.path.join(DATA_DIR, remaining[0])
                os.rename(src, USERS_CSV)
                print(f'自動指定 {remaining[0]} -> users.csv')
            else:
                raise FileNotFoundError(
                    f'\u627e不到 users.csv，請手動模擬資料目錄: '
                    + str(os.listdir(DATA_DIR))
                )

download_if_missing()

# ── 3. 載入資料 ───────────────────────────────────────────────────────────
tracks = pd.read_csv(TRACKS_CSV)
users  = pd.read_csv(USERS_CSV)
df = pd.merge(users, tracks, on='track_id', how='inner')
print(f'資料已載入，共 {len(df)} 筆記錄')

# ── 標籤對照表 ────────────────────────────────────────────────────────────
FEATURE_ZH = {
    'danceability':     '舞動度',
    'energy':           '能量',
    'valence':          '情緒正向度',
    'acousticness':     '原聲度',
    'speechiness':      '語音度',
    'instrumentalness': '純器樂度',
}
FEATURES = list(FEATURE_ZH.keys())
AGE_ORDER = ['13-17', '18-24', '25-34', '35-44', '45+']
AGE_ZH    = ['13-17歲', '18-24歲', '25-34歲', '35-44歲', '45歲以上']
GENDER_ZH = {'Male': '男性', 'Female': '女性', 'Non-binary': '非二元性別'}
REGION_ZH = {'Western': '西方', 'Asia': '亞洲', 'Latin America': '拉丁美洲', 'Africa': '非洲'}

# ── 4. 年齡 × 折線圖 ─────────────────────────────────────────────────────
def plot_age_line():
    df['age_group'] = pd.Categorical(df['age_group'], categories=AGE_ORDER, ordered=True)
    age_mean = df.groupby('age_group')[FEATURES].mean().reindex(AGE_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('各年齡層音樂特徵平均值（折線圖）', fontsize=16, fontweight='bold')
    for ax, feat in zip(axes.flatten(), FEATURES):
        ax.plot(AGE_ZH, age_mean[feat].values, marker='o', linewidth=2.5,
                color='steelblue', markersize=7)
        ax.fill_between(AGE_ZH, age_mean[feat].values, alpha=0.15, color='steelblue')
        ax.set_title(FEATURE_ZH[feat], fontsize=13)
        ax.set_xlabel('年齡層', fontsize=10)
        ax.set_ylabel('平均值 (0-1)', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for i, v in enumerate(age_mean[feat].values):
            ax.annotate(f'{v:.2f}', (AGE_ZH[i], v), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z1_年齡折線圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z1_年齡折線圖.png')

# ── 5. 年齡 × 小提琴圖 ─────────────────────────────────────────────────────
def plot_age_violin():
    df['age_group'] = pd.Categorical(df['age_group'], categories=AGE_ORDER, ordered=True)
    plot_feats = ['danceability', 'energy', 'valence', 'acousticness']
    colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000', '#FF6B6B']
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle('各年齡層音樂特徵分布（小提琴圖）', fontsize=15, fontweight='bold')
    for ax, feat in zip(axes, plot_feats):
        data_by_age = [df[df['age_group'] == ag][feat].dropna().values for ag in AGE_ORDER]
        parts = ax.violinplot(data_by_age, positions=range(len(AGE_ORDER)),
                              showmedians=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)
        ax.set_xticks(range(len(AGE_ORDER)))
        ax.set_xticklabels(AGE_ZH, rotation=30, fontsize=10)
        ax.set_title(FEATURE_ZH[feat], fontsize=13)
        ax.set_ylabel('數值 (0-1)', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z2_年齡小提琴圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z2_年齡小提琴圖.png')

# ── 6. 性別 × 箱形圖 ─────────────────────────────────────────────────────
def plot_gender_boxplot():
    gender_order = ['Male', 'Female', 'Non-binary']
    gender_labels_list = [GENDER_ZH[g] for g in gender_order]
    colors = ['#5B9BD5', '#FF9AA2', '#B5EAD7']
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('各性別音樂特徵分布（箱形圖）', fontsize=15, fontweight='bold')
    for ax, feat in zip(axes.flatten(), FEATURES):
        data_by_gender = [df[df['gender'] == g][feat].dropna().values for g in gender_order]
        bp = ax.boxplot(data_by_gender, patch_artist=True, notch=False,
                        medianprops=dict(color='darkorange', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        stat, p = stats.kruskal(*data_by_gender)
        sig = 'ns' if p > 0.05 else ('*' if p > 0.01 else '**')
        ax.text(0.97, 0.97, f'K-W {sig}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'))
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(gender_labels_list, fontsize=11)
        ax.set_title(FEATURE_ZH[feat], fontsize=13)
        ax.set_ylabel('數值 (0-1)', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z3_性別箱形圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z3_性別箱形圖.png')

# ── 7. 性別 × Genre 堆疊柱狀圖 ───────────────────────────────────────────
def plot_gender_genre_bar():
    gender_order = ['Male', 'Female', 'Non-binary']
    ct = pd.crosstab(df['gender'], df['genre'])
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
                       color=cmap(i / len(genres)), label=genre, alpha=0.85)
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

# ── 8. 地區 × 雷達圖 ─────────────────────────────────────────────────────
def plot_region_radar():
    radar_feats = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness']
    labels_zh = [FEATURE_ZH[f] for f in radar_feats]
    N = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    region_means = df.groupby('region')[radar_feats].mean()
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FF4B4B']
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle('各地區音樂特徵輪廓（雷達圖）', fontsize=15, fontweight='bold', y=1.02)
    for region, color in zip(REGION_ZH.keys(), colors):
        if region not in region_means.index:
            continue
        values = region_means.loc[region, radar_feats].tolist() + \
                 [region_means.loc[region, radar_feats[0]]]
        ax.plot(angles, values, linewidth=2.5, color=color, label=REGION_ZH[region])
        ax.fill(angles, values, alpha=0.12, color=color)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_zh, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=9)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z5_地區雷達圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z5_地區雷達圖.png')

# ── 執行全部 ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('開始生成中文圖表...')
    plot_age_line()
    plot_age_violin()
    plot_gender_boxplot()
    plot_gender_genre_bar()
    plot_region_radar()
    print('全部完成！圖表存放於 outputs/figures_zh/')
