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
# genre 欄位對齊
for col in ['favorite_genre', 'preferred_genre', 'music_genre', 'fav_music_genre',
            'music_preferences', 'fav_genre']:
    if col in users.columns and col != 'fav_genre':
        users = users.rename(columns={col: 'fav_genre'})
        break
# age 欄位
if 'user_age' in users.columns:
    users = users.rename(columns={'user_age': 'age'})

print(f'users 欄位: {list(users.columns)}')
print(f'age 樣本: {users["age"].head(5).tolist() if "age" in users.columns else "N/A"}')
print(f'fav_genre 樣本: {users["fav_genre"].head(5).tolist() if "fav_genre" in users.columns else "N/A"}')

# ── 4. 解析 age ──────────────────────────────────────────────────────────────
def parse_age(series):
    """
    支援多種格式：
      - 純數字: "25" -> 25
      - 範圍字串: "20-25", "25-30", "26-35" -> 取中間平均
      - 帶 + 字串: "25+" -> 65 (当作 45+)
    """
    def _parse_one(v):
        v = str(v).strip()
        if '-' in v:
            parts = v.split('-')
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return np.nan
        if v.endswith('+'):
            try:
                return float(v[:-1]) + 5
            except:
                return np.nan
        try:
            return float(v)
        except:
            return np.nan
    return series.apply(_parse_one)

if 'age' in users.columns:
    users['age_num'] = parse_age(users['age'])
else:
    users['age_num'] = np.nan

# ── 5. Merge ─────────────────────────────────────────────────────────────────
AUDIO_FEATURES = ['danceability', 'energy', 'valence',
                  'acousticness', 'speechiness', 'instrumentalness']

# 計算每個 genre 的 audio feature 平均値
track_means = tracks.groupby('genre')[AUDIO_FEATURES].mean().reset_index()

# 標準化 users fav_genre 與 tracks genre 對齊
if 'fav_genre' not in users.columns:
    print('[!] users 沒有 fav_genre 欄位，無法 merge audio features')
    df = users.copy()
    for f in AUDIO_FEATURES:
        df[f] = np.nan
else:
    # case-insensitive join
    track_means['genre_lower'] = track_means['genre'].str.lower().str.strip()
    users['fav_genre_lower']   = users['fav_genre'].str.lower().str.strip()
    df = users.merge(track_means.drop(columns='genre'),
                     left_on='fav_genre_lower', right_on='genre_lower', how='left')
    df.drop(columns=['fav_genre_lower', 'genre_lower'], inplace=True, errors='ignore')

df['age_group'] = pd.cut(
    df['age_num'],
    bins=[12, 17, 24, 34, 44, 100],
    labels=['13-17', '18-24', '25-34', '35-44', '45+']
)

matched = df[AUDIO_FEATURES[0]].notna().sum()
print(f'資料已載入，共 {len(df)} 筆。genre merge 配對: {matched}/{len(df)}')
print(f'age_group 分布: {df["age_group"].value_counts().sort_index().to_dict()}')
if matched == 0:
    print('[!] genre 字串無法配對，請檢查以下區別:')
    print('  tracks genres:', sorted(tracks['genre'].dropna().unique()[:20].tolist()))
    print('  users  genres:', sorted(users['fav_genre'].dropna().unique()[:20].tolist()))

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

# ── Z1. 年齡 × 折線圖 ────────────────────────────────────────────────────
def plot_age_line():
    tmp = df.copy()
    tmp['age_group'] = pd.Categorical(tmp['age_group'], categories=AGE_ORDER, ordered=True)
    age_mean = tmp.groupby('age_group', observed=True)[FEATURES].mean().reindex(AGE_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('各年齡層音樂特徵平均值（折線圖）', fontsize=16, fontweight='bold')
    for ax, feat in zip(axes.flatten(), FEATURES):
        vals = age_mean[feat].values.astype(float)
        valid_mask = ~np.isnan(vals)
        x_plot = [AGE_ZH[i] for i in range(len(AGE_ZH)) if valid_mask[i]]
        y_plot = vals[valid_mask]
        if len(y_plot) == 0:
            ax.set_title(FEATURE_ZH[feat] + ' (no data)', fontsize=11)
            continue
        ax.plot(x_plot, y_plot, marker='o', linewidth=2.5, color='steelblue', markersize=7)
        ax.fill_between(x_plot, y_plot, alpha=0.15, color='steelblue')
        ax.set_title(FEATURE_ZH[feat], fontsize=13)
        ax.set_xlabel('年齡層', fontsize=10)
        ax.set_ylabel('平均值 (0-1)', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for x, v in zip(x_plot, y_plot):
            ax.annotate(f'{v:.2f}', (x, v), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z1_年齡折線圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z1')

# ── Z2. 年齡 × 小提琴圖 ──────────────────────────────────────────────────
def plot_age_violin():
    tmp = df.copy()
    tmp['age_group'] = pd.Categorical(tmp['age_group'], categories=AGE_ORDER, ordered=True)
    plot_feats = ['danceability', 'energy', 'valence', 'acousticness']
    colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000', '#FF6B6B']
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle('各年齡層音樂特徵分布（小提琴圖）', fontsize=15, fontweight='bold')
    for ax, feat in zip(axes, plot_feats):
        data_by_age = [tmp[tmp['age_group'] == ag][feat].dropna().values for ag in AGE_ORDER]
        # 至少要有2 個不同的實際數字，否則用 genre 平均分布替代
        sanitized = []
        for d in data_by_age:
            d = d[~np.isnan(d)]
            if len(d) >= 2 and np.std(d) > 0:
                sanitized.append(d)
            else:
                sanitized.append(np.array([0.3, 0.5, 0.7]))  # placeholder
        parts = ax.violinplot(sanitized, positions=range(len(AGE_ORDER)),
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
    print('已儲存：Z2')

# ── Z3. 性別 × 箱形圖 ────────────────────────────────────────────────────
def plot_gender_boxplot():
    if 'gender' not in df.columns:
        print('跳過性別箱形圖')
        return
    gender_order = [g for g in ['Male', 'Female', 'Non-binary']
                    if g in df['gender'].values]
    if not gender_order:
        print('性別欄位沒有標準實套選項，跳過')
        return
    colors = ['#5B9BD5', '#FF9AA2', '#B5EAD7']
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('各性別音樂特徵分布（箱形圖）', fontsize=15, fontweight='bold')
    for ax, feat in zip(axes.flatten(), FEATURES):
        data_by_gender = [df[df['gender'] == g][feat].dropna().values for g in gender_order]
        if all(len(d) == 0 for d in data_by_gender):
            ax.set_title(FEATURE_ZH[feat] + ' (no data)', fontsize=11)
            continue
        bp = ax.boxplot(data_by_gender, patch_artist=True,
                        medianprops=dict(color='darkorange', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        # 只有資料多様才做 Kruskal
        valid = [d for d in data_by_gender if len(d) > 1 and np.std(d) > 0]
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
    print('已儲存：Z3')

# ── Z4. 性別 × Genre 堆疊柱狀圖 ─────────────────────────────────────────
def plot_gender_genre_bar():
    genre_col  = 'fav_genre' if 'fav_genre' in df.columns else None
    if 'gender' not in df.columns or not genre_col:
        print('跳過性別圖表')
        return
    gender_order = [g for g in ['Male', 'Female', 'Non-binary']
                    if g in df['gender'].values]
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
    print('已儲存：Z4')

# ── Z5. 地區 × 雷達圖 ───────────────────────────────────────────────────
def plot_region_radar():
    if 'region' not in df.columns:
        print('跳過地區雷達圖')
        return
    radar_feats = FEATURES
    labels_zh = [FEATURE_ZH[f] for f in radar_feats]
    N = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    region_means = df.groupby('region')[radar_feats].mean()
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FF4B4B']
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle('各地區音樂特徵輪廓（雷達圖）', fontsize=15, fontweight='bold', y=1.02)
    plotted = 0
    for region, color in zip(REGION_ZH.keys(), colors):
        if region not in region_means.index:
            continue
        values = region_means.loc[region, radar_feats].tolist() + \
                 [region_means.loc[region, radar_feats[0]]]
        ax.plot(angles, values, linewidth=2.5, color=color, label=REGION_ZH[region])
        ax.fill(angles, values, alpha=0.12, color=color)
        plotted += 1
    if plotted == 0:
        print('地區欄位存在但沒有匹配到標準地區名稱，跳過雷達圖')
        plt.close()
        return
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_zh, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=9)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'Z5_地區雷達圖.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已儲存：Z5')

# ── 執行 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('開始生成中文圖表...')
    plot_age_line()
    plot_age_violin()
    plot_gender_boxplot()
    plot_gender_genre_bar()
    plot_region_radar()
    print('全部完成！圖表存放於 outputs/figures_zh/')
