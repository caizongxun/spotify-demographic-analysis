"""
00_data_summary.py
列出專案所有 Kaggle 資料來源、筆數、欄位數
"""

import os
import pandas as pd

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE, 'data')

DATASETS = [
    {
        'name'   : 'Spotify Tracks（音頻特徵）',
        'kaggle' : 'maharshipandya/-spotify-tracks-dataset',
        'file'   : 'tracks.csv',
        '用途'   : '提供 danceability / energy 等 6 項音頻特徵，作為分析主體',
    },
    {
        'name'   : 'Spotify User Behavior（用戶行為）',
        'kaggle' : 'meeraajayakumar/spotify-user-behavior-dataset',
        'file'   : 'users.csv',
        '用途'   : '提供用戶年齡、性別、訂閱方案、喜好 Genre 等人口屬性',
    },
    {
        'name'   : 'Spotify Global Top 50（全球榜單）',
        'kaggle' : 'kevinam/spotify-global-top-50-song-data-oct-18-nov-19',
        'file'   : 'global_top50.csv',
        '用途'   : '提供 72 個國家 Top50 榜單含音頻特徵，用於地區分析（S4–S6）',
    },
]

print('=' * 70)
print('專案資料來源總覽')
print('=' * 70)

total_rows = 0
for ds in DATASETS:
    path = os.path.join(DATA_DIR, ds['file'])
    if not os.path.exists(path):
        print(f"\n{ds['name']}")
        print(f"  Kaggle : {ds['kaggle']}")
        print(f"  狀態   : 檔案不存在（{path}）")
        continue

    df = pd.read_csv(path)
    rows, cols = df.shape
    total_rows += rows

    print(f"\n{ds['name']}")
    print(f"  Kaggle : kaggle.com/datasets/{ds['kaggle']}")
    print(f"  檔案   : {ds['file']}")
    print(f"  筆數   : {rows:,} 筆")
    print(f"  欄位數 : {cols} 欄")
    print(f"  欄位   : {list(df.columns)}")
    print(f"  用途   : {ds['用途']}")

print()
print('=' * 70)
print(f'三個資料集合計：{total_rows:,} 筆')
print('=' * 70)
