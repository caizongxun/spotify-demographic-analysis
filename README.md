# Spotify Demographic Analysis

> 利用 Spotify Tracks Dataset 推論聽眾人口統計特徵（年齡、性別、地區），並以統計學方法解釋音樂特徵與人口統計的對應關係。

## 研究問題

1. **年齡**：老歌（pre-2000）的音樂特徵是否與新歌顯著不同？能以此推論聽眾年齡層？
2. **性別**：音樂特徵（valence, energy）是否存在性別音樂指紋？
3. **地區**：不同地區音樂類型是否導致特徵分佈差異？

## 資料集

- 來源：[Kaggle - Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- 樣本數：~114,000 首歌曲
- 特徵：11 個音頻特徵（danceability, energy, valence, acousticness, loudness, speechiness, instrumentalness, tempo 等）

## 統計方法

| 步驟 | 方法 | 說明 |
|------|------|------|
| 正態性檢定 | Shapiro-Wilk | 判斷特徵分佈，選擇參數或非參數方法 |
| 多群比較 | Kruskal-Wallis | 年齡群/地區間特徵差異 |
| 雙群比較 | Mann-Whitney U | 性別間特徵差異 |
| 效應量 | Cohen's d, η², r | 量化實質差異大小 |
| 預測模型 | 羅吉斯迴歸, Random Forest | 年齡、性別、地區分類 |
| 降維 | PCA | 地區文化群聚視覺化 |

## 專案結構

```
spotify-demographic-analysis/
├── data/
│   └── README.md               # 資料下載說明
├── notebooks/
│   ├── 01_eda.py               # 探索性資料分析 + 圖表
│   └── 02_statistical_tests.py # 假設檢定
├── outputs/
│   └── figures/                # 輸出圖表
├── requirements.txt
└── README.md
```

## 快速開始

```bash
# 安裝套件
pip install -r requirements.txt

# 1. 執行 EDA + 圖表
python notebooks/01_eda.py

# 2. 執行統計檢定
python notebooks/02_statistical_tests.py
```

## 主要發現（初步）

- **年齡**：acousticness 隨年份顯著下降（η²=0.075, H=2984.5, p<0.001）
- **性別**：女性偏好高 valence 音樂（Mann-Whitney U, p<0.001, r=0.12）
- **地區**：僅 acousticness 有顯著差異（H=9.85, p=0.007），其餘特徵全球同質化

## 授權

MIT License
