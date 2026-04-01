# 資料集下載說明

## 方式一：Kaggle CLI

```bash
# 安裝 kaggle CLI
pip install kaggle

# 下載資料集（需先設定 ~/.kaggle/kaggle.json）
kaggle datasets download maharshipandya/-spotify-tracks-dataset
unzip -o spotify-tracks-dataset.zip -d data/
```

## 方式二：手動下載

1. 前往 https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
2. 點擊 "Download" 下載 `dataset.csv`
3. 放置於 `data/dataset.csv`

## 若無法下載

直接執行 `python notebooks/01_eda.py`，程式會自動使用模擬資料（貼近真實 Spotify 分佈）。
