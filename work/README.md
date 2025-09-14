# 重構後的 Bitcoin 價格預測專案

## 目錄結構

```
work/
├── __init__.py
├── models.py              # 統一的模型定義 (所有模型)
├── config.py              # 共用配置
├── compat.py              # 相容性修復
├── data_loader.py         # 資料載入
├── features.py            # 特徵工程
├── run_all.py             # 主程式 (執行所有任務)
├── regression/            # 回歸任務
│   ├── __init__.py
│   ├── main.py            # Transformer 回歸
│   ├── main_gru.py        # GRU 回歸
│   ├── train.py           # 回歸訓練
│   ├── evaluate.py        # 回歸評估
│   └── compare.py         # 回歸比較
└── classification/        # 分類任務
    ├── __init__.py
    ├── main_classification.py      # 分類主程式
    ├── classification_data.py      # 分類資料處理
    ├── train_classification.py     # 分類訓練
    └── compare_classification.py   # 分類比較
```

## 模型定義

所有模型都定義在 `models.py` 中：

### 回歸模型
- `BTC_Transformer`: Transformer 回歸模型
- `GRUSeq2Seq`: GRU 回歸模型

### 分類模型
- `TransformerClassifier`: Transformer 分類模型
- `GRUClassifier`: GRU 分類模型

### 共用組件
- `SineActivation`: 正弦激活層 (用於位置編碼)

## 使用方法

### 1. 執行所有任務
```bash
cd work
python run_all.py
```

### 2. 執行回歸任務比較
```bash
cd work
python regression/compare.py
```

### 3. 執行分類任務比較
```bash
cd work
python classification/compare_classification.py
```

### 4. 執行單一模型
```bash
# Transformer 回歸
cd work
python regression/main.py

# GRU 回歸
cd work
python regression/main_gru.py

# 分類任務
cd work
python classification/main_classification.py
```

## 重構優勢

1. **清晰的任務分離**: 回歸和分類任務分別在獨立目錄中
2. **統一的模型管理**: 所有模型定義在一個檔案中，便於維護
3. **共用的基礎設施**: 資料載入、特徵工程等共用功能
4. **靈活的執行方式**: 支援套件模式和直接執行模式
5. **易於擴展**: 新增模型或任務類型時結構清晰

## 依賴關係

- 回歸和分類任務都依賴於共用的基礎模組 (`config.py`, `data_loader.py`, `features.py`)
- 所有模型都定義在 `models.py` 中
- 每個任務目錄包含該任務特定的訓練、評估和比較邏輯
