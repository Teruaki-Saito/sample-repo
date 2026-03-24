# Iris Dataset EDA & 予測モデル レポート

## 1. データ概要

| 項目 | 内容 |
|------|------|
| サンプル数 | 150件 |
| 特徴量数 | 4 |
| クラス数 | 3（setosa / versicolor / virginica） |
| 各クラスのサンプル数 | 各50件（均等） |

**特徴量一覧：**
- `sepal length (cm)` — がく片の長さ
- `sepal width (cm)` — がく片の幅
- `petal length (cm)` — 花びらの長さ
- `petal width (cm)` — 花びらの幅

---

## 2. EDA 結果

### 基本統計量

| 特徴量 | 平均 | 標準偏差 | 最小 | 最大 |
|--------|------|----------|------|------|
| sepal length | 5.84 | 0.83 | 4.30 | 7.90 |
| sepal width  | 3.06 | 0.44 | 2.00 | 4.40 |
| petal length | 3.76 | 1.77 | 1.00 | 6.90 |
| petal width  | 1.20 | 0.76 | 0.10 | 2.50 |

### 特徴量間の相関（上位3ペア）

| 特徴量ペア | 相関係数 |
|-----------|---------|
| petal length × petal width | **0.96**（強い正の相関） |
| sepal length × petal length | **0.87**（強い正の相関） |
| sepal length × petal width | **0.82**（強い正の相関） |

> `sepal width` は他の特徴量との相関が低く（最大 -0.37）、独立した情報を持つ。

### クラス別特徴量平均

| クラス | sepal length | sepal width | petal length | petal width |
|--------|-------------|-------------|--------------|-------------|
| setosa     | 5.01 | 3.43 | 1.46 | 0.25 |
| versicolor | 5.94 | 2.77 | 4.26 | 1.33 |
| virginica  | 6.59 | 2.97 | 5.55 | 2.03 |

> `setosa` は petal length/width が他クラスと明確に分離。
> `versicolor` と `virginica` は sepal 系で重なりがあるため分類が難しい傾向。

---

## 3. 予測モデル（RandomForest）

### モデル設定

| パラメータ | 値 |
|-----------|----|
| アルゴリズム | RandomForestClassifier |
| 木の本数 | 100 |
| 学習/テスト分割 | 80% / 20% |
| 前処理 | StandardScaler |

### 評価結果

| 指標 | 値 |
|------|----|
| テスト精度 | **90.0%** |
| 5-fold 交差検証（平均） | **96.7%** ± 2.1% |

### クラス別精度

| クラス | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| setosa     | 1.00 | 1.00 | **1.00** |
| versicolor | 0.82 | 0.90 | 0.86 |
| virginica  | 0.89 | 0.80 | 0.84 |

### 混同行列

|  | 予測: setosa | 予測: versicolor | 予測: virginica |
|--|------------|----------------|---------------|
| 実: setosa     | **10** | 0 | 0 |
| 実: versicolor | 0 | **9** | 1 |
| 実: virginica  | 0 | 2 | **8** |

> `versicolor` と `virginica` の間で3件の誤分類が発生。EDA通り、この2クラス間の境界が曖昧。

### 特徴量重要度

| 特徴量 | 重要度 |
|--------|--------|
| petal width (cm)  | 0.437 ████████████████ |
| petal length (cm) | 0.432 ████████████████ |
| sepal length (cm) | 0.116 ████ |
| sepal width (cm)  | 0.015 ▏ |

> petal 系2特徴量で重要度の **86.9%** を占める。sepal width の貢献はほぼなし。

---

## 4. まとめ

- **EDA**: petal length / petal width はクラス判別に有効で、相互の相関も高い（0.96）。setosa は他クラスと明確に分離されるが、versicolor と virginica は重なりがある。
- **モデル精度**: テスト精度 90.0%、交差検証 96.7% と高い精度を達成。setosa は完全に分類できており、誤分類はすべて versicolor ↔ virginica 間で発生。
- **改善余地**: versicolor / virginica の分離精度向上にはハイパーパラメータチューニングや特徴量エンジニアリングが有効と考えられる。
