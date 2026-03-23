"""
Iris Dataset - EDA & Classification Model
sklearnのIrisデータセットを用いた探索的データ分析と分類モデルの構築
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. データ読み込み
# ============================================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 50)
print("1. データ概要")
print("=" * 50)
print(f"サンプル数: {len(df)}")
print(f"特徴量: {list(iris.feature_names)}")
print(f"クラス: {list(iris.target_names)}")
print()

# ============================================================
# 2. EDA - 基本統計
# ============================================================
print("=" * 50)
print("2. 基本統計量")
print("=" * 50)
print(df.describe().round(2))
print()

print("クラス分布:")
print(df["species"].value_counts())
print()

# ============================================================
# 3. EDA - 相関分析
# ============================================================
print("=" * 50)
print("3. 特徴量間の相関係数")
print("=" * 50)
corr = df.drop(columns="species").corr().round(2)
print(corr)
print()

# 最も相関の高いペアを表示
corr_pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
)
print("相関の高いペア Top3:")
print(corr_pairs.head(3))
print()

# ============================================================
# 4. EDA - クラス別統計
# ============================================================
print("=" * 50)
print("4. クラス別平均値")
print("=" * 50)
print(df.groupby("species").mean().round(2))
print()

# ============================================================
# 5. 予測モデル構築
# ============================================================
print("=" * 50)
print("5. RandomForest 分類モデル")
print("=" * 50)

X = iris.data
y = iris.target

# 学習・テスト分割 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデル学習
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 評価
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"テストデータ精度: {accuracy:.4f} ({accuracy*100:.1f}%)")
print()

# 交差検証
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"5-fold 交差検証スコア: {cv_scores.round(3)}")
print(f"平均: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print()

# 分類レポート
print("分類レポート:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 混同行列
print("混同行列:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)
print(cm_df)
print()

# ============================================================
# 6. 特徴量重要度
# ============================================================
print("=" * 50)
print("6. 特徴量重要度")
print("=" * 50)
importances = pd.Series(model.feature_importances_, index=iris.feature_names)
importances_sorted = importances.sort_values(ascending=False)
for feat, imp in importances_sorted.items():
    bar = "#" * int(imp * 40)
    print(f"  {feat:<30} {imp:.4f}  {bar}")
print()

print("完了")
