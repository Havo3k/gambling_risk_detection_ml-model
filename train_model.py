"""
Обучение модели детекции риск-паттерна (формирующейся игровой зависимости)
на синтетических признаках, сгенерированных generate_dataset.py.

Использование:
    python3 train_model.py --features features.csv

Что делает скрипт:
    1. Загружает таблицу признаков на клиента (features.csv)
    2. Делит клиентов на train/test
    3. Обучает две модели для сравнения: логистическую регрессию (baseline,
       интерпретируемая — важно для банковского/регуляторного контекста)
       и градиентный бустинг (XGBoost) — более сильную, но менее прозрачную
    4. Считает метрики качества: accuracy, precision, recall, F1, ROC-AUC
    5. Сохраняет ROC-кривую и таблицу важности признаков как картинки —
       их можно прямо вставлять в презентацию
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


FEATURE_COLS = [
    "n_bets", "bet_frequency", "total_bet_amount", "avg_bet_amount",
    "bet_amount_escalation", "night_bet_share", "n_microloans",
    "loans_same_day_as_bet",
]
TARGET_COL = "label_at_risk"


def evaluate(model_name, y_true, y_pred, y_proba):
    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.3f}")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(y_true, y_pred))
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="features.csv")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    # --- Baseline: логистическая регрессия (интерпретируемая модель) ---
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed)
    logreg.fit(X_train_s, y_train)
    y_pred_lr = logreg.predict(X_test_s)
    y_proba_lr = logreg.predict_proba(X_test_s)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr))

    print("\nВажность признаков (коэффициенты логрегрессии, |вес|):")
    coef_importance = pd.Series(np.abs(logreg.coef_[0]), index=FEATURE_COLS).sort_values(ascending=False)
    print(coef_importance.round(3))

    fig, ax = plt.subplots(figsize=(6, 4))
    coef_importance.sort_values().plot(kind="barh", ax=ax, color="#2b6cb0")
    ax.set_title("Важность признаков (Logistic Regression)")
    ax.set_xlabel("|коэффициент| (после стандартизации)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    # --- Градиентный бустинг (если доступен) ---
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=args.seed
        )
        xgb.fit(X_train, y_train)  # бустингу стандартизация не нужна
        y_pred_xgb = xgb.predict(X_test)
        y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
        results.append(evaluate("XGBoost", y_test, y_pred_xgb, y_proba_xgb))
        best_proba_for_roc = {"Logistic Regression": y_proba_lr, "XGBoost": y_proba_xgb}
    else:
        print("\n[XGBoost не установлен — пропускаю, используем только логрегрессию]")
        best_proba_for_roc = {"Logistic Regression": y_proba_lr}

    # --- ROC-кривая (для презентации) ---
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, proba in best_proba_for_roc.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="случайное угадывание")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая: детекция риск-паттерна")
    ax.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.close()

    # --- Итоговая таблица метрик (для доклада) ---
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_metrics.csv", index=False)
    print("\n=== Итоговая таблица метрик сохранена в model_metrics.csv ===")
    print(results_df.round(3).to_string(index=False))
    print("\nГрафики сохранены: feature_importance.png, roc_curve.png")


if __name__ == "__main__":
    main()
