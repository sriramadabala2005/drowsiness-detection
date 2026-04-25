import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(BASE_DIR, "features.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

CLASS_NAMES = ["alert", "microsleep", "drowsy", "extreme_fatigue"]
FEATURE_COLS = ["left_ear", "right_ear", "avg_ear", "mar",
                "pitch", "yaw", "roll", "eye_closed"]

# ─────────────────────────────────────────────
#  LOAD & PREPARE DATA
# ─────────────────────────────────────────────
def load_data():
    print("\n  Loading features.csv ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Total rows loaded : {len(df)}")
    print(f"  Columns           : {list(df.columns)}")

    print("\n  Class distribution:")
    for cls in CLASS_NAMES:
        count = len(df[df["class_name"] == cls])
        bar   = "█" * int(count / 25) + "░" * (20 - int(count / 25))
        print(f"    {cls:<20} [{bar}] {count}")

    # drop rows with NaN
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < before:
        print(f"\n  Dropped {before - len(df)} rows with missing values")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    return X, y, df


# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved → {path}")


def plot_feature_importance(model, model_name):
    if not hasattr(model.named_steps["clf"], "feature_importances_"):
        return
    importances = model.named_steps["clf"].feature_importances_
    indices     = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if i == indices[0] else "#90CAF9" for i in range(len(FEATURE_COLS))]
    ax.bar(range(len(FEATURE_COLS)),
           importances[indices],
           color=[colors[i] for i in range(len(FEATURE_COLS))],
           edgecolor="white")
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in indices], rotation=30, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved → {path}")


def plot_class_distribution(df):
    counts = [len(df[df["class_name"] == c]) for c in CLASS_NAMES]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_NAMES, counts, color=colors, edgecolor="white", width=0.6)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset class distribution", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────
#  TRAIN & EVALUATE
# ─────────────────────────────────────────────
def train_and_evaluate(name, clf, X_train, X_test, y_train, y_test):
    print(f"\n  {'─'*45}")
    print(f"  Training: {name}")
    print(f"  {'─'*45}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf)
    ])

    # cross validation
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  Cross-val accuracy : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # final fit
    pipeline.fit(X_train, y_train)
    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  Test accuracy      : {accuracy*100:.2f}%")
    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, name)
    plot_feature_importance(pipeline, name)

    return pipeline, accuracy


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  MODEL TRAINING  —  Drowsiness Detection")
    print("="*55)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: features.csv not found at {CSV_PATH}")
        print("Run extract_features.py first.")
        return

    X, y, df = load_data()

    # plot class distribution
    print("\n  Generating plots ...")
    plot_class_distribution(df)

    # train / test split — stratified so each class is represented equally
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train samples : {len(X_train)}")
    print(f"  Test samples  : {len(X_test)}")

    # ── define models ──
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
            probability=True
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),
    }

    results   = {}
    pipelines = {}

    for name, clf in models.items():
        pipeline, accuracy = train_and_evaluate(
            name, clf, X_train, X_test, y_train, y_test
        )
        results[name]   = accuracy
        pipelines[name] = pipeline

    # ── pick best model ──
    best_name     = max(results, key=results.get)
    best_pipeline = pipelines[best_name]
    best_accuracy = results[best_name]

    print("\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = "  ← BEST" if name == best_name else ""
        print(f"  {name:<25} {acc*100:.2f}%{marker}")

    # ── save best model ──
    model_path  = os.path.join(MODELS_DIR, "drowsiness_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    joblib.dump(best_pipeline, model_path)
    joblib.dump(best_pipeline.named_steps["scaler"], scaler_path)

    print(f"\n  Best model  : {best_name}  ({best_accuracy*100:.2f}%)")
    print(f"  Saved to    : {model_path}")
    print(f"  Plots saved : {PLOTS_DIR}")
    print("="*55)

    # ── save model info ──
    info_path = os.path.join(MODELS_DIR, "model_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Best Model    : {best_name}\n")
        f.write(f"Test Accuracy : {best_accuracy*100:.2f}%\n")
        f.write(f"Features Used : {FEATURE_COLS}\n")
        f.write(f"Classes       : {CLASS_NAMES}\n")
        f.write(f"Train samples : {len(X_train)}\n")
        f.write(f"Test samples  : {len(X_test)}\n")
        for name, acc in results.items():
            f.write(f"{name} accuracy: {acc*100:.2f}%\n")

if __name__ == "__main__":
    main()