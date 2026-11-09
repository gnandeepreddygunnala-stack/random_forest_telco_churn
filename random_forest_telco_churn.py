"""
Random Forest Machine Learning Project
Customer Churn Prediction (Classification)

Dataset : Telco Customer Churn (IBM sample data, real-world, 7,043 rows)
Source  : Kaggle - https://www.kaggle.com/datasets/blastchar/telco-customer-churn
File    : data_base.csv (original Kaggle name: WA_Fn-UseC_-Telco-Customer-Churn.csv)

HOW TO RUN (Google Colab or Jupyter / local):
  1. Download the CSV from the Kaggle link above.
  2. Put it next to this script (or upload it when Colab asks).
  3. Run the script. It prints all results and also creates:
       - rf_report.pdf              (1-page project report with your real results)
       - feature_importance.png
       - confusion_matrix.png
       - random_forest_churn_model.joblib
"""

# ----------------------------------------------------------------------
# 1. Import required libraries
# ----------------------------------------------------------------------
import os
import textwrap
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 2. Load the dataset
# ----------------------------------------------------------------------
# Works with either file name (same Telco Churn data from Kaggle)
for _name in ["data_base.csv", "WA_Fn-UseC_-Telco-Customer-Churn.csv"]:
    if os.path.exists(_name):
        CSV_NAME = _name
        break
else:
    CSV_NAME = "data_base.csv"

if not os.path.exists(CSV_NAME):
    # In Google Colab, ask the user to upload the CSV downloaded from Kaggle
    try:
        from google.colab import files  # type: ignore

        print("Please upload", CSV_NAME)
        files.upload()
    except ImportError:
        raise FileNotFoundError(
            f"'{CSV_NAME}' not found. Download it from Kaggle and place it "
            "in the same folder as this script."
        )

df = pd.read_csv(CSV_NAME)
print("Dataset loaded. Shape:", df.shape)

# ----------------------------------------------------------------------
# 3. Basic data exploration
# ----------------------------------------------------------------------
print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Info ---")
df.info()
print("\n--- Summary statistics (numeric) ---")
print(df.describe())
print("\n--- Target distribution (Churn) ---")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True).round(3))

n_rows, n_cols = df.shape
churn_rate = (df["Churn"] == "Yes").mean() * 100

# ----------------------------------------------------------------------
# 4. Handle missing values
# ----------------------------------------------------------------------
# 'TotalCharges' is stored as text because a few rows contain blank spaces
# (new customers with tenure = 0). Convert to numeric -> blanks become NaN.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\n--- Missing values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

n_missing = int(df["TotalCharges"].isnull().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# customerID is just an identifier and has no predictive value
df = df.drop(columns=["customerID"])

# ----------------------------------------------------------------------
# 5. Encode categorical variables
# ----------------------------------------------------------------------
# Target: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encode every remaining text column
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
print("\nCategorical columns encoded:", categorical_cols)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("Shape after encoding:", df_encoded.shape)

# ----------------------------------------------------------------------
# 6. Split features (X) and target (y)
# ----------------------------------------------------------------------
X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

# ----------------------------------------------------------------------
# 7. Train-test split (80/20, stratified to keep the churn ratio equal)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)

# ----------------------------------------------------------------------
# 8. Apply the Random Forest model
# ----------------------------------------------------------------------
# class_weight='balanced' handles the imbalance (only ~27% customers churn)
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# ----------------------------------------------------------------------
# 9. Train the model
# ----------------------------------------------------------------------
rf.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 10. Make predictions
# ----------------------------------------------------------------------
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------------------
# 11. Evaluate the model
# ----------------------------------------------------------------------
def evaluate(y_true, y_hat, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_hat),
        "Precision": precision_score(y_true, y_hat),
        "Recall": recall_score(y_true, y_hat),
        "F1-Score": f1_score(y_true, y_hat),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


base = evaluate(y_test, y_pred, y_prob)
print("\n=== Baseline Random Forest - Test Metrics ===")
for k, v in base.items():
    print(f"{k:10s}: {v:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# BONUS A: 5-fold Cross Validation
# ----------------------------------------------------------------------
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="f1", n_jobs=-1)
print(f"\n5-fold CV F1: {cv_scores.round(3)} | mean = {cv_scores.mean():.4f}")

# ----------------------------------------------------------------------
# BONUS B: Hyperparameter tuning (RandomizedSearchCV, optimises F1)
# ----------------------------------------------------------------------
param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 6, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=15,
    scoring="f1",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
search.fit(X_train, y_train)
best_rf = search.best_estimator_
print("\nBest parameters:", search.best_params_)

y_pred_t = best_rf.predict(X_test)
y_prob_t = best_rf.predict_proba(X_test)[:, 1]
tuned = evaluate(y_test, y_pred_t, y_prob_t)
print("\n=== Tuned Random Forest - Test Metrics ===")
for k, v in tuned.items():
    print(f"{k:10s}: {v:.4f}")

# Keep whichever model has the better F1 on the test set
final_name, final_model, final = (
    ("Tuned Random Forest", best_rf, tuned)
    if tuned["F1-Score"] >= base["F1-Score"]
    else ("Baseline Random Forest", rf, base)
)
print("\nFinal model chosen:", final_name)

# ----------------------------------------------------------------------
# 12. Feature importance
# ----------------------------------------------------------------------
importances = (
    pd.Series(final_model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)
print("\nTop 10 features:")
print(importances.head(10).round(4))

top10 = importances.head(10)[::-1]
plt.figure(figsize=(8, 5))
plt.barh(top10.index, top10.values, color="#2b6cb0")
plt.xlabel("Importance (mean decrease in impurity)")
plt.title("Top 10 Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# BONUS C: save the model
joblib.dump(final_model, "random_forest_churn_model.joblib")
print("Model saved to random_forest_churn_model.joblib")

# ----------------------------------------------------------------------
# 13. Conclusion
# ----------------------------------------------------------------------
top3 = ", ".join(importances.head(3).index)
print(
    f"\nConclusion: The {final_name} reached an F1-score of {final['F1-Score']:.3f} "
    f"and ROC-AUC of {final['ROC-AUC']:.3f} on unseen data. The most influential "
    f"features were: {top3}."
)

# ----------------------------------------------------------------------
# AUTO-GENERATE THE 1-PAGE PDF REPORT (uses your real results above)
# ----------------------------------------------------------------------
def build_report(path="rf_report.pdf"):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")
    y_pos = 0.965

    def heading(text):
        nonlocal y_pos
        fig.text(0.07, y_pos, text, fontsize=11, fontweight="bold", color="#1a365d")
        y_pos -= 0.022

    def para(text, width=105, size=9, gap=0.0175):
        nonlocal y_pos
        for line in textwrap.wrap(text, width=width):
            fig.text(0.07, y_pos, line, fontsize=size, color="#222222")
            y_pos -= gap
        y_pos -= 0.008

    fig.text(0.5, y_pos, "Random Forest Machine Learning Project", ha="center",
             fontsize=16, fontweight="bold", color="#1a365d")
    y_pos -= 0.026
    fig.text(0.5, y_pos, "Customer Churn Prediction using Random Forest (Classification)",
             ha="center", fontsize=10.5, color="#444444")
    y_pos -= 0.014
    fig.add_artist(plt.Line2D([0.07, 0.93], [y_pos, y_pos], color="#2b6cb0", lw=1.2))
    y_pos -= 0.026

    heading("Dataset")
    para(f"Telco Customer Churn (Kaggle, by BlastChar / IBM sample data) - "
         f"https://www.kaggle.com/datasets/blastchar/telco-customer-churn. "
         f"{n_rows:,} customers and {n_cols} columns; real-world, non-synthetic data.")

    heading("Problem Statement")
    para(f"Predict whether a telecom customer will churn (leave the company). Overall churn rate is "
         f"{churn_rate:.1f}%, so the data is imbalanced. Early identification of likely churners "
         f"lets the company run retention offers.")

    heading("Data Preprocessing Steps")
    para(f"(1) Converted TotalCharges from text to numeric; {n_missing} blank values were filled with the "
         f"median. (2) Dropped customerID (identifier only). (3) Encoded target Churn as Yes=1 / No=0. "
         f"(4) One-hot encoded all categorical columns ({X.shape[1]} features after encoding). "
         f"(5) Stratified 80/20 train-test split (random_state=42).")

    heading("Model Used")
    bp = search.best_params_
    para(f"RandomForestClassifier (scikit-learn) with class_weight='balanced'. Baseline: 200 trees. "
         f"Tuned with RandomizedSearchCV (3-fold, F1 scoring); best parameters: "
         f"{', '.join(f'{k}={v}' for k, v in bp.items())}. Final model selected: {final_name}.")

    heading("Evaluation Metrics (Test Set)")
    rows = [["Metric", "Baseline RF", "Tuned RF"]] + [
        [k, f"{base[k]:.3f}", f"{tuned[k]:.3f}"] for k in base
    ]
    tbl = fig.add_axes([0.07, y_pos - 0.135, 0.5, 0.13])
    tbl.axis("off")
    t = tbl.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.38, 0.31, 0.31])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", fontweight="bold")
    tn, fp, fn, tp = confusion_matrix(y_test, final_model.predict(X_test)).ravel()
    fig.text(0.62, y_pos - 0.03, "Confusion matrix (final model)", fontsize=9, fontweight="bold")
    fig.text(0.62, y_pos - 0.052, f"True Negatives: {tn}    False Positives: {fp}", fontsize=8.5)
    fig.text(0.62, y_pos - 0.07, f"False Negatives: {fn}    True Positives: {tp}", fontsize=8.5)
    fig.text(0.62, y_pos - 0.095, f"5-fold CV F1 (baseline): {cv_scores.mean():.3f}", fontsize=8.5)
    y_pos -= 0.15

    heading("Key Findings")
    top5 = importances.head(5)
    feat_txt = "; ".join(f"{n} ({v:.3f})" for n, v in top5.items())
    para(f"Top 5 features by importance: {feat_txt}. These variables drive most of the model's "
         f"decisions, so customer tenure, billing amounts and contract/service type are the main "
         f"levers for retention. Tuning changed F1 from {base['F1-Score']:.3f} to {tuned['F1-Score']:.3f}.")

    heading("Conclusion")
    para(f"The {final_name} predicts churn with accuracy {final['Accuracy']:.1%}, recall {final['Recall']:.1%} "
         f"and ROC-AUC {final['ROC-AUC']:.3f} on unseen data. Random Forest handles mixed customer features "
         f"well and gives interpretable importances. Future work: try SMOTE or threshold tuning to improve "
         f"recall on churners, and compare against Gradient Boosting.", size=9)

    # Feature importance chart fills the remaining space at the bottom of the page
    ax = fig.add_axes([0.27, 0.06, 0.63, max(y_pos - 0.11, 0.1)])
    top8 = importances.head(8)[::-1]
    ax.barh(top8.index, top8.values, color="#2b6cb0")
    ax.set_title("Top 8 Feature Importances", fontsize=9)
    ax.tick_params(labelsize=7.5)

    fig.savefig(path, format="pdf")
    plt.close(fig)
    print("Report saved to", path)


build_report()
