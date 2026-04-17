import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("outputs/results.csv")

print("Class distribution:")
print(df['folder_label'].value_counts())

# -------------------------------
# FEATURES (UPDATED WITH EDGE)
# -------------------------------
features = ['tsci', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'edge_density']

X = df[features].values
y = (df['folder_label'] == 'good').astype(int)  # 1=good, 0=bad

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -------------------------------
# MAIN MODEL — BALANCED SVM
# -------------------------------
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced'))
])

y_pred = cross_val_predict(model, X, y, cv=cv)

print("\n=== FINAL MODEL (SVM + EDGE FEATURE) ===")
print(classification_report(y, y_pred, target_names=['Bad/Worn', 'Good']))

acc_full = accuracy_score(y, y_pred)


# -------------------------------
# CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=['Bad/Worn', 'Good'])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix — SVM (Enhanced Features)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
print("Saved → outputs/confusion_matrix.png")


# -------------------------------
# BASELINES
# -------------------------------
print("\n=== BASELINES ===")

# TSCI threshold baseline
tsci_pred = (df['tsci'] > 0.50).astype(int)
acc_tsci = accuracy_score(y, tsci_pred)

# Homogeneity baseline
homog_pred = (df['homogeneity'] < 0.25).astype(int)
acc_homog = accuracy_score(y, homog_pred)

print(f"TSCI Threshold       : {acc_tsci:.2%}")
print(f"Homogeneity Only     : {acc_homog:.2%}")
print(f"Proposed SVM Method  : {acc_full:.2%}")


# -------------------------------
# ABLATION STUDY (UPDATED)
# -------------------------------
print("\n=== ABLATION STUDY ===")

ablation_configs = {
    'TSCI only': ['tsci'],
    'GLCM only': ['contrast','dissimilarity','homogeneity','energy','correlation'],
    'Edge only': ['edge_density'],
    'GLCM + Edge': ['contrast','dissimilarity','homogeneity','energy','correlation','edge_density'],
    'Full (All Features)': features,
}

ablation_results = {}

for name, feats in ablation_configs.items():
    X_ab = df[feats].values

    model_ab = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced'))
    ])

    preds = cross_val_predict(model_ab, X_ab, y, cv=cv)
    acc = accuracy_score(y, preds)

    ablation_results[name] = acc
    print(f"{name:<25}: {acc:.2%}")


# -------------------------------
# ABLATION PLOT
# -------------------------------
plt.figure(figsize=(6, 4))

names = list(ablation_results.keys())
values = list(ablation_results.values())

plt.bar(names, values)
plt.ylim(0.5, 1.0)
plt.title("Ablation Study — Feature Contribution")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)

for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.2%}", ha='center')

plt.tight_layout()
plt.savefig("outputs/ablation.png", dpi=150)
print("Saved → outputs/ablation.png")


# -------------------------------
# SAVE FINAL PREDICTIONS (3-CLASS)
# -------------------------------
model.fit(X, y)
df['binary_pred'] = model.predict(X)

def assign_3class(row):
    if row['binary_pred'] == 1:
        return 'Safe'
    elif row['tsci'] > 0.40:
        return 'Warning'
    else:
        return 'Dangerous'

df['predicted_label'] = df.apply(assign_3class, axis=1)
df.to_csv("outputs/results_final.csv", index=False)

print("Saved → outputs/results_final.csv")