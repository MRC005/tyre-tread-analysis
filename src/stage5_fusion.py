"""
STAGE 5 — KNN Training, Evaluation & Fusion Decision
=====================================================
Module : Fusion (M3 + M6 integrated)
Purpose: (A) Train a K-Nearest-Neighbour classifier on texture
             features extracted from the full dataset.
         (B) Define the fusion logic that selects between the
             primary TSCI path and the KNN fallback path.

Fusion Strategy
---------------
Primary path  (TSCI)  — used when the DFT signal is reliable.
              Indicator: TSCI value is not near the decision
              boundaries (|TSCI - threshold| > margin).

Fallback path (KNN)   — used when TSCI is ambiguous, i.e. the
              DFT energy ratio falls in the uncertain band near
              a threshold.

This hybrid design gives the system two independent channels of
evidence — frequency-domain and texture-domain — each grounded
in a different module of classical DIP.
"""

import numpy as np
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_MAP    = {0: "Safe", 1: "Warning", 2: "Dangerous"}
KNN_K        = 5       # tuned by cross-validation
TEST_SIZE    = 0.30
RANDOM_STATE = 42
TSCI_MARGIN  = 0.05    # ambiguity margin around each threshold


def train_knn(X: np.ndarray, y: np.ndarray,
              save_dir: str = "outputs"):
    """
    Train, evaluate, and save a KNN classifier on texture features.

    Parameters
    ----------
    X        : ndarray (N, 31)  Feature matrix (GLCM + LBP per image)
    y        : ndarray (N,)     Integer labels  0=Safe 1=Warning 2=Dangerous
    save_dir : str              Directory for saving model + report figure

    Returns
    -------
    knn     : fitted KNeighborsClassifier
    scaler  : fitted StandardScaler (must be applied at inference time)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Scale features (KNN is distance-based — scaling is essential)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split (stratified to preserve class balance)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # K selection via 5-fold cross-validation on training set
    best_k, best_score = KNN_K, 0.0
    for k in range(3, 12, 2):
        cv_scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=k, metric='euclidean'),
            X_tr, y_tr,
            cv=StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE),
            scoring='accuracy'
        )
        if cv_scores.mean() > best_score:
            best_score, best_k = cv_scores.mean(), k

    print(f"\n  [Stage 5] Best K = {best_k}  (CV accuracy = {best_score:.3f})")

    # Final model
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn.fit(X_tr, y_tr)

    # Evaluation
    y_pred = knn.predict(X_te)
    print("\n  [Stage 5] Classification Report:")
    print(classification_report(y_te, y_pred,
                                target_names=["Safe", "Warning", "Dangerous"],
                                digits=3))

    # Confusion matrix figure
    cm = confusion_matrix(y_te, y_pred)
    _plot_confusion(cm, save_path=os.path.join(save_dir, "confusion_matrix.png"))

    # Save model and scaler
    joblib.dump(knn,    os.path.join(save_dir, "knn_model.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    print(f"  [Stage 5] ✅ Model saved → {save_dir}/knn_model.pkl")

    return knn, scaler


def load_knn(save_dir: str = "outputs"):
    """Load a previously trained KNN model and scaler."""
    knn_path    = os.path.join(save_dir, "knn_model.pkl")
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    if not os.path.exists(knn_path):
        return None, None
    knn    = joblib.load(knn_path)
    scaler = joblib.load(scaler_path)
    return knn, scaler


def fusion_decision(tsci: float,
                    texture_features: np.ndarray,
                    knn=None,
                    scaler=None,
                    tsci_safe: float = 0.55,
                    tsci_warn: float = 0.35,
                    margin: float    = TSCI_MARGIN):
    """
    Hybrid fusion: choose TSCI or KNN path based on confidence.

    Decision rule
    -------------
    If TSCI is clearly above/below both thresholds by at least
    `margin`, the DFT signal is unambiguous → use TSCI label.
    Otherwise the image falls in a transition zone → use KNN.

    Parameters
    ----------
    tsci             : float   TSCI value from Stage 3
    texture_features : ndarray (31,) from Stage 4
    knn, scaler      : trained classifier and scaler (may be None)
    tsci_safe        : upper classification threshold
    tsci_warn        : lower classification threshold
    margin           : ambiguity band half-width around thresholds

    Returns
    -------
    label  : str   'Safe' | 'Warning' | 'Dangerous'
    method : str   description of path taken
    """
    # Check whether TSCI falls in an ambiguous transition zone
    near_safe = abs(tsci - tsci_safe) < margin
    near_warn = abs(tsci - tsci_warn) < margin
    ambiguous = near_safe or near_warn

    if not ambiguous:
        # TSCI is confident — use frequency-domain classification
        if tsci > tsci_safe:
            label = "Safe"
        elif tsci > tsci_warn:
            label = "Warning"
        else:
            label = "Dangerous"
        method = "Frequency-domain (TSCI) — high confidence"

    elif knn is not None and scaler is not None:
        # TSCI is ambiguous — defer to texture KNN
        feat_scaled = scaler.transform([texture_features])
        pred        = knn.predict(feat_scaled)[0]
        label       = LABEL_MAP.get(pred, "Unknown")
        method      = "Texture-domain (KNN) — TSCI ambiguous"

    else:
        # No trained KNN available — fall back to raw TSCI
        if tsci > tsci_safe:
            label = "Safe"
        elif tsci > tsci_warn:
            label = "Warning"
        else:
            label = "Dangerous"
        method = "Frequency-domain (TSCI) — KNN not trained yet"

    return label, method


def _plot_confusion(cm: np.ndarray, save_path: str):
    """Plot and save a labelled confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes = ["Safe", "Warning", "Dangerous"]
    ticks   = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(ticks); ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_title("KNN Confusion Matrix (Test Set)", fontsize=12,
                 fontweight='bold')

    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=13,
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Stage 5] ✅ Confusion matrix saved → {save_path}")