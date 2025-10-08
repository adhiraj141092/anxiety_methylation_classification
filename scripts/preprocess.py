import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt

def preprocess_data(beta_train, labels_train, beta_val, labels_val):
    """
    Preprocess the data by extracting features and labels, and standardizing the features.
    Parameters:
    beta_train (DataFrame): Methylation beta values for training set (probes x samples).
    labels_train (DataFrame): Labels for training set.
    beta_val (DataFrame): Methylation beta values for validation set (probes x samples).
    labels_val (DataFrame): Labels for validation set.
    Returns:
    X_train (ndarray): Standardized feature set for training.
    y_train (ndarray): Target labels for training.
    X_val (ndarray): Standardized feature set for validation.
    y_val (ndarray): Target labels for validation.
    """
    print("Starting preprocessing...")
    print(f"Initial train shape: {beta_train.shape}, val shape: {beta_val.shape}")
    # Extract features (transpose: samples as rows, probes as columns)
    X_train = beta_train.T.values
    X_val = beta_val.T.values 
    # Extract and map labels
    y_train_str = labels_train['diagnosis']
    y_train = y_train_str.map({'case': 1, 'control': 0}).values
    y_val_str = labels_val["characteristics_ch1.2"]
    y_val = y_val_str.map({'disease state: Major depressive disorder': 1, 'disease state: Healthy': 0}).values

    print(f"Train shape: {X_train.shape}, Labels unique: {np.unique(y_train)}, Counts: {np.bincount(y_train)}")
    print(f"Val shape: {X_val.shape}, Labels unique: {np.unique(y_val)}, Counts: {np.bincount(y_val)}")
    # Standardization per feature (probe)
    mean_train = np.mean(X_train, axis=0)  # Shape: (62692,)
    std_train = np.std(X_train, axis=0)
    std_train[std_train == 0] = 1  # Avoid division by zero
    X_train = (X_train - mean_train) / std_train
    X_val = (X_val - mean_train) / std_train  # Broadcasts correctly now
    print("Preprocessing complete. X_train mean/std per feature: ~0/1")
    print(f"Sample X_train[0, :5]: {X_train[0, :5]}")  # Quick check

    return X_train, y_train, X_val, y_val

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to the training data to handle class imbalance.

    Parameters:
    X_train (DataFrame or ndarray): Feature set for training.
    y_train (Series or ndarray): Target labels for training.

    Returns:
    X_res (ndarray): Resampled feature set.
    y_res (ndarray): Resampled target labels.
    """
    # Initialize SMOTE
    smote = SMOTE(random_state=42)

    # Apply it to training data
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", np.bincount(y_train))
    print("After SMOTE :", np.bincount(y_train_res))
    print("X_train_res shape:", X_train_res.shape)
    print("y_train_res shape:", y_train_res.shape)

    return X_train_res, y_train_res

def apply_near_miss(X_train, y_train):
    """
    Apply NearMiss to the training data to handle class imbalance.

    Parameters:
    X_train (DataFrame or ndarray): Feature set for training.
    y_train (Series or ndarray): Target labels for training.

    Returns:
    X_res (ndarray): Resampled feature set.
    y_res (ndarray): Resampled target labels.
    """

    # Initialize NearMiss
    nm = NearMiss()

    # Apply it to training data
    X_train_res, y_train_res = nm.fit_resample(X_train, y_train)

    print("Before NearMiss:", np.bincount(y_train))
    print("After NearMiss :", np.bincount(y_train_res))
    print("X_train_res shape:", X_train_res.shape)
    print("y_train_res shape:", y_train_res.shape)

    return X_train_res, y_train_res



def pca_transformation(X_train, X_val, variance_threshold=0.95):
    """
    Preprocess the data by applying SMOTE, variance filtering, standardization, and PCA.

    Parameters:
    X_train (DataFrame or ndarray): Feature set for training.
    y_train (Series or ndarray): Target labels for training.
    X_val (DataFrame or ndarray): Feature set for validation.

    Returns:
    X_train_pca (ndarray): Preprocessed training feature set after PCA.
    X_val_pca (ndarray): Preprocessed validation feature set after PCA.
    """
    # Apply SMOTE

    # Variance filter (remove near-constant CpGs)
    vt = VarianceThreshold(threshold=1e-5)
    X_train_vt = vt.fit_transform(X_train)
    X_val_vt = vt.transform(X_val)

    print("After variance filter:", X_train_vt.shape[1], "features retained")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_vt)
    X_val_scaled = scaler.transform(X_val_vt)

    # PCA to retain 95% variance
    pca = PCA(n_components=variance_threshold, svd_solver="full", random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    print("PCA retained", X_train_pca.shape[1], f"components ({variance_threshold} variance)")

    return X_train_pca, X_val_pca


def pca_scree_plot(X, transformation, threshold=0.95):
    """
    PCA, and plot the scree plot showing explained variance.

    Parameters:
    X (DataFrame or ndarray): Feature set.
    y (Series or ndarray): Target labels.
    """

    # Fit PCA with enough components
    pca = PCA(n_components=None)  # keep all components initially
    pca.fit(X)

    # Explained variance ratio (cumulative)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # Determine number of components for 95% variance
    n_components_95 = np.argmax(cum_var >= threshold) + 1
    print(f"Number of components to retain {threshold} variance: {n_components_95}")

    # Plot Scree plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
             pca.explained_variance_ratio_, marker='.', label='Individual variance')
    plt.plot(range(1, len(cum_var)+1),
             cum_var, marker='.', linestyle='--', label='Cumulative variance')

    plt.axvline(n_components_95, color='r', linestyle=':', label=f'95% cutoff = {n_components_95} comps')
    plt.axhline(0.95, color='g', linestyle='--')

    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot after NearMiss (95% variance cutoff)')
    plt.legend()
    plt.savefig(f"results/plots/PCA_Scree_{transformation}.png", dpi=300)
    plt.show()
    print("Scree plot saved as:", f"results/plots/PCA_Scree_{transformation}.png")