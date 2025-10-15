from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import expon, uniform
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, average_precision_score, precision_recall_curve,
    matthews_corrcoef, f1_score, recall_score, precision_score, accuracy_score
)
import numpy as np
from scipy.stats import randint, uniform
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time


def train_svm(X_train_pca, y_train_res, transformation):
    """
    Train an SVM model with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
    X_train_pca (ndarray): PCA-transformed training features.
    y_train_res (ndarray): Resampled training labels.


    Returns:
    best_svm (SVC): Best SVM model after hyperparameter tuning.
    """
    print("Starting SVM training with hyperparameter tuning...")
# --- Hyperparameter space for SVM (RBF kernel example) ---
    param_dist = {
        'C': uniform(0.1, 10),       # regularization
        'gamma': expon(scale=0.1),   # RBF kernel width
        'kernel': ['rbf', 'linear']  # optionally test linear too
    }

    # --- Stratified 5-fold CV ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- RandomizedSearchCV ---
    svm_search = RandomizedSearchCV(
        estimator=SVC(probability=True, random_state=42),
        param_distributions=param_dist,
        n_iter=30,
        cv=cv,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1,
        return_train_score=True,
        random_state=42
    )

    # --- Fit on training data ---
    svm_search.fit(X_train_pca, y_train_res)

    # --- Best model and evaluation ---
    best_svm = svm_search.best_estimator_
        # --- Save CV results to CSV ---
    svm_results_df = pd.DataFrame(svm_search.cv_results_)
    svm_results_df.to_csv(f"results/cv_score/svm_randomsearch_{transformation}_results.csv", index=False)
    print("Saved SVM hyperparameter search results to svm_randomsearch_results.csv")
    print("Best params:", svm_search.best_params_)
    print("Best CV score:", svm_search.best_score_)

    return best_svm, svm_search

def model_prediction(classifier, X):
    """
    Generate predictions and probabilities using the trained classifier.
    Parameters:
    classifier (estimator): Trained classifier.
    X (ndarray): Features to predict on.
    Returns:
    y_pred (ndarray): Predicted class labels.
    y_proba (ndarray): Predicted probabilities for the positive class.
    """
    y_pred = classifier.predict(X)
    y_proba = classifier.predict_proba(X)[:, 1]

    return y_pred, y_proba

def model_evaluation(y_val, y_pred, y_proba, model_name):
    """
    Evaluate model performance and return metrics.  
    Parameters:
    y_val (ndarray): True labels.
    y_pred (ndarray): Predicted class labels.
    y_proba (ndarray): Predicted probabilities for the positive class.
    model_name (str): Name of the model for tracking results.
    Returns:
    dict: Metrics including confusion matrix, scalar scores, and curves.
    """
     # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    # Metrics
    metrics = {
        "model": model_name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),   # aka sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "f1": f1_score(y_val, y_pred),
        "mcc": matthews_corrcoef(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "pr_auc": average_precision_score(y_val, y_proba),
    }

    print(f"Validation results for {model_name}:")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # For plotting later
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    prec, rec, _ = precision_recall_curve(y_val, y_proba)
    metrics["roc_curve"] = (fpr, tpr)
    metrics["pr_curve"] = (prec, rec)

    return metrics


def model_validation(classifier, X_val, y_val, model_name="model"):
    """
    Validate the trained model on the validation set and return metrics.

    Parameters:
    classifier (estimator): Trained classifier.
    X_val (ndarray): Validation features.
    y_val (ndarray): Validation labels.
    model_name (str): Name of the model for tracking results.

    Returns:
    dict: Metrics including confusion matrix, scalar scores, and curves.
    """
    # Predictions
    y_pred, y_proba = model_prediction(classifier, X_val)
    # Evaluation
    metrics = model_evaluation(y_val, y_pred, y_proba, model_name)

    return metrics

   


def train_rf(X_train_pca, y_train_res, transformation):
    """
    Train a Random Forest model with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
    X_train_pca (ndarray): PCA-transformed training features.
    y_train_res (ndarray): Resampled training labels.
    transformation (str): Name of the resampling technique used.

    Returns:
    best_rf (RandomForestClassifier): Best Random Forest model after hyperparameter tuning.
    """
    print("Starting Random Forest training with hyperparameter tuning...")

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # --- Hyperparameter space for Random Forest ---
    # --- Hyperparameter space ---
    param_dist = {
        "n_estimators": randint(100, 500),          # number of trees
        "max_depth": randint(5, 50),                # depth of trees
        "min_samples_split": randint(2, 20),        # min samples to split a node
        "min_samples_leaf": randint(1, 10),         # min samples in a leaf
        "max_features": ["sqrt", "log2", None],     # feature selection at splits
        "bootstrap": [True, False]
    }

    # --- Stratified 5-fold CV ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- RandomizedSearchCV ---
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,                # number of random combos to try
        scoring="accuracy",       # you can also use 'roc_auc'
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )

    # --- Fit on training data ---
    rf_search.fit(X_train_pca, y_train_res)

    # --- Best model and evaluation ---
    best_rf = rf_search.best_estimator_

    # --- Save CV results to CSV ---
    rf_results_df = pd.DataFrame(rf_search.cv_results_)
    rf_results_df.to_csv(f"results/cv_score/rf_randomsearch_{transformation}_results.csv", index=False)
    print("Saved Random Forest hyperparameter search results to rf_randomsearch_results.csv")
    print("Best params:", rf_search.best_params_)
    print("Best CV score:", rf_search.best_score_)

    return best_rf, rf_search

class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification.

    Parameters:
    input_dim (int): Number of input features.
    hidden_dim (int): Number of neurons in the hidden layer.
    dropout (float): Dropout rate for regularization.

    
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x



def objective(trial, X_train, y_train, results=None):
    """
    Objective function for Optuna hyperparameter optimization of MLP.
    
    Parameters:
    trial (optuna.trial.Trial): Optuna trial object.
    X_train (ndarray): Training features.
    y_train (ndarray): Training labels.
    results (list): List to store results of each trial.
    
    """

    if results is None:
        results = []
    # Define hyperparameter search space
    params = {
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 512, step=32),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.1),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }

    # Initialize 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    fit_times = []
    score_times = []

    X_train_np = X_train
    y_train_np = y_train

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        # Split data
        X_train_fold = X_train_np[train_idx]
        y_train_fold = y_train_np[train_idx]
        X_val_fold = X_train_np[val_idx]
        y_val_fold = y_train_np[val_idx]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        # Initialize model
        model = MLP(input_dim=X_train_tensor.shape[1], hidden_dim=params['hidden_dim'], dropout=params['dropout'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        # Training loop with early stopping
        start_time = time.time()
        best_loss = np.inf
        patience = 5
        counter = 0
        epochs = 50

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

            # Validation loss for early stopping
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        fit_time = time.time() - start_time

        # Evaluate on validation fold (accuracy)
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_pred = torch.argmax(val_logits, dim=1).numpy()
            fold_score = accuracy_score(y_val_fold, val_pred)
            cv_scores.append(fold_score)
        score_time = time.time() - start_time

        fit_times.append(fit_time)
        score_times.append(score_time)

    # Store results for this trial
    trial_result = {
        'mean_fit_time': np.mean(fit_times),
        'std_fit_time': np.std(fit_times),
        'mean_score_time': np.mean(score_times),
        'std_score_time': np.std(score_times),
        'params': params,
        'split0_test_score': cv_scores[0],
        'split1_test_score': cv_scores[1],
        'split2_test_score': cv_scores[2],
        'split3_test_score': cv_scores[3],
        'split4_test_score': cv_scores[4],
        'mean_test_score': np.mean(cv_scores),
        'std_test_score': np.std(cv_scores),
        'rank_test_score': 0  # Will be updated after all trials
    }
    results.append(trial_result)

    # Return average CV score for Optuna optimization
    return np.mean(cv_scores)


def train_optuna(X_train, y_train, transformation, n_trials=10):
    """
    Train an MLP model with hyperparameter tuning using Optuna.
    Parameters:
    X_train (ndarray): Training features.
    y_train (ndarray): Training labels.
    transformation (str): Name of the resampling technique used.
    n_trials (int): Number of Optuna trials.
    Returns:
    best_params_ (dict): Best hyperparameters found by Optuna.
    """
    results = []

    def optuna_func(trial):
        mean_score = objective(trial, X_train, y_train, results)
        return mean_score

    # Run Optuna study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    study.optimize(optuna_func, n_trials)


    # Rank trials based on mean_test_score
    sorted_results = sorted(results, key=lambda x: x['mean_test_score'], reverse=True)
    for i, res in enumerate(sorted_results):
        res['rank_test_score'] = i + 1

    # Convert results to DataFrame
    cv_results = pd.DataFrame([
        {
            'mean_fit_time': res['mean_fit_time'],
            'std_fit_time': res['std_fit_time'],
            'mean_score_time': res['mean_score_time'],
            'std_score_time': res['std_score_time'],
            'param_hidden_dim': res['params']['hidden_dim'],
            'param_dropout': res['params']['dropout'],
            'param_learning_rate': res['params']['learning_rate'],
            'param_batch_size': res['params']['batch_size'],
            'params': res['params'],
            'split0_test_score': res['split0_test_score'],
            'split1_test_score': res['split1_test_score'],
            'split2_test_score': res['split2_test_score'],
            'split3_test_score': res['split3_test_score'],
            'split4_test_score': res['split4_test_score'],
            'mean_test_score': res['mean_test_score'],
            'std_test_score': res['std_test_score'],
            'rank_test_score': res['rank_test_score']
        } for res in results
    ])

    # Print CV results
    print("\nCV Results (Scikit-learn style):")
    print(cv_results.to_string(index=False))

    # Extract best parameters and score
    best_params_ = study.best_params
    best_score_ = study.best_value
    print("\nBest Hyperparameters:", best_params_)
    print("Best Average CV Accuracy:", best_score_)

    cv_results.to_csv(f"results/cv_score/mlp_optuna_randomsearch_results_{transformation}.csv", index=False)
    print("Saved MLP Optuna randomized search results to mlp_optuna_randomsearch_results.csv")

    return best_params_


def optuna_validation(X_train, y_train, X_val, y_val, best_params_, transformation):
    """
    Validate the best MLP model on the validation set using the best hyperparameters.
    Parameters:
    X_train (ndarray): Training features.
    y_train (ndarray): Training labels.
    X_val (ndarray): Validation features.
    y_val (ndarray): Validation labels.
    best_params_ (dict): Best hyperparameters found by Optuna.
    """
    print("\nValidating MLP with best hyperparameters on validation set...")
    # Train final model with best parameters (unchanged from previous code)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=best_params_['batch_size'], shuffle=True)

    model = MLP(
        input_dim=X_train_tensor.shape[1],
        hidden_dim=best_params_['hidden_dim'],
        dropout=best_params_['dropout']
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params_['learning_rate'])

    best_loss = np.inf
    patience = 5
    counter = 0
    epochs = 100

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"final_mlp_model_{transformation}.pt")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model
    model.load_state_dict(torch.load(f"final_mlp_model_{transformation}.pt"))

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        y_val_logits = model(X_val_tensor)
        y_val_pred = torch.argmax(y_val_logits, dim=1).numpy()
        y_val_proba = torch.softmax(y_val_logits, dim=1)[:, 1].numpy()
    
    # Evaluation
    metrics = model_evaluation(y_val, y_val_pred, y_val_proba, model_name="MLP")

    return metrics