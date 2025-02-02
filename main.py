import onchain_data
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


DB_FILE_NAME = "btc_onchain.db"

# onchain_data.create_btc_onchain_data_db()

def price_history_data_processing():

    engine = create_engine(f'sqlite:///{DB_FILE_NAME}')

    table_name = "BTC_price_history"
    query = f"SELECT * FROM {table_name}"
    price_history_df = pd.read_sql(query, engine)

    # Create shifted price columns
    price_history_df["price in 7D"] = price_history_df["price"].shift(-7)
    price_history_df["price in 30D"] = price_history_df["price"].shift(-30)
    price_history_df["price in 90D"] = price_history_df["price"].shift(-90)
    price_history_df["price in 365D"] = price_history_df["price"].shift(-365)

    # Calculate ROI for each time range
    price_history_df['7D_ROI'] = (price_history_df['price in 7D'] - price_history_df['price']) / price_history_df['price']
    price_history_df['30D_ROI'] = (price_history_df['price in 30D'] - price_history_df['price']) / price_history_df['price']
    price_history_df['90D_ROI'] = (price_history_df['price in 90D'] - price_history_df['price']) / price_history_df['price']
    price_history_df['365D_ROI'] = (price_history_df['price in 365D'] - price_history_df['price']) / price_history_df['price']

    # Create binary columns with NaN where future data is not available
    price_history_df['7D_greater'] = (price_history_df['price'] < price_history_df['price in 7D']).where(price_history_df['price in 7D'].notna(), pd.NA).astype('Int64')
    price_history_df['30D_greater'] = (price_history_df['price'] < price_history_df['price in 30D']).where(price_history_df['price in 30D'].notna(), pd.NA).astype('Int64')
    price_history_df['90D_greater'] = (price_history_df['price'] < price_history_df['price in 90D']).where(price_history_df['price in 90D'].notna(), pd.NA).astype('Int64')
    price_history_df['365D_greater'] = (price_history_df['price'] < price_history_df['price in 365D']).where(price_history_df['price in 365D'].notna(), pd.NA).astype('Int64')

    # rolling standard deviation (volatility)
    price_history_df['volatility_7D'] = price_history_df['price'].pct_change().rolling(window=7).std()
    price_history_df['volatility_30D'] = price_history_df['price'].pct_change().rolling(window=30).std()
    price_history_df['volatility_90D'] = price_history_df['price'].pct_change().rolling(window=90).std()
    price_history_df['volatility_365D'] = price_history_df['price'].pct_change().rolling(window=365).std()

    return price_history_df


def percent_supply_profit_data_processing():

    engine = create_engine(f'sqlite:///{DB_FILE_NAME}')

    sth_table_name = "STH_percent_supply_in_profit_7D_MA"
    lth_table_name = "LTH_percent_supply_in_profit_7D_MA"
    
    sth_query = f"SELECT * FROM {sth_table_name}"
    lth_query = f"SELECT * FROM {lth_table_name}"

    sth_percent_supply_profit_df = pd.read_sql(sth_query, engine)
    lth_percent_supply_profit_df = pd.read_sql(lth_query, engine)

    sth_percent_supply_profit_df["STH 7D MA 30D MA"] = sth_percent_supply_profit_df["STH 7D MA"].rolling(window=30).mean()
    sth_percent_supply_profit_df["STH 7D MA 90D MA"] = sth_percent_supply_profit_df["STH 7D MA"].rolling(window=90).mean()
    sth_percent_supply_profit_df["STH 7D MA 365D MA"] = sth_percent_supply_profit_df["STH 7D MA"].rolling(window=365).mean()

    price_history_df = price_history_data_processing()
    sth_percent_supply_profit_df = pd.concat([sth_percent_supply_profit_df, price_history_df[["7D_greater","30D_greater","90D_greater","365D_greater"]]], axis=1)

    return sth_percent_supply_profit_df

def data_exploration_plot_check():

    df = percent_supply_profit_data_processing()
    df["date"] = pd.to_datetime(df["date"])

    plt.plot(df["date"], df["STH 7D MA"])
    plt.plot(df["date"], df["STH 7D MA 30D MA"])
    plt.plot(df["date"], df["STH 7D MA 90D MA"])
    plt.plot(df["date"], df["STH 7D MA 365D MA"])
    plt.grid()
    plt.show()

# def btc_risk_model():

    # Load and preprocess the data
    sth_percent_supply_profit_df = percent_supply_profit_data_processing()

    # Drop non-numeric columns
    sth_percent_supply_profit_df = sth_percent_supply_profit_df.drop(columns=["date"], errors="ignore")

    # Features: Use all rolling averages and raw values
    feature_columns = [col for col in sth_percent_supply_profit_df.columns if "STH 7D MA" in col]

    # Targets: Future price movements (binary classification)
    target_columns = ["7D_greater", "30D_greater", "90D_greater", "365D_greater"]

    models = {}
    for target in target_columns:
        # Drop rows with NaN values (usually first N rows due to rolling averages)
        df_cleaned = sth_percent_supply_profit_df.dropna()

        # Define features (X) and labels (y)
        X = df_cleaned[feature_columns]


        # import seaborn as sns
        # import matplotlib.pyplot as plt

        # corr_matrix = X.corr()
        # plt.figure(figsize=(10,8))
        # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        # plt.title("Feature Correlation Matrix")
        # plt.show()

        y = df_cleaned[target]

        split_idx = int(len(X) * 0.8)  # 80% train, 20% test
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)

        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {target}: {accuracy:.4f}")

        # Cross-validation to check for overfitting
        cv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            cv_scores.append(score)

        print(f"Cross-Validation Accuracy for {target}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        # Store the model for later use
        models[target] = model

        # Print the model feature importances (can be used for further analysis)
        print(f"Feature importances for {target}: {model.feature_importances_}")
        print("\n")
    
    return models


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import os
from datetime import datetime

# Fix Matplotlib backend error
matplotlib.use('Agg')

# Create folders for logs and plots
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Log file setup
log_file = "logs/model_log.txt"
def log_message(message):
    """
    Logs messages to both a file and the console.

    Parameters:
        message (str): The message to log.
    """
    with open(log_file, "a", encoding="utf-8") as log:  # Use UTF-8 encoding
        log.write(f"{datetime.now()} - {message}\n")
    print(message)  # Still prints to console for convenience


def rolling_window_validation(X, y, model, initial_train_size=0.5, step=0.1):
    """Rolling-window validation."""
    n = len(X)
    train_size = int(n * initial_train_size)
    step_size = int(n * step)
    scores = []

    for start in range(0, n - train_size, step_size):
        end = start + train_size
        if end >= n:
            break
        X_train, X_test = X.iloc[:end], X.iloc[end:end + step_size]
        y_train, y_test = y.iloc[:end], y.iloc[end:end + step_size]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
        

    return np.mean(scores), np.std(scores)


def plot_roc_curve(y_test, y_prob, target, save_path):
    """Plots ROC curve with a dotted diagonal line."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc(fpr, tpr):.2f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {target}")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    log_message(f"📂 Saved {save_path}")
    plt.close()


def plot_precision_recall_curve(y_test, y_prob, target, save_path):
    """Plots Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="Precision-Recall", color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve for {target}")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    log_message(f"📂 Saved {save_path}")
    plt.close()


def plot_confusion_matrix(cm, target, save_path):
    """Plots confusion matrix."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix for {target}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path)
    log_message(f"📂 Saved {save_path}")
    plt.close()


def plot_learning_curve(model, X, y, rolling_mean, rolling_std, target, save_path):
    """Plots learning curve with rolling mean and std."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring="accuracy", n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label="Training Accuracy", color="red")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="red", alpha=0.2)
    plt.plot(train_sizes, val_mean, label="Validation Accuracy", color="green")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color="green", alpha=0.2)

    # Add rolling mean and std
    plt.axhline(rolling_mean, color="blue", linestyle="--", label="Rolling-Window Mean")
    plt.fill_between(
        train_sizes, rolling_mean - rolling_std, rolling_mean + rolling_std, color="blue", alpha=0.1, label="Rolling-Window Std"
    )

    plt.title(f"Learning Curve for {target}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    log_message(f"📂 Saved {save_path}")
    plt.close()


def plot_permutation_importance(model, X, y, target, save_path):
    """Plots permutation importance."""
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()

    plt.figure(figsize=(8, 6))
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx], color="purple")
    plt.xlabel("Permutation Importance")
    plt.title(f"Feature Importance for {target}")
    plt.savefig(save_path)
    log_message(f"📂 Saved {save_path}")
    plt.close()


def btc_risk_model(use_smote=True, features=None, timeframes=None):
    """
    Train and evaluate a Bitcoin risk model with flexible features, targets, and options.
    
    Parameters:
        use_smote (bool): Whether to apply SMOTE for oversampling.
        features (list): List of features to use. Defaults to all available.
        timeframes (list): List of timeframes to assess. Defaults to ["7D_greater", "30D_greater", "90D_greater", "365D_greater"].
    """
    log_message("\n--- Starting BTC Risk Model ---")

    # Load and preprocess the data
    sth_percent_supply_profit_df = percent_supply_profit_data_processing()

    # Drop non-numeric columns
    sth_percent_supply_profit_df = sth_percent_supply_profit_df.drop(columns=["date"], errors="ignore")

    # Default features and timeframes
    if features is None:
        features = [col for col in sth_percent_supply_profit_df.columns if "STH" in col]
    if timeframes is None:
        timeframes = ["7D_greater", "30D_greater", "90D_greater", "365D_greater"]

    models = {}

    for target in timeframes:
        log_message(f"\n🔹 Training for Target: {target}")

        # Clean and split data
        df_cleaned = sth_percent_supply_profit_df.dropna()
        X = df_cleaned[features]
        y = df_cleaned[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply SMOTE if enabled
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            log_message("✅ Applied SMOTE")

        # Initialize model
        model = RandomForestClassifier(
            n_estimators=50,         # Reduced number of trees for efficiency
            max_depth=3,             # Limit tree depth to avoid overfitting
            min_samples_split=20,    # Larger splits to improve generalization
            max_features="sqrt",     # Reduced features per split
            class_weight="balanced", # Balance class weights to address class imbalance
            random_state=42,
        )

        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        log_message(f"Accuracy for {target}: {accuracy:.4f}")

        # Rolling-window validation
        rolling_mean, rolling_std = rolling_window_validation(X, y, model)

        log_message(f"📊 Final Rolling-Window Accuracy for {target}: {rolling_mean:.4f} ± {rolling_std:.4f}")

        log_message(f"Confusion Matrix for {target}:\n{conf_matrix}")
        log_message(f"Classification Report for {target}:\n{classification_report(y_test, y_pred)}")

        # Generate and save plots
        plot_confusion_matrix(conf_matrix, target, f"plots/{target}_conf_matrix.png")
        plot_learning_curve(model, X, y, rolling_mean, rolling_std, target, f"plots/{target}_learning_curve.png")
        plot_permutation_importance(model, X_test, y_test, target, f"plots/{target}_perm_importance.png")
        y_prob = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob, target, f"plots/{target}_roc_curve.png")
        plot_precision_recall_curve(y_test, y_prob, target, f"plots/{target}_pr_curve.png")

        # Store the model
        models[target] = model

    log_message("\n--- Model Complete ---")
    return models


# Example of calling the revamped function
btc_risk_model(
    use_smote=False,  # Enable SMOTE
    features=["STH 7D MA", "STH 7D MA 30D MA", "STH 7D MA 90D MA", "STH 7D MA 365D MA"],
    timeframes=["7D_greater"]
)



