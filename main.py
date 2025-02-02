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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
import os

# Fix Matplotlib backend error
matplotlib.use('TkAgg')

# Create a folder to save plots
os.makedirs("plots", exist_ok=True)

def rolling_window_validation(X, y, model, initial_train_size=0.5, step=0.1):
    """Rolling-window validation where the training set expands over time."""
    n = len(X)
    train_size = int(n * initial_train_size)
    step_size = int(n * step)
    
    scores = []
    
    for start in range(0, n - train_size, step_size):
        end = start + train_size
        if end >= n:
            break  # Stop if we reach the end
        
        X_train, X_test = X.iloc[:end], X.iloc[end:end+step_size]
        y_train, y_test = y.iloc[:end], y.iloc[end:end+step_size]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
    
    return np.mean(scores), np.std(scores)

def btc_risk_model():
    # Load and preprocess the data
    sth_percent_supply_profit_df = percent_supply_profit_data_processing()
    
    # Drop non-numeric columns
    sth_percent_supply_profit_df = sth_percent_supply_profit_df.drop(columns=["date"], errors="ignore")
    
    # Features
    feature_columns = [col for col in sth_percent_supply_profit_df.columns if "STH 7D MA" in col]
    target_columns = ["7D_greater", "30D_greater", "90D_greater", "365D_greater"]
    
    models = {}

    # Reduce global font size
    plt.rcParams.update({'font.size': 8})

    # Create separate figures for each metric in a **2×2 grid**
    fig_learning, axes_learning = plt.subplots(2, 2, figsize=(10, 8))
    fig_conf_matrix, axes_conf_matrix = plt.subplots(2, 2, figsize=(10, 8))
    fig_roc, axes_roc = plt.subplots(2, 2, figsize=(10, 8))
    fig_pr, axes_pr = plt.subplots(2, 2, figsize=(10, 8))
    fig_perm_importance, axes_perm_importance = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten axes for easy indexing
    axes_learning = axes_learning.flatten()
    axes_conf_matrix = axes_conf_matrix.flatten()
    axes_roc = axes_roc.flatten()
    axes_pr = axes_pr.flatten()
    axes_perm_importance = axes_perm_importance.flatten()

    for i, target in enumerate(target_columns):
        df_cleaned = sth_percent_supply_profit_df.dropna()
        X = df_cleaned[feature_columns]
        y = df_cleaned[target]

        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Model initialization
        model = RandomForestClassifier(    
            n_estimators=50,
            max_depth=3,
            min_samples_split=20,
            max_features="sqrt",
            class_weight="balanced",  # Automatically balances class weights
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🔹 Accuracy for {target}: {accuracy:.4f}")

        # **Rolling-Window Validation**
        mean_acc, std_acc = rolling_window_validation(X, y, model)
        print(f"📊 Rolling-Window Accuracy for {target}: {mean_acc:.4f} ± {std_acc:.4f}")

        # Store model
        models[target] = model

        # **Feature Importance**
        importance = model.feature_importances_
        print(f"🛠 Feature Importances for {target}: {importance}\n")

        # **Learning Curve with Standard Deviation**
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, scoring="accuracy", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        axes_learning[i].plot(train_sizes, train_mean, 'o-', color="r", label="Train")
        axes_learning[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        axes_learning[i].plot(train_sizes, test_mean, 'o-', color="g", label="Validation")
        axes_learning[i].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        axes_learning[i].set_title(f"Learning Curve: {target}")
        axes_learning[i].legend()

        # **Confusion Matrix**
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes_conf_matrix[i])
        axes_conf_matrix[i].set_title(f"Confusion Matrix: {target}")

        # **ROC Curve**
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes_roc[i].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
        axes_roc[i].plot([0, 1], [0, 1], linestyle='--', color='gray')
        axes_roc[i].set_title(f"ROC Curve: {target}")

        # **Precision-Recall Curve**
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        axes_pr[i].plot(recall, precision, marker='.', color="b")
        axes_pr[i].set_title(f"Precision-Recall: {target}")

        # **Permutation Importance**
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        sorted_idx = result.importances_mean.argsort()
        axes_perm_importance[i].barh(np.array(feature_columns)[sorted_idx], result.importances_mean[sorted_idx], color='purple')
        axes_perm_importance[i].set_title(f"Permutation Importance: {target}")

    # Save and display plots
    for fig, name in zip([fig_learning, fig_conf_matrix, fig_roc, fig_pr, fig_perm_importance],
                         ["learning_curve", "conf_matrix", "roc_curve", "pr_curve", "perm_importance"]):
        fig.tight_layout(pad=2.0, h_pad=0.8, w_pad=0.8)
        fig.savefig(f"plots/{name}.png")  # Save plot
        print(f"📂 Saved {name}.png in 'plots' folder")

    plt.show()

    return models



btc_risk_model()


