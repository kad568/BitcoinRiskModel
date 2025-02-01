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

# def rolling_window_validation(X, y, model, initial_train_size=0.5, step=0.1):
#     """
#     Rolling-window validation where the training set expands over time.
    
#     Parameters:
#         X (DataFrame): Features
#         y (Series): Target
#         model: ML model to evaluate
#         initial_train_size (float): Initial fraction of data for training (default=50%)
#         step (float): Fractional increase of train set in each iteration (default=10%)
    
#     Returns:
#         List of accuracy scores for each step
#     """
#     n = len(X)
#     train_size = int(n * initial_train_size)
#     step_size = int(n * step)
    
#     scores = []
    
#     for start in range(0, n - train_size, step_size):
#         end = start + train_size
#         if end >= n:
#             break  # Stop if we reach the end
        
#         X_train, X_test = X.iloc[:end], X.iloc[end:end+step_size]
#         y_train, y_test = y.iloc[:end], y.iloc[end:end+step_size]
        
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
        
#         acc = accuracy_score(y_test, y_pred)
#         scores.append(acc)
    
#     return np.mean(scores), np.std(scores)

# def btc_risk_model():
#     # Load and preprocess the data
#     sth_percent_supply_profit_df = percent_supply_profit_data_processing()
    
#     # Drop non-numeric columns
#     sth_percent_supply_profit_df = sth_percent_supply_profit_df.drop(columns=["date"], errors="ignore")
    
#     # Features: Use all rolling averages and raw values
#     feature_columns = [col for col in sth_percent_supply_profit_df.columns if "STH 7D MA" in col]
    
#     # Targets: Future price movements (binary classification)
#     target_columns = ["7D_greater", "30D_greater", "90D_greater", "365D_greater"]
    
#     models = {}
#     for target in target_columns:
#         df_cleaned = sth_percent_supply_profit_df.dropna()
#         X = df_cleaned[feature_columns]
#         y = df_cleaned[target]

#         # Train-test split
#         split_idx = int(len(X) * 0.8)
#         X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#         y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

#         # Initialize and train the model
#         model = RandomForestClassifier(n_estimators=100, 
#                                max_depth=5,  # Limit depth to avoid overfitting
#                                min_samples_split=10,  # Prevent overfitting on small subsets
#                                max_features="sqrt",  # Reduce feature selection per tree
#                                random_state=42,
#                                min_samples_leaf=5)

#         model.fit(X_train, y_train)

#         # Evaluate the model on the test set
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f"Accuracy for {target}: {accuracy:.4f}")

#         # Rolling-window validation
#         mean_acc, std_acc = rolling_window_validation(X, y, model)
#         print(f"Rolling-Window Accuracy for {target}: {mean_acc:.4f} ± {std_acc:.4f}")

#         # Store the model for later use
#         models[target] = model

#         # Print feature importances
#         print(f"Feature importances for {target}: {model.feature_importances_}\n")
    
#     return models


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, TimeSeriesSplit

import pandas as pd
from sqlalchemy import create_engine

df = percent_supply_profit_data_processing()

# Define feature and target columns
feature_columns = ["STH 7D MA", "STH 7D MA 30D MA", "STH 7D MA 90D MA"]
target_columns = ["7D_greater", "30D_greater", "90D_greater",]

results = {}

for target in target_columns:
    # Drop NaN rows
    df_cleaned = df.dropna()
    
    # Features and target
    X = df_cleaned[feature_columns]
    y = df_cleaned[target]

    # Train-test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Initialize and train the model
    model = RandomForestClassifier(
    n_estimators=200,          # More trees to capture patterns
    max_depth=5,               # Slightly deeper trees for more flexibility
    min_samples_split=20,      # Higher value to prevent overfitting
    min_samples_leaf=15,       # Prevent small, overly specific leaf nodes
    max_features="sqrt",       # Keep feature selection optimized
    random_state=42,
    class_weight="balanced"    # Handle potential class imbalance
)

    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importances
    feature_importance = model.feature_importances_

    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Store results
    results[target] = {
        "accuracy": accuracy,
        "feature_importance": feature_importance,
        "train_sizes": train_sizes,
        "train_scores": train_scores.mean(axis=1),
        "test_scores": test_scores.mean(axis=1),
    }

# Plot learning curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, target in zip(axes.flatten(), target_columns):
    train_sizes = results[target]["train_sizes"]
    train_scores = results[target]["train_scores"]
    test_scores = results[target]["test_scores"]
    
    ax.plot(train_sizes, train_scores, 'o-', label="Training score")
    ax.plot(train_sizes, test_scores, 'o-', label="Validation score")
    ax.set_title(f"Learning Curve for {target}")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.legend()
    
plt.tight_layout()
plt.show()

# Plot feature importances
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, target in zip(axes.flatten(), target_columns):
    importance = results[target]["feature_importance"]
    ax.bar(feature_columns, importance)
    ax.set_title(f"Feature Importance for {target}")
    ax.set_xticklabels(feature_columns, rotation=45)

plt.tight_layout()
plt.show()


# btc_risk_model()


