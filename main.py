import onchain_data
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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

def btc_risk_model():

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
        y = df_cleaned[target]

        # Split data into training and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {target}: {accuracy:.4f}")

        # Cross-validation to check for overfitting
        cv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X, y, cv=cv)
        print(f"Cross-Validation Accuracy for {target}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Store the model for later use
        models[target] = model

        # Print the model feature importances (can be used for further analysis)
        print(f"Feature importances for {target}: {model.feature_importances_}")
        print("\n")
    
    return models


btc_risk_model()


