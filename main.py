import onchain_data
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
    ...

data_exploration_plot_check()


