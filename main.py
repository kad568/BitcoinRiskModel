import onchain_data
import sklearn
import pandas as pd
from sqlalchemy import create_engine


DB_FILE_NAME = "btc_onchain.db"

# onchain_data.create_btc_onchain_data_db()

# price history data preprtation
engine = create_engine(f'sqlite:///{DB_FILE_NAME}')

table_name = "BTC_price_history"
query = f"SELECT * FROM {table_name}"
price_history_df = pd.read_sql(query, engine)

# Create shifted price columns
price_history_df["price greater in 7D"] = price_history_df["price"].shift(-7)
price_history_df["price greater in 30D"] = price_history_df["price"].shift(-30)
price_history_df["price greater in 90D"] = price_history_df["price"].shift(-90)
price_history_df["price greater in 365D"] = price_history_df["price"].shift(-365)

# Calculate ROI for each time range
price_history_df['7D_ROI'] = (price_history_df['price greater in 7D'] - price_history_df['price']) / price_history_df['price']
price_history_df['30D_ROI'] = (price_history_df['price greater in 30D'] - price_history_df['price']) / price_history_df['price']
price_history_df['90D_ROI'] = (price_history_df['price greater in 90D'] - price_history_df['price']) / price_history_df['price']
price_history_df['365D_ROI'] = (price_history_df['price greater in 365D'] - price_history_df['price']) / price_history_df['price']

# Create binary columns with NaN where future data is not available
price_history_df['7D_greater'] = (price_history_df['price'] < price_history_df['price greater in 7D']).where(price_history_df['price greater in 7D'].notna(), pd.NA).astype('Int64')
price_history_df['30D_greater'] = (price_history_df['price'] < price_history_df['price greater in 30D']).where(price_history_df['price greater in 30D'].notna(), pd.NA).astype('Int64')
price_history_df['90D_greater'] = (price_history_df['price'] < price_history_df['price greater in 90D']).where(price_history_df['price greater in 90D'].notna(), pd.NA).astype('Int64')
price_history_df['365D_greater'] = (price_history_df['price'] < price_history_df['price greater in 365D']).where(price_history_df['price greater in 365D'].notna(), pd.NA).astype('Int64')


# rolling standard deviation (volatility)
price_history_df['volatility_7D'] = price_history_df['price'].pct_change().rolling(window=7).std()
price_history_df['volatility_30D'] = price_history_df['price'].pct_change().rolling(window=30).std()
price_history_df['volatility_90D'] = price_history_df['price'].pct_change().rolling(window=90).std()
price_history_df['volatility_365D'] = price_history_df['price'].pct_change().rolling(window=365).std()


# add price performance to indicators df