import onchain_data
import sklearn
import pandas as pd
from sqlalchemy import create_engine


DB_FILE_NAME = "btc_onchain.db"

onchain_data.create_btc_onchain_data_db()

# engine = create_engine(f'sqlite:///{DB_FILE_NAME}')

# # example pull from db
# table_name = "STH_percent_supply_in_profit_7D_MA"
# query = f"SELECT * FROM {table_name}"
# df = pd.read_sql(query, engine)

