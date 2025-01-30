import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
import chompjs
from sqlalchemy import create_engine


CHAIN_EXPOSED_BASE_URL = "https://chainexposed.com"
DB_FILE_NAME = "btc_onchain.db"

def get_percent_supply_in_profit():

    percent_supply_in_profit_url = f"{CHAIN_EXPOSED_BASE_URL}/XthInProfitPct.html"

    response = requests.get(percent_supply_in_profit_url)
    soup = bs(response.text, "html.parser")
    scripts = soup.find_all('script')
    plot_script_raw = scripts[5].text
    plot_data_dict = list(chompjs.parse_js_objects(plot_script_raw)) 

    STH_data = plot_data_dict[3]
    STH_df = pd.DataFrame(data={"STH 7D MA": STH_data["y"], "date": STH_data["x"]})
    STH_df["date"] = pd.to_datetime(STH_df["date"])
    STH_df["STH 7D MA"] = pd.to_numeric(STH_df["STH 7D MA"])

    LTH_data = plot_data_dict[4]
    LTH_df = pd.DataFrame(data={"LTH 7D MA": LTH_data["y"], "date": LTH_data["x"]})
    LTH_df["date"] = pd.to_datetime(LTH_df["date"])
    LTH_df["LTH 7D MA"] = pd.to_numeric(LTH_df["LTH 7D MA"])

    return STH_df, LTH_df 

def create_btc_onchain_data_db():

    engine = create_engine(f'sqlite:///{DB_FILE_NAME}')

    sth_df, lth_df = get_percent_supply_in_profit()

    sth_df.to_sql("STH_percent_supply_in_profit_7D_MA", con=engine, if_exists='replace', index=False)
    lth_df.to_sql("LTH_percent_supply_in_profit_7D_MA", con=engine, if_exists='replace', index=False)
