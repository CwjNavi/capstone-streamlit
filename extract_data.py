__VERSION__ = "v1.0"
__AUTHOR__ = ["Jaslyn"]

import datetime
import os
from typing import Optional

import clickhouse_connect
import pandas as pd
from dotenv import load_dotenv

# loading variables from .env file
load_dotenv()

class ExtractData:
    def __init__(self):
        self.client = self.establish_connection()
        
    def extract_data_for_prediction_by_ticker(self, ticker, days_into_future):
        df = self.run_query(f"select ticker, tradedate as date, iv10d, iv20d, iv30d, iv60d, iv90d, iv6m, iv1yr, fclshv{days_into_future}d as {days_into_future}_days_future_volatility from volatility_shifted where ticker='{ticker}' and tradedate >= date '2014-11-13' and tradedate <= date '2023-9-29'")
        return df
    
    def extract_data_for_prediction_by_group(self, days_into_future:int, sic_sector:Optional[str]=None, fama_industry:Optional[str]=None, scale_market_cap:Optional[str]=None, start_date:datetime.date=datetime.date(2014, 11, 13), end_date:datetime.date=datetime.date(2023, 9, 29), additional_features=False, sentiment_feature=False, intraday_actual_vol=False):
        query = f"""
            select
                s.ticker as ticker, t.sicsector, t.famaindustry, s.tradedate as date, s.iv10d, s.iv20d, s.iv30d, s.iv60d, s.iv90d, s.iv6m, s.iv1yr,
                {'fclshv' if intraday_actual_vol == False else 'forhv'}{days_into_future}d as {days_into_future}_days_future_volatility
                {',p.volume, f.de, f.epsusd, f.roe, m.ev, m.evebitda, m.marketcap, m.pe, m.pb, m.ps' if additional_features == True else ''}
                {', ads.avg_daily_sentiment' if sentiment_feature == True else ''}
                from volatility_shifted s
                left join ticker_sectors t on s.ticker = t.ticker
                {"left join price_base p on s.ticker = p.ticker and s.tradedate = p.date left join (select * from fundamentals_base where dimension = 'MRQ') f on f.ticker = s.ticker and f.calendardate = s.tradedate left join dailymetrics_base m on m.date = s.tradedate and m.ticker = s.ticker" if additional_features == True else ''}
                {'right join avg_daily_sentiment ads on s.ticker = ads.Ticker and s.tradedate = ads.Date' if sentiment_feature == True else ''}
                where isdelisted=0
                and tradedate >= date '{start_date.strftime('%Y-%m-%d')}' and tradedate <= date '{end_date.strftime('%Y-%m-%d')}'
                and iv{days_into_future}d < 200
                and fclshv{days_into_future}d < 200
            """
        
        if sic_sector is not None:
            query += f"and sicsector in '{sic_sector}'"
            
        if fama_industry is not None:
            query += f"and famaindustry in '{fama_industry}'"
            
        if scale_market_cap is not None:
            query += f"and scalemarketcap in '{scale_market_cap}'"
                    
        df = self.run_query(query=query)
        return df
        
    def extract_ticker_data_from_table(self, columns:list[str], table_name:str, ticker:str, start_date:datetime.date, end_date:datetime.date, date_column_name:str="date") -> pd.DataFrame:
        """
        start_date and end_date are inclusive of the date specified
        """
        df = self.run_query(f"select {', '.join(columns)} FROM {table_name} where Ticker='{ticker}' and {date_column_name} >= date '{start_date.strftime('%Y-%m-%d')}' and {date_column_name} <= '{end_date.strftime('%Y-%m-%d')}'")
        return df
        
    def run_query(self, query:str):
        df = self.client.query_df(query)
        return df
    
    def establish_connection(self):
        # ClickHouse connection
        client = clickhouse_connect.get_client(
            host='l1cqwxg3ce.ap-southeast-1.aws.clickhouse.cloud',
            port=8443,
            username='default',
            password=(os.getenv("clickhouse_password")), # Get the updated password from tele grp
            secure=True
        )
        return client


"""
# ClickHouse table name - CHANGE TO THE CLICKHOUSE TABLE YOU WANT TO EXTRACT
table_name = 'dailymetrics_base'

# Fetch the table that you need
dailymetrics_base = client.query_df(f"SELECT * FROM {table_name}")



#########
# EXAMPLE TEST CODE (BUT REPLACE WITH YOUR PYTHON SCRIPT HERE)
#########
filtered_data = dailymetrics_base[dailymetrics_base['ticker'] == 'AAPL']
#########
########



# Display filtered results
print(filtered_data)
"""

def demo():
    ED = ExtractData()
    df = ED.extract_ticker_data_from_table(table_name="news_headlines", columns=["*"], ticker="AAPL", date_column_name="Date", start_date=datetime.date(2014, 11, 13), end_date=datetime.date(2023, 9, 29))
    print(df)
    df.to_csv('AAPL_news.csv')
    
if __name__ == "__main__":
    demo()