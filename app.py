__VERSION__ = "v1.0"
__AUTHOR__ = ["Ivan"]

import streamlit as st
import datetime
import os
import clickhouse_connect
import pandas as pd
from extract_data import ExtractData as ED
from relevance_based_prediction.evaluate_rbp import EvaluateRBP as ER
from extract_data import ExtractData as ED
from dotenv import load_dotenv

# client = clickhouse_connect.get_client(
#             host='l1cqwxg3ce.ap-southeast-1.aws.clickhouse.cloud',
#             port=8443,
#             username='default',
#             password=(os.getenv("clickhouse_password")), # Get the updated password from tele grp
#             secure=True
#         )

st.set_page_config(page_title='JBxSUTD Julius Baer Equity Risk Monitoring')

st.title('JBxSUTD Julius Baer Equity Risk Monitoring')

# chosen_ticker = st.selectbox('Stock Tickers', ['AAPL', 'TSLA'])
# chosen_model = st.selectbox('Model', ['RBP', 'LNN'])
# start_date = st.date_input('Data start date: ')
# end_date = st.date_input('Data end date: ')

# option_strike_price = st.number_input('Option Strike Price', min_value=0.0, max_value=1000000.0, value=100.0, step=0.01)
# time_to_maturity = st.number_input('Time to Maturity (days)', min_value=0, max_value=365, value=60, step=1)


# ER = ER()
# # df = init_df()
# ticker = "KELYA"
# days_into_future = 60

# # df = ED().extract_data_for_prediction_by_group(days_into_future=60, fama_industry="Business Services", start_date=datetime.date(2014,11,13), end_date=datetime.date(2023,9,29))
# # df.to_csv("streamlit_df.csv")
# df = pd.read_csv("streamlit_df.csv")
# df["iv_error"] = df[f"iv60d"] - df[f"60_days_future_volatility"]
# print("df from extract_data_for_prediction_by_group", df)

# response_column="iv_error"

# #predictor_columns = df.drop(columns=["date", "ticker", "lastupdated", "ev","evebit","evebitda","marketcap", "open", "high", "low", "closeadj", "closeunadj", "10 days future volatility", "20 days future volatility", "30 days future volatility", "DJIA", "SP500"]).columns.to_list()
# predictor_columns = ['iv10d', 'iv20d', 'iv30d', 'iv60d', 'iv90d', 'iv6m']

# ## save this so that we don't have to wait for the model to run again
# predicted_vs_actual_df = ER.get_predicted_vs_actual_df(df=df, ticker=ticker, ticker_column="ticker", predictor_columns=predictor_columns, date_column="date", testing_size=0.25, days_into_future=60, model="RBP", response_column=response_column)
# predicted_vs_actual_df = pd.read_csv("streamlit_predicted_vs_actual_df.csv")

# st.dataframe(predicted_vs_actual_df)