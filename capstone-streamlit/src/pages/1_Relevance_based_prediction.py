__VERSION__ = "v1.0"
__AUTHOR__ = ["Ivan"]

import streamlit as st
import datetime
import os
import clickhouse_connect
import pandas as pd
from extract_data import ExtractData as ED
from relevance_based_prediction.evaluate_rbp import EvaluateRBP
from streamlit_scripts.evaluate_rbp import EvaluateRBP as ST_EvaluateRBP
from streamlit_scripts.black_scholes import BlackScholes
from dotenv import load_dotenv
from streamlit_echarts import st_echarts
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Relevance Based Prediction")


if "options_price_90" not in st.session_state:
    st.session_state.options_price_90 = None
if "options_price_10" not in st.session_state:
    st.session_state.options_price_10 = None
if "volatility_90" not in st.session_state:
    st.session_state.volatility_90 = None
if "volatility_10" not in st.session_state:
    st.session_state.volatility_10 = None

if "extracted_df" not in st.session_state:
    st.session_state.extracted_df = None
if 'predicted_df' not in st.session_state:
    st.session_state.predicted_df = None
if "line_graph_options" not in st.session_state:
    st.session_state.line_graph_options = None
if 'dates_for_graph' not in st.session_state:
    st.session_state.dates_for_graph = None

if 'blackscholes_prediction_row' not in st.session_state:
    st.session_state.blackscholes_prediction_row = pd.DataFrame()

histogram_event = None
line_graph_event = None

CLICKHOUSE_PASSWORD = os.getenv("clickhouse_password")
if not os.getenv("clickhouse_password"):
    CLICKHOUSE_PASSWORD = st.secrets["clickhouse_password"]

client = clickhouse_connect.get_client(
            host='l1cqwxg3ce.ap-southeast-1.aws.clickhouse.cloud',
            port=8443,
            username='default',
            password=(CLICKHOUSE_PASSWORD), # Get the updated password from tele grp
            secure=True
        )

st.title('JBxSUTD Julius Baer Equity Risk Monitoring')
st.subheader('Relevance Based Prediction')

all_tickers_list = ED().run_query(query="SELECT DISTINCT ticker FROM ticker_sectors")
all_tickers_list = all_tickers_list['ticker'].tolist()
all_tickers_list = sorted(all_tickers_list)
all_tickers_list.insert(0, 'CCRN')
ticker = st.selectbox('Stock Tickers', all_tickers_list, help="choose a stock ticker ya dick")
print(ticker)

start_date = st.date_input('Data start date: ', datetime.date(2014,11,13))
end_date = st.date_input('Data end date: ', datetime.date(2023,9,29))

ER = EvaluateRBP()
streamlit_ER = ST_EvaluateRBP()

#df = init_df()
days_into_future = 60

@st.cache_data
def extract_data(days_into_future, start_date, end_date, fama_industry, scale_market_cap, ticker):
    # ticker is just here to ensure that the data is re-extracted when ticker changes
    df = ED().extract_data_for_prediction_by_group(days_into_future=days_into_future, fama_industry=fama_industry, scale_market_cap=scale_market_cap, start_date=datetime.date(2014,11,13), end_date=datetime.date(2023,9,29), additional_features=True)
    df["iv_error"] = df[f"iv{days_into_future}d"] - df[f"{days_into_future}_days_future_volatility"]
    print("extracted data: ", df)

    return df

response_column="iv_error"

folder_path = "relevance_based_prediction/analysis/business_services_small_cap"
#predictor_columns = df.drop(columns=["date", "ticker", "lastupdated", "ev","evebit","evebitda","marketcap", "open", "high", "low", "closeadj", "closeunadj", "10 days future volatility", "20 days future volatility", "30 days future volatility", "DJIA", "SP500"]).columns.to_list()
predictor_columns = ['iv10d', 'iv20d', 'iv30d', 'iv60d', 'iv90d', 'iv6m', 'volume', 'evebitda','m.marketcap', 'm.pb', 'm.pe']
max_lookback = pd.Timedelta(days=120)
r_threshold=0

def run_prediction():
    ticker_sector_df = ED().run_query(query=f"SELECT * FROM ticker_sectors WHERE ticker='{ticker}'")
    scale_market_cap = ticker_sector_df['scalemarketcap'].iloc[0] # CCRN '3 - Small'
    fama_industry = ticker_sector_df['famaindustry'].iloc[0] # "Business Services"
    print(scale_market_cap, fama_industry)

    df = extract_data(days_into_future, start_date, end_date, fama_industry, scale_market_cap, ticker)
    st.session_state['extracted_df'] = df
    predicted_vs_actual_df = predict(df=df, ticker=ticker)

    predicted_vs_actual_df['Upper_CI'] = predicted_vs_actual_df['rbp_prediction'] + predicted_vs_actual_df['margin of error']
    predicted_vs_actual_df['Lower_CI'] = predicted_vs_actual_df['rbp_prediction'] - predicted_vs_actual_df['margin of error']
    st.session_state['predicted_df'] = predicted_vs_actual_df

    dates_for_graph = predicted_vs_actual_df['date'].to_list()
    dates_for_graph = [str(date) for date in dates_for_graph]
    st.session_state.dates_for_graph = dates_for_graph

    actual_list = predicted_vs_actual_df["60_days_future_volatility"].to_list()
    predicted_list = predicted_vs_actual_df["rbp_prediction"].to_list()

    upper_CI_list = predicted_vs_actual_df["Upper_CI"].to_list()
    lower_CI_list = predicted_vs_actual_df["Lower_CI"].to_list()
    band_fill = [u - l for u, l in zip(upper_CI_list, lower_CI_list)]


    st.session_state["line_graph_options"] = {
        "title": {"text": "Relevance Based Prediction"},
        "tooltip": {
            "trigger": "axis",
        },
        "legend": {
            "top": 30,
            "left": "left",
            "data": ["Actual", "Predicted", "Confidence Interval"]
            },
        "grid": {
            "top": 80,
        },
        "xAxis": {"data": dates_for_graph},
        "yAxis": {},
        "series": [
            {
                "name": "Actual",
                "type": "line",
                "data": actual_list,
                "color": "#008000"
            },
            {
                "name": "Predicted",
                "type": "line",
                "data": predicted_list,
                "color": "#FF0000"
            },
            # Invisible baseline
            {
                "name": "Lower Bound",
                "type": "line",
                "data": lower_CI_list,
                "lineStyle": {"opacity": 0},
                "stack": "confidence-band",
                "symbol": "none"
            },
            # Visible shaded band
            {
                "name": "Confidence Interval",
                "type": "line",
                "data": band_fill,
                "lineStyle": {"opacity": 0},
                "areaStyle": {
                    "color": "#ccc",
                    "opacity": 0.4
                },
                "stack": "confidence-band",
                "symbol": "none",
                "tooltip": {"show": False}
            },
            # Transparent upper bound line for tooltip
            {
                "name": "Upper Confidence Bound",
                "type": "line",
                "data": upper_CI_list,
                "lineStyle": {"opacity": 0},
                "symbol": "none",
                "tooltip": {"show": True}
            }
        ]
    }


@st.cache_data
def predict(df, ticker):
    predicted_vs_actual_df = streamlit_ER.get_predicted_vs_actual_df(df=df, 
                                                        ticker=ticker, 
                                                        ticker_column="ticker", 
                                                        predictor_columns=predictor_columns, 
                                                        date_column="date", 
                                                        testing_size=0.02, 
                                                        days_into_future=days_into_future, 
                                                        max_lookback=pd.Timedelta(days=120), 
                                                        model="RBP", 
                                                        response_column=response_column, 
                                                        normal_experiment=True, 
                                                        r_threshold=0, 
                                                        folder_path=folder_path)

    predicted_vs_actual_df['rbp_prediction'] = predicted_vs_actual_df['iv60d'] - predicted_vs_actual_df['prediction']
    
    return predicted_vs_actual_df

# if 'predicted_df' not in st.session_state:
#     st.session_state['predicted_df'] = predict(df, ticker)


st.button("Predict", key="predict_button", on_click=run_prediction, type="primary")
# if 'predicted_df' in st.session_state: ## uncomment to see the dataframe
#     st.dataframe(st.session_state['predicted_df'])

# Create the chart and capture clicked point
line_graph_options = st.session_state.line_graph_options
if line_graph_options:
    line_graph_event = st_echarts(
        options=line_graph_options,
        height="400px",
        key="chart",
        events={"click": "function(params) { return params.dataIndex; }"}  # Capture click
    )

@st.cache_data
def show_news(date=datetime.datetime(2014,11,13)):
    if type(date) == str:
        date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    start_date = date - datetime.timedelta(7)
    end_date = date + datetime.timedelta(7)
    news_df = ED().extract_ticker_data_from_table(['Ticker', 'Date', 'Headline', 'Text'], 'news_headlines', ticker, start_date, end_date, 'Date')
    st.dataframe(news_df)


## histogram component
selected_points = None
histogram_options = None
norm_df = None
histogram_bin_labels = None

@st.cache_data
def get_histogram(response_column, df=st.session_state['extracted_df'], ticker=ticker, predictor_columns=predictor_columns, ticker_column='ticker', date_column='date',  prediction_date=datetime.datetime(2014,11,13), days_into_future=60):
    if df.empty:
        return
    
    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d %H:%M:%S")

    histogram_options, subset_df, bin_labels = streamlit_ER.get_pyplot_iv_errors(df=df, 
                                            ticker=ticker, 
                                            predictor_columns=predictor_columns, 
                                            ticker_column=ticker_column, 
                                            date_column=date_column, 
                                            lookback=max_lookback, 
                                            adj_threshold=r_threshold, 
                                            prediction_date=prediction_date, 
                                            days_into_future=days_into_future, 
                                            response_column=response_column)    
    
    return histogram_options, subset_df, bin_labels

# Display the clicked point data
if line_graph_event:
    clicked_date = st.session_state.dates_for_graph[line_graph_event]
    st.write(f"Chosen Date: {clicked_date}")
    show_news(clicked_date)
    histogram_options, subset_df, bin_labels = get_histogram(response_column, prediction_date=clicked_date, days_into_future=days_into_future)
    norm_df = subset_df
    histogram_bin_labels = bin_labels


if histogram_options:
    histogram_event = st_echarts(options=histogram_options,
                                 height="400px",
                                 key="histogram",
                                 events={"click": "function(params) { return params.dataIndex; }"}# Capture click
                                 )
    
if histogram_event:
    lower = float(histogram_bin_labels[histogram_event])
    upper = float(histogram_bin_labels[histogram_event+1])
    selected_norm_df = norm_df.loc[(norm_df['iv_error'] >= lower) & (norm_df['iv_error'] <= upper)]
    st.subheader(f"Points in the bin {lower} - {upper}")
    st.dataframe(selected_norm_df)


st.subheader("Black Scholes")
date_today = st.date_input("Today's date", datetime.date(2023,8,1)) # datetime.date
option_strike_price = st.number_input('Option Strike Price (USD)', min_value=0.0, max_value=1000000.0, value=100.0, step=0.01)
dividend_rate = st.number_input('Dividend Rate (%)', value=5)

time_to_maturity = st.number_input('Time to Maturity (days)', min_value=0, max_value=365, value=60, step=1)

option_type = st.selectbox('Option Type', ['put', 'call'])

rfr_query = f"""
        SELECT GS10
    FROM macro_base
    WHERE GS10 IS NOT NULL AND GS10 != 0
    AND DATE <= '{date_today}'
    ORDER BY DATE DESC
    LIMIT 1
    """ 
risk_free_rate = ED().run_query(query=rfr_query)
risk_free_rate = risk_free_rate['GS10'].iloc[0]
st.number_input(label='Risk Free Rate (%)', value=risk_free_rate, disabled=True)

stock_price_query = f"""
        SELECT close
    FROM price_base
    WHERE close IS NOT NULL AND close != 0
    AND ticker = '{ticker}'
    AND date <= '{date_today}'
    ORDER BY date DESC
    LIMIT 1
    """ 
stock_price = ED().run_query(query=stock_price_query)
stock_price = stock_price['close'].iloc[0]

st.number_input(label='Stock Price (USD)', value=stock_price, disabled=True)
st.number_input(label='Volatility 90% CI (%)', value=st.session_state.volatility_90, disabled=True)
st.number_input(label='Volatility 10% CI (%)', value=st.session_state.volatility_10, disabled=True)

date_of_maturity = date_today + datetime.timedelta(time_to_maturity)

def find_options_price(option_type=option_type, stock_price=stock_price, risk_free_rate=risk_free_rate, time_to_maturity=time_to_maturity, dividend_rate=dividend_rate, option_strike_price=option_strike_price):
    predicted_vs_actual_df = st.session_state['predicted_df']
    prediction_row = predicted_vs_actual_df.loc[predicted_vs_actual_df['date'] == pd.to_datetime(date_today)]
    
    st.session_state.blackscholes_prediction_row = prediction_row
    if prediction_row.empty:
        st.write("No options price prediction found")
        volatility_lower_CI = None
        volatility_upper_CI = None
    else:
        volatility_lower_CI = (prediction_row['rbp_prediction'] - prediction_row['margin of error']).iloc[0]
        volatility_upper_CI = (prediction_row['rbp_prediction'] + prediction_row['margin of error']).iloc[0]
    
    if volatility_lower_CI == None:
        return
    
    st.session_state.volatility_90 = volatility_upper_CI
    st.session_state.volatility_10 = volatility_lower_CI

    blackScholes = BlackScholes()
    time_to_maturity /= 365
    volatility_lower_CI /= 100
    volatility_upper_CI /= 100
    risk_free_rate /= 100
    dividend_rate /= 100

    print('option type', option_type, '\n', 
        'stock price', stock_price, '\n',
        'option strike price', option_strike_price, '\n', 
        'risk free rate', risk_free_rate, '\n', 
        'dividend rate', dividend_rate, '\n', 
        'volatility lower ci', volatility_lower_CI, '\n', 
        'volatility upper ci', volatility_upper_CI, '\n', 
        'time to maturity', time_to_maturity, '\n')

    st.session_state.options_price_90 = blackScholes.calculate_premium(options_type=option_type, St=stock_price, X=option_strike_price, rf=risk_free_rate, q=dividend_rate, vol=volatility_upper_CI, T=time_to_maturity)
    st.session_state.options_price_10 = blackScholes.calculate_premium(options_type=option_type, St=stock_price, X=option_strike_price, rf=risk_free_rate, q=dividend_rate, vol=volatility_lower_CI, T=time_to_maturity)
    print('90%', st.session_state.options_price_90)
    print('10%', st.session_state.options_price_10)

def find_options_price_wrapper():
    find_options_price(option_type=option_type, 
                       stock_price=stock_price, 
                       risk_free_rate=risk_free_rate, 
                       time_to_maturity=time_to_maturity, 
                       dividend_rate=dividend_rate, 
                       option_strike_price=option_strike_price)

st.button("Get Options Price", key="options_price_button", on_click=find_options_price_wrapper)


if st.session_state.options_price_90 is not None and st.session_state.options_price_10 is not None:
    st.write(f'90 percent quartile options price: {st.session_state.options_price_90:.4f}')
    st.write(f'10 percent quartile options price: {st.session_state.options_price_10:.4f}')
elif st.session_state.blackscholes_prediction_row.empty:
    st.write(f"No option price prediction for {date_of_maturity}")
else:
    st.write('No prediction made yet')
    
