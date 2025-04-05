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


st.set_page_config(page_title="Relevance Based Prediction")
st.title('JBxSUTD Julius Baer Equity Risk Monitoring')
st.subheader('Liquid Neural Networks')

st.write("🚧 work in progress, check back very soon!")