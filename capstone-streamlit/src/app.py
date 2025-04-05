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

# Title with emoji for a touch of branding flair
st.title("📈 JB x SUTD: Julius Baer Equity Risk Monitoring")

# Stylish project team section
st.markdown("""
### 💼 Project Team S37
**Ivan Chan** · **Kwa Yu Liang** · **Loy Xing Jun**  
**Timothy Wee** · **Terry Yeo** · **Jaslyn Yee**
""")

# Optional: Horizontal rule for visual separation
st.markdown("---")
