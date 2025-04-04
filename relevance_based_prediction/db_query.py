import pandas as pd

import clickhouse_connect

# Connect to ClickHouse Cloud
client = clickhouse_connect.get_client(
    host='d9vqysi32x.ap-southeast-1.aws.clickhouse.cloud',  # Replace with your ClickHouse Cloud host
    port=8443,                          # HTTPS port for ClickHouse Cloud
    username='default',                 # Replace with your ClickHouse username
    password='q30x4MabSKA~M',           # Replace with your ClickHouse password
    secure=True                         # Enable SSL/TLS for a secure connection
)

result = client.query("Select * from daily_metrics")
print(pd.DataFrame(result))