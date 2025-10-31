import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from pandasai import SmartDatalake
from pandasai.config import Config
from pandasai_litellm.litellm import LiteLLM
import plotly.express as px

# ------------------------------------------------------------
# 🧠 Setup
# ------------------------------------------------------------
load_dotenv("/Users/navneetasharma/Desktop/Ask Data App New/open_ai_key.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = LiteLLM(model="gpt-4.1-mini", api_key=openai_api_key)
config = Config(llm=llm)

# ------------------------------------------------------------
# 🗄️ Load SQLite data
# ------------------------------------------------------------
db_path = "/Users/navneetasharma/Desktop/Ask Data App New/sales.db"
conn = sqlite3.connect(db_path)

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
tables_dict = {t[0]: pd.read_sql(f"SELECT * FROM {t[0]}", conn) for t in tables.values}

# Normalize column names
for df in tables_dict.values():
    df.columns = [c.strip().lower() for c in df.columns]
    if "sales_value_usd" in df.columns:
        df.rename(columns={"sales_value_usd": "sales_value"}, inplace=True)

# Create SmartDatalake
sdl = SmartDatalake(list(tables_dict.values()), config=config)

# ------------------------------------------------------------
# 🧠 Test Query
# ------------------------------------------------------------
print("🧠 Running test query:")
query = "Compute the average sales value by region."
response = sdl.chat(query)

# ------------------------------------------------------------
# 🧩 Handle DataFrameResponse
# ------------------------------------------------------------
if hasattr(response, "value") and isinstance(response.value, pd.DataFrame):
    df_result = response.value
elif isinstance(response, pd.DataFrame):
    df_result = response
else:
    df_result = None

# ------------------------------------------------------------
# 📈 Visualization
# ------------------------------------------------------------
if df_result is not None:
    print("\n✅ Data received for visualization:")
    print(df_result)

    x_col = df_result.columns[0]
    y_col = df_result.columns[1]

    fig = px.bar(
        df_result,
        x=x_col,
        y=y_col,
        text_auto=True,
        title="Average Sales by Region",
        color=x_col,
    )

    fig.update_layout(
        template="simple_white",
        title_x=0.5,
        font=dict(size=14, color="black"),
        showlegend=False,
        xaxis_title=x_col.capitalize(),
        yaxis_title=y_col.replace("_", " ").capitalize(),
    )

    fig.show()

else:
    print("❌ No DataFrame result to visualize. Response type:", type(response))
    print(response)