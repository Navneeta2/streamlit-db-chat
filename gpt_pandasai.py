import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------

load_dotenv(dotenv_path="/Users/navneetasharma/Desktop/Ask Data App New/open_ai_key.env")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OpenAI API key not found. Please set it in your .env file.")

# -----------------------------
# 2️⃣ Connect to SQLite database
# -----------------------------
db_path = "/Users/navneetasharma/Desktop/Ask Data App New/sales.db"

if not os.path.exists(db_path):
    raise FileNotFoundError(f"❌ Database not found at: {db_path}")

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM sales", conn)
conn.close()

print(f"✅ Loaded {len(df)} rows from {db_path}")

# -----------------------------
# 3️⃣ Initialize LLM and SmartDataframe
# -----------------------------
llm = LiteLLM(model="gpt-4o-mini", api_key=api_key)
sdf = SmartDataframe(df, config={"llm": llm})

# -----------------------------
# 4️⃣ Ask questions to GPT
# -----------------------------
while True:
    question = input("\n💬 Ask a question about your data (or type 'exit' to quit): ").strip()
    if question.lower() in ["exit", "quit"]:
        print("👋 Exiting. Goodbye!")
        break

    try:
        response = sdf.chat(question)
        print("\n🧠 Response:")
        print(response)
    except Exception as e:
        print(f"\n❌ Error: {e}")