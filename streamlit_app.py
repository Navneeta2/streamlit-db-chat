import os
import re
import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
# from dotenv import load_dotenv
from pandasai import SmartDatalake
from pandasai.config import Config
from pandasai_litellm.litellm import LiteLLM

# ------------------------------------------------------------
# 🔧 Setup Streamlit UI
# ------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("🤖 Ask Your Database")

# Load environment variables
ENV_PATH = "/Users/navneetasharma/Desktop/Ask Data App New/open_ai_key.env"
# load_dotenv(dotenv_path=ENV_PATH)
openai_api_key = os.getenv("OPENAI_API_KEY") # LiteLLM can use this key for Gemini/GPT

if not openai_api_key:
    st.error("⚠️ OPENAI_API_KEY not found in environment variables. Please set it up.")
    st.stop()

# Define paths (Adjust to your actual path)
# DB_PATH = "/Users/navneetasharma/Desktop/Ask Data App New/sales.db"
DB_PATH = "sales.db"

# ------------------------------------------------------------
# ✅ FIX: Initialize Session State at the Top Level
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
# ------------------------------------------------------------


# --- ADAPTIVE COLUMN SELECTION ---
Y_AXIS_KEYWORDS = {
    'price': ['price', 'cost', 'value', 'unit_price'],
    'quantity': ['quantity', 'count', 'amount'],
    'sales': ['sales', 'revenue', 'income', 'total', 'total_sales_value', 'sales_value', 'aggregated_sales'] 
}

def find_best_y_column(df, query, numeric_cols):
    """Dynamically selects the best Y-axis column based on the query."""
    lower_query = query.lower()
    
    for keyword, variations in Y_AXIS_KEYWORDS.items():
        if any(v in lower_query for v in variations):
            for col in numeric_cols:
                if any(v in col.lower() for v in variations):
                    return col
    
    sales_candidates = [col for col in numeric_cols if any(s in col.lower() for s in ['sales', 'value', 'total'])]
    if sales_candidates:
        return sorted(sales_candidates, key=len)[0] 
        
    return numeric_cols[0] if numeric_cols else None

# ------------------------------------------------------------
# 🗄️ Cached Data Loading and SmartDatalake Initialization
# ------------------------------------------------------------

@st.cache_data(show_spinner="Connecting to DB and Loading Tables...")
def load_data_and_create_sdl(db_path, openai_key):
    """Loads data, preprocesses it, and creates the SmartDatalake instances."""
    try:
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        table_names = [t[0] for t in tables.values]

        tables_dict = {}
        for table in table_names:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Standardize sales column names 
            if "sales_value_usd" in df.columns:
                df.rename(columns={"sales_value_usd": "sales_value"}, inplace=True)
            elif "total_price" in df.columns:
                 df.rename(columns={"total_price": "sales_value"}, inplace=True)
                 
            df.reset_index(drop=True, inplace=True)
            tables_dict[table] = df
            
        conn.close() 
        
        # Use a powerful model for reasoning and code generation
        llm = LiteLLM(model="openai/gpt-4o", api_key=openai_key) 
        
        # SDL for structured DataFrame output (Simulates generate_data_code tool)
        config_df = Config(llm=llm, save_charts=False, output_type='dataframe')
        sdl_df = SmartDatalake(list(tables_dict.values()), config=config_df)
        
        # SDL for unstructured text output (Simulates generate_observation tool)
        config_text = Config(llm=llm, save_charts=False, output_type='text')
        sdl_text = SmartDatalake(list(tables_dict.values()), config=config_text)
        
        return tables_dict, table_names, sdl_df, sdl_text 

    except Exception as e:
        st.error(f"⚠️ Failed to load data or connect to DB: {e}")
        return {}, [], None, None

tables_dict, table_names, sdl_df, sdl_text = load_data_and_create_sdl(DB_PATH, openai_api_key)

# ------------------------------------------------------------
# 💬 Main App Logic (Chat Interface)
# ------------------------------------------------------------

if sdl_df and sdl_text:
    
    # 2. Sidebar and Table Preview
    with st.sidebar:
        st.header("📋 Available Tables")
        st.write(table_names)
        
        selected_table = st.selectbox("👁️ Preview a table:", table_names)
        if selected_table in tables_dict:
            st.write(f"### Preview of `{selected_table}`", tables_dict[selected_table].head())
            
    # 3. Display Past Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                 st.markdown(message["content"])
            if "df" in message:
                st.write("**Raw Data Table:**")
                st.dataframe(message["df"], use_container_width=True)
            if "fig" in message:
                st.write("**Visual Chart:**")
                st.pyplot(message["fig"])
    
    # 4. Handle New User Input
    if query := st.chat_input("Ask a question about your data (e.g., 'top products by quantity')..."):
        
        # A. Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # B. Start Assistant's Response
        with st.chat_message("assistant"):
            status = st.empty() 
            status.info("🤔 **Reasoning:** Calling `generate_data_code` tool...")
            
            # Temporary object to collect results before appending to history
            assistant_response = {"role": "assistant", "content": f"**Query:** *{query}*"}
            
            try:
                # ------------------------------------------------------------
                # 🛠️ TOOL 1: generate_data_code (Executed by sdl_df)
                # ------------------------------------------------------------
                prompt_df = f"""
                You are a data analyst assistant. The user asked: "{query}"

                # IMPORTANT CONTEXT:
                # 1. Date columns are named 'order_date' or 'ship_date' and are in YYYY-MM-DD format.
                # 2. Sales figures are always in the column named 'sales_value'.
                # 3. Always prioritize the 'transactions' table for sales queries.
                
                **Strict Output Requirements:**
                1. Always return a single, pre-aggregated pandas DataFrame.
                2. The primary categorical column (e.g., 'product_name', 'month', 'region') must be the **FIRST** column.
                3. The primary numeric value column (e.g., the metric being requested like price or sales) must be the **SECOND** column.
                4. Rename the final aggregated value column to be descriptive, e.g., 'aggregated_sales'.
                5. Do not include any index column.
                
                Do not describe or explain the result — only return the pandas DataFrame object.
                """

                response_df = sdl_df.chat(prompt_df) 

                df_result = None
                if hasattr(response_df, "value") and isinstance(response_df.value, pd.DataFrame):
                    df_result = response_df.value
                elif isinstance(response_df, pd.DataFrame):
                    df_result = response_df
                
                # ------------------------------------------------------------
                # 🛠️ TOOL 2: generate_observation (Executed by sdl_text)
                # ------------------------------------------------------------
                if df_result is not None and not df_result.empty:
                    
                    status.info("🧠 **Analysis:** Calling `generate_observation` tool for insights...")
                    
                    commentary_prompt = f"""
                    Provide 1-2 concise, professional observations or key insights related to the user's original query: "{query}" based on the aggregated data. Do not use numbering or bullet points.
                    
                    Data Table:\n{df_result.to_markdown(index=False)}
                    
                    **Strict requirement:** Only return the observation text. Do not include introductory phrases.
                    """
                    
                    commentary_response = sdl_text.chat(commentary_prompt)
                    
                    commentary_text = ""
                    if isinstance(commentary_response, str):
                        commentary_text = commentary_response
                    elif hasattr(commentary_response, 'value') and isinstance(commentary_response.value, str):
                        commentary_text = commentary_response.value

                    clean_commentary = re.sub(r'[\r\n]+', ' ', commentary_text).strip()
                    
                    st.markdown(f"**Insight:** {clean_commentary}")
                    assistant_response["content"] = f"**Insight:** {clean_commentary}" 
                    
                    
                    # ------------------------------------------------------------
                    # 📈 Visualization (Post-Tool Execution)
                    # ------------------------------------------------------------
                    status.info("📈 **Visualization:** Generating chart...")
                    
                    st.write("**Raw Data Table:**")
                    st.dataframe(df_result, use_container_width=True) 
                    assistant_response["df"] = df_result 

                    # --- Axis Selection Logic (for plotting) ---
                    datetime_cols = [col for col in df_result.columns if 'date' in col.lower() or 'month' in col.lower() or 'year' in col.lower()]
                    
                    for col in datetime_cols:
                        try:
                            df_result[col] = pd.to_datetime(df_result[col], errors='coerce', infer_datetime_format=True)
                            df_result.dropna(subset=[col], inplace=True)
                        except Exception:
                            pass 
                    
                    numeric_cols = df_result.select_dtypes(include="number").columns.tolist()
                    y_col = find_best_y_column(df_result, query, numeric_cols)
                    x_col = None
                    chart_type = None

                    lower_q = query.lower()
                    is_trend_query = any(w in lower_q for w in ["trend", "over time", "growth", "monthly", "daily", "last"])
                    
                    if is_trend_query and datetime_cols and y_col:
                        chart_type = "line"
                        x_col = datetime_cols[0]
                        df_result.sort_values(by=x_col, inplace=True)
                        
                    elif y_col and df_result.columns.size >= 2:
                        chart_type = "bar"
                        x_col = df_result.columns[0] if df_result.columns[0] != y_col else df_result.columns[1]

                    # --- Matplotlib Plotting ---
                    if x_col and y_col:
                        st.write("**Visual Chart:**")

                        x_title = x_col.replace('_', ' ').title()
                        y_title = y_col.replace('_', ' ').title()
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        df_plot = df_result.set_index(x_col)
                        
                        if chart_type == "line":
                            df_plot[y_col].plot(kind='line', ax=ax)
                        else:
                            df_plot_sorted = df_plot.sort_values(by=y_col, ascending=False).head(20) 
                            df_plot_sorted[y_col].plot(kind='bar', ax=ax, color='#4c78a8')
                            ax.tick_params(axis='x', rotation=45, labelsize=8) 
                        
                        ax.set_title(query.title(), fontsize=12)
                        ax.set_xlabel(x_title, fontsize=10)
                        ax.set_ylabel(y_title, fontsize=10)
                        ax.set_ylim(bottom=0)
                        plt.tight_layout()

                        st.pyplot(fig)
                        assistant_response["fig"] = fig
                        plt.close(fig) 

                        st.markdown(f"*Chart shows data using **{x_title}** on the X-axis and **{y_title}** on the Y-axis.*")
                    else:
                        st.warning("⚠️ Could not find suitable X and Y axis columns in the resulting DataFrame for visualization.")
                        
                else:
                    error_msg = f"❌ The AI did not return a valid DataFrame for the query. Please refine your question."
                    if isinstance(response_df, str):
                        error_msg += f" (Received text: {response_df[:100]}...)"
                    st.error(error_msg)
                    assistant_response["content"] = f"**Error:** {error_msg}"

            except Exception as err:
                error_msg = f"⚠️ An unexpected application error occurred during processing: {err}"
                st.error(error_msg)
                assistant_response["content"] = f"**Error:** {error_msg}"
            
            # Remove the processing status message and replace with completion
            status.empty()
            st.success("Analysis complete!")

            # Append the full assistant response object to the session state history
            st.session_state.messages.append(assistant_response)