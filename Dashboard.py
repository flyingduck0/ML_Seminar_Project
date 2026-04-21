import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page to wide mode
st.set_page_config(page_title="Market Insights", layout="wide")

st.title("📈 Market Analysis Dashboard")

# 1. Check if file exists
file_name = "FEATURES_PREPARED.csv"

if not os.path.exists(file_name):
    st.error(f"❌ File not found: {file_name}")
    st.info("Make sure the CSV is in the same folder as this script.")
else:
    # 2. Load Data
    try:
        df = pd.read_csv(file_name)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 3. Identify Columns
        market_cols = [c for c in df.columns if "_chg" in c]
        macro_cols = [c for c in df.columns if "_DELTA" in c and "missing" not in c]
        
        st.sidebar.header("Controls")
        selected_assets = st.sidebar.multiselect(
            "Select Assets", 
            options=market_cols, 
            default=["BTC_chg", "SPY_chg", "Gold_chg"]
        )

        if not selected_assets:
            st.warning("Please select at least one asset in the sidebar.")
        else:
            # 4. Calculate Cumulative Performance
            # We fill NaNs with 0 so they don't break the math
            plot_df = df[['Date']].copy()
            for col in selected_assets:
                # Math: (1 + change).cumprod() 
                # This shows growth starting from 1.0
                plot_df[col] = (1 + df[col].fillna(0)).cumprod()

            # 5. Create the Chart
            fig = px.line(
                plot_df, 
                x='Date', 
                y=selected_assets,
                title="Asset Performance (Relative Growth)",
                template="plotly_dark"
            )
            
            fig.update_layout(
                hovermode="x unified",
                yaxis_title="Growth Factor (1.0 = Start)",
                legend_title="Asset"
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # 6. Show Macro Events
            st.subheader("Latest Macro Event Values")
            # Only show rows where something actually happened
            event_df = df[df[macro_cols].sum(axis=1) != 0].tail(10)
            if not event_df.empty:
                st.dataframe(event_df[['Date'] + macro_cols])
            else:
                st.write("No macro events detected in the recent data.")

    except Exception as e:
        st.error(f"⚠️ App Crash: {e}")