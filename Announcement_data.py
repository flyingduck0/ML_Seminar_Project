# %%
import pandas as pd

def create_master_timeline():
    # 1. Define the timeframe (2025-2026)
    start_date = '2024-01-01 00:00:00'
    end_date = '2026-12-31 23:00:00'

    # 2. Generate the range in UTC
    hourly_index = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')

    # 3. Create the DataFrame
    df_master = pd.DataFrame(index=hourly_index)
    
    # 4. Reset index and rename column to "Date" as seen in your image
    df_master = df_master.reset_index()
    df_master = df_master.rename(columns={'index': 'Date'})
    
    # 5. Format to match your image (removing the '+00:00' suffix)
    # This keeps the time exactly as is but makes it "naive" for the CSV
    df_master['Date'] = df_master['Date'].dt.tz_localize(None)
    
    return df_master

# Run and save
df_timeline = create_master_timeline()
df_timeline.to_csv('Announcement_data.csv', index=False)

print("File saved as 'Announcement_data.csv' in the format from your image.")
print(df_timeline.head(3))
# %%

# %%
import pandas as pd

# ==========================================
# PHASE 1: THE EVENT REGISTRY (Placeholders)
# ==========================================
# When you get the real dates, just paste them into these lists.
# Format must be 'YYYY-MM-DD'

fomc_dates = ['2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18', '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18', '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-10', '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09']
cpi_dates = ['2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-15', '2024-06-12', '2024-07-11', '2024-08-14', '2024-09-11', '2024-10-10', '2024-11-13', '2024-12-11', '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10', '2025-05-13', '2025-06-11', '2025-07-15', '2025-08-12', '2025-09-11', '2025-10-24', '2025-12-18', '2026-01-13', '2026-02-13', '2026-03-11', '2026-04-10', '2026-05-12', '2026-06-10', '2026-07-14', '2026-08-12', '2026-09-11', '2026-10-14', '2026-11-10', '2026-12-10']
emp_dates = ['2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05', '2024-05-03', '2024-06-07', '2024-07-05', '2024-08-02', '2024-09-06', '2024-10-04', '2024-11-01', '2024-12-06', '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04', '2025-05-02', '2025-06-06', '2025-07-03', '2025-08-01', '2025-09-05', '2025-11-20', '2025-12-16', '2026-01-09', '2026-02-11', '2026-03-06', '2026-04-03', '2026-05-08', '2026-06-05', '2026-07-02', '2026-08-07', '2026-09-04', '2026-10-02', '2026-11-06', '2026-12-04']
gdp_dates = ['2024-01-25', '2024-02-28', '2024-03-28', '2024-04-25', '2024-05-30', '2024-06-27', '2024-07-25', '2024-08-29', '2024-09-26', '2024-10-30', '2024-11-27', '2024-12-19', '2025-01-30', '2025-02-27', '2025-03-27', '2025-04-30', '2025-05-29', '2025-06-26', '2025-07-30', '2025-08-28', '2025-09-25', '2025-12-23', '2026-01-22', '2026-02-20', '2026-03-13', '2026-04-09', '2026-04-30', '2026-05-28', '2026-06-25', '2026-07-30', '2026-08-26', '2026-09-30', '2026-10-29', '2026-11-25', '2026-12-23']

# ==========================================
# PHASE 2: THE RULE ENGINE
# ==========================================
def process_event_dates():
    # 1. Build a raw list combining the dates, event types, and the New York Time hour rules.
    # Notice we use '08:00:00' to snap the 8:30 AM events to the top of the hour bucket.
    raw_events = []
    
    for d in fomc_dates:
        raw_events.append({'date': d, 'event_type': 'FOMC', 'ny_time': '14:00:00'})
        
    for d in cpi_dates:
        raw_events.append({'date': d, 'event_type': 'CPI', 'ny_time': '08:00:00'})
        
    for d in emp_dates:
        raw_events.append({'date': d, 'event_type': 'Employment', 'ny_time': '08:00:00'})
        
    for d in gdp_dates:
        raw_events.append({'date': d, 'event_type': 'GDP', 'ny_time': '08:00:00'})

    # 2. Convert to a Pandas DataFrame
    df_events = pd.DataFrame(raw_events)
    
    # 3. Combine Date and Time into a single string column
    df_events['datetime_str'] = df_events['date'] + ' ' + df_events['ny_time']
    
    # 4. Turn that string into an actual Pandas Datetime object
    df_events['datetime_obj'] = pd.to_datetime(df_events['datetime_str'])
    
    # 5. THE MAGIC: Tell Pandas this is New York time, then convert it to UTC.
    # This automatically calculates Daylight Saving Time for every individual date.
    df_events['datetime_ny'] = df_events['datetime_obj'].dt.tz_localize('America/New_York', ambiguous='infer')
    df_events['timestamp_utc'] = df_events['datetime_ny'].dt.tz_convert('UTC')
    
    # 6. Remove the timezone "+00:00" text so it perfectly matches our Master Timeline format
    df_events['Date'] = df_events['timestamp_utc'].dt.tz_localize(None)
    
    # 7. Clean up: Keep only the two columns we need for the merge
    df_final_events = df_events[['Date', 'event_type']]
    
    return df_final_events

# Run the engine
df_processed_events = process_event_dates()

# Print the results to verify
print("Phase 2 Complete. Here is the processed mini-dataset:\n")
print(df_processed_events)
# %%

# %%
# ==========================================
# PHASE 3: THE FINAL MERGE
# ==========================================

# 1. Merge the Master Timeline with our Processed Events
df_merged = pd.merge(df_timeline, df_processed_events, on='Date', how='left')

# 2. Pivot the 'event_type' into individual columns (Dummies)
# This creates a column for each: FOMC, CPI, Employment, GDP
df_final = pd.get_dummies(df_merged, columns=['event_type'], prefix='Ann')

# 3. Clean up column names (e.g., 'Ann_CPI', 'Ann_FOMC')
# get_dummies puts 0s and 1s, but we need to ensure the hourly rows with NO events stay 0
# Currently, those rows would be all 0s anyway because of the 'left' merge.

# 4. (Optional) If an event appears multiple times in one hour, we sum/max them 
# to keep exactly one row per hour.
df_final = df_final.groupby('Date').max().reset_index()

# Fill any NaNs with 0 (rows where no announcement happened)
df_final = df_final.fillna(0)

# Save the final product
df_final.to_csv('Final_Announcement_Dummies.csv', index=False)

print("\nMISSION ACCOMPLISHED: 'Final_Announcement_Dummies.csv' is ready for the model.")
print(df_final[df_final['Ann_CPI'] == 1].head()) # Sanity check for a CPI event
# %%