import pandas as pd

# Original file ka sahi path daal (apne computer ke hisaab se)
input_file = r'C:\Users\sutha\OneDrive\Desktop\UAC-Care-Load-Analytics\attachments\HHS_Unaccompanied_Alien_Children_Program (1).csv'

print("Reading file from:", input_file)

df = pd.read_csv(input_file)

# Cleaning
df = df.dropna(subset=['Date']).copy()
df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y')
df['Children in HHS Care'] = df['Children in HHS Care'].str.replace(',', '').astype(int)

numeric_cols = ['Children apprehended and placed in CBP custody*', 
                'Children in CBP custody', 
                'Children transferred out of CBP custody', 
                'Children discharged from HHS Care']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.sort_values('Date').reset_index(drop=True)

# New metrics
df['Total_System_Load'] = df['Children in CBP custody'] + df['Children in HHS Care']
df['Net_Intake'] = df['Children transferred out of CBP custody'] - df['Children discharged from HHS Care']
df['Daily_Growth_Rate'] = df['Total_System_Load'].pct_change() * 100
df['7d_Rolling_HHS'] = df['Children in HHS Care'].rolling(7, min_periods=1).mean()

# Save
df.to_csv('../data/cleaned_uac_data.csv', index=False)
df.to_excel('../data/cleaned_uac_data.xlsx', index=False)

print("✅ Cleaned files saved successfully in data folder!")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())