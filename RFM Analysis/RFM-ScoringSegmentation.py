import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import rfm_scoring, frequency_score, segment_assignment
df = pd.read_excel("RFM Analysis\Online Retail.xlsx", sheet_name='Online Retail')

# Data Cleaning
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df.drop_duplicates()

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print(snapshot_date)
print(df['InvoiceDate'].min())


# Calculate RFM Metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)


# Apply scoring to each RFM column
rfm['R_Score'] = rfm_scoring(rfm['Recency'], reverse=True).astype(float)
rfm['F_Score'] = frequency_score(rfm['Frequency']).astype(float)
rfm['M_Score'] = rfm_scoring(rfm['Monetary']).astype(float)

rfm['RFM_Score'] = (
    rfm['R_Score'].fillna(0).astype(int).astype(str) +
    rfm['F_Score'].fillna(0).astype(int).astype(str) +
    rfm['M_Score'].fillna(0).astype(int).astype(str)
)

rfm['RFM_Segment'] = rfm['RFM_Score'].apply(segment_assignment)

rfm.reset_index().to_csv('rfm_scoring.csv', index = False)

segment_counts = rfm['RFM_Segment'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(segment_counts.index, segment_counts.values, color='skyblue')

plt.title("Customer Count per Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()