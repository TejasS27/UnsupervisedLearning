import pandas as pd
import numpy as np

# RFM Scoring
def rfm_scoring(series, reverse = False):
    try:
        bins = pd.qcut(series, q=5, duplicates='drop')
        number_of_bins = bins.cat.categories.size
        if number_of_bins < 2:
            return pd.Series([np.nan] * len(series))
        labels = list(range(number_of_bins, 0, -1)) if reverse else list(range(1, number_of_bins + 1))
        return pd.qcut(series, q=number_of_bins, labels=labels, duplicates='drop')
    except ValueError as e:
        print(f"Error in scoring {series.name}: {e}")
        return pd.Series([np.nan]*len(series))

def frequency_score(series):
    try:
        # Use ranking to avoid duplicate edge errors
        ranked = series.rank(method='first')
        bins = pd.qcut(ranked, q=5, duplicates='drop')
        num_bins = bins.cat.categories.size
        if num_bins < 2:
            return pd.Series([np.nan] * len(series))
        labels = list(range(1, num_bins + 1))
        return pd.qcut(ranked, q=num_bins, labels=labels, duplicates='drop')
    except ValueError as e:
        print("F_Score error:", e)
        return pd.Series([np.nan] * len(series))


# Segmentation based on RFM score
def segment_assignment(score):
    if score == '555':
        return 'Champions'
    elif score[0] == '5':
        return 'Loyal Customers'
    elif score[1] == '5':
        return 'Frequent Buyers'
    elif score[2] == '5':
        return 'Big Spenders'
    elif score.startswith('1'):
        return 'At Risk'
    elif score.endswith('1'):
        return 'Low Value'
    else:
        return 'Others'
    

def k_means_labeling (row, r_33, r_66, f_33, f_66, m_33, m_66):
    r = row['Recency']
    f = row['Frequency']
    m = row['Monetary']

    if r <= r_33 and f >= f_66 and m >= m_66:
        return 'Champions'
    elif r <= r_66 and m >= m_66:
        return 'Potential Loyalists'
    else:
        return 'Hibernating'

