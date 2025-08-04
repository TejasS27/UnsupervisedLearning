# 🧠 Customer Segmentation using RFM Analysis & K-Means Clustering

This project applies data-driven segmentation on retail customers using **RFM Analysis** (Recency, Frequency, Monetary) and **K-Means Clustering**. It helps marketers identify valuable customer groups and tailor strategies accordingly.

---

## 📌 Project Objectives

- Perform RFM (Recency, Frequency, Monetary) analysis on retail transaction data
- Generate RFM scores to rank customer behavior
- Apply K-Means Clustering to segment customers
- Auto-label clusters into meaningful groups (e.g., Champions, At Risk, Loyal Customers)
- Visualize customer distribution and behavior by segment

---

## 📁 Dataset

- **Source**: Online Retail dataset (UCI ML Repository / Kaggle)
- **Records**: ~500,000 transaction rows
- **Fields Used**: `CustomerID`, `InvoiceDate`, `InvoiceNo`, `Quantity`, `UnitPrice`

> 💡 *Only transactions with valid `CustomerID` and positive quantity/price were used.*

---

## 🧪 Techniques Used

- RFM feature engineering
- Quantile-based RFM scoring
- Feature scaling with `StandardScaler`
- Elbow method to find optimal `k`
- K-Means clustering
- Segment labeling using dynamic quantile thresholds
- Data visualization with `matplotlib` and `seaborn`

---

## 🧱 RFM Model

| Metric    | Definition                              |
|-----------|------------------------------------------|
| Recency   | Days since last purchase                 |
| Frequency | Number of purchases                      |
| Monetary  | Total money spent                        |

---

## 🧩 Customer Segments

| Segment             | Description                                   |
|---------------------|-----------------------------------------------|
| Champions           | Recent, frequent, and high-spending customers |
| Potential Loyalists | Active recently with good monetary value      |
| Hibernating         | Low frequency and low value customers         |

---

## 📊 Visualizations

- Bar chart of customer count by segment
- Average R, F, M values per segment
- Elbow plot to determine optimal `k`

---

## 🛠️ Installation

1. Clone the repo
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-rfm-kmeans.git
   cd customer-segmentation-rfm-kmeans
