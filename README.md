# Interactive Customer Segmentation & Retention Dashboard

---

## Objective / Problem Statement

In the evolving retail landscape, understanding customer behavior is critical for retention and business growth. The primary goal of this project is to analyze customer transaction data, segment customers using RFM analysis, and create a professional, highly interactive dashboard to aid non-technical stakeholders in making data-driven decisions. This dashboard is designed to visually represent customer insights such as revenue, engagement, and retention using advanced techniques such as clustering, trend analysis, and anomaly detection.

---

## Purpose

* Perform RFM (Recency, Frequency, Monetary) analysis on historical retail data.
* Cluster customers into meaningful segments using KMeans clustering.
* Build an interactive and professional dashboard that presents visual analytics in a comprehensible way for both technical and non-technical users.
* Enhance business intelligence with features such as anomaly detection, customer search, segment comparison, revenue mapping, and downloadable insights.

---

## Dataset Description

* **Source:** Online Retail Dataset (UCI / Kaggle)
* **File:** `retail_data.csv`
* **Attributes:**
    * `InvoiceNo`: Unique invoice number
    * `StockCode`: Product code
    * `Description`: Product description
    * `Quantity`: Number of products purchased
    * `InvoiceDate`: Date and time of purchase
    * `UnitPrice`: Price per item
    * `CustomerID`: Unique customer ID
    * `Country`: Country of the customer

---

## Live Dashboard

You can interact with the fully functional Customer Segmentation Dashboard here:

**[Launch Live App on Replit](https://76185987-9975-4670-998e-c6a95ed53207-00-3ds4baow1nkz1.pike.replit.dev/)**  
*(Note: It may take a few seconds to load if the server is temporarily asleep.)* 

*(Due to hosting limitations, PDF report export works in the local version only.)*
## Tools & Technologies Used

* **Language:** Python
* **Libraries:** Pandas, NumPy, Plotly, Dash, Scikit-learn, DateTime
* **Dashboard Framework:** Dash by Plotly
* **Clustering Algorithm:** KMeans with Silhouette Score Evaluation
* **RFM Analysis:** Manual calculation of Recency, Frequency, Monetary values

---

## Steps Taken

### 1. Data Cleaning & Preprocessing

* Removed null and duplicate entries.
* Converted `InvoiceDate` to datetime.
* Filtered cancelled transactions.
* Created a `Revenue` column as `Quantity * UnitPrice`.
* Handled missing `CustomerID` values.

### 2. Feature Engineering

* Generated RFM features:
    * **Recency:** Days since last purchase
    * **Frequency:** Number of transactions
    * **Monetary:** Total revenue per customer
* Normalized RFM values using MinMaxScaler.

### 3. Customer Segmentation

* Applied KMeans clustering on scaled RFM data.
* Used Silhouette Score to determine optimal number of clusters.
* Created customer segments like "Champions", "At Risk", etc.

### 4. Dashboard Development

* Built a full-featured, clean Dash dashboard using multiple components:
    * **Sidebar Filters:** Clusters, Country, Customer ID, Reset
    * **Main KPIs:** Total Customers, Total Revenue, Avg Recency/Frequency/Monetary
    * **Visualizations:**
        * RFM Scatter Plot with segment coloring
        * Monthly Revenue Trends (Line Plot)
        * Top 10 Customers by Revenue (Bar Chart)
        * Revenue by Region (Map)
        * Monthly Unique Customers (Line Plot)
        * Revenue Heatmap by Country
        * Returning vs One-Time Customers (Bar Chart)
    * **Interactive Features:**
        * Customer Search
        * CSV Download (Filtered Data)
        * Anomaly Detection in Revenue
        * Dynamic Business Insights Box

### 5. Anomaly Detection

* Detected unusual revenue spikes/drops using rolling averages and standard deviation thresholds.
* Automatically added alerts to the insight box.

### 6. PDF Export (Planned)

* Will capture charts as images and generate a downloadable summary report with all KPIs and trends using fpdf or reportlab.

---


## Features

* Fully interactive filtering by cluster, country, customer ID, and date range.
* Highly professional layout with auto-handling of HTML/CSS.
* Each chart is supported by clear tooltips.
* Dashboard supports both technical analysis and non-technical storytelling.
* Modular and well-structured code for maintainability.

---

## Testing & Improvements

* Tested on multiple filters to ensure no-crash behavior.
* Added hovertemplate, color maps, and chart legends.
* Resolved layout issues with sidebar overlap.
* Improved label clarity and value separation.

---

## Future Work

* Add customer lifetime value prediction.
* Connect with live data sources.
---

## Author

Zeeshan Akram – [LinkedIn](https://www.linkedin.com/in/zeeshan-akram-ds/) | GitHub: [@zeeshan-akram-ds](https://github.com/zeeshan-akram-ds)

---

## Folder Structure
├── app.py
├── retail_data.csv
├── README.md
├── generate_report.py
├── requirements.txt
├── screenshots

---

This project demonstrates a blend of machine learning, dashboard engineering, business analytics, and storytelling – all aimed at real-world decision-making in customer-centric industries.
