import os
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use('Agg') 
from fpdf import FPDF
import kaleido
import re
def clean_markdown(text):
    """
    Removes markdown formatting (like **bold** and bullets) for plain-text PDF rendering.
    """
    # Remove **bold** formatting
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    
    # Remove bullet points (-)
    text = re.sub(r"^\s*-\s*", "", text, flags=re.MULTILINE)
    
    return text
def save_charts(data, rfm):
    import plotly.express as px
    import os

    os.makedirs("charts", exist_ok=True)

    # 1. Monthly Revenue Trend
    monthly = data.copy()
    monthly['YearMonth'] = monthly['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_rev = monthly.groupby('YearMonth')['Revenue'].sum().reset_index()
    fig1 = px.line(monthly_rev, x='YearMonth', y='Revenue', title="Monthly Revenue")
    fig1.write_image("charts/monthly_revenue.png", width=1000, height=500)

    # 2. Top Customers
    top_customers = data.groupby('CustomerID')['Revenue'].sum().nlargest(10).reset_index()
    fig2 = px.bar(top_customers, x='CustomerID', y='Revenue', title="Top 10 Customers by Revenue")
    fig2.write_image("charts/top_customers.png", width=1000, height=500)

    # 3. RFM Scatter Plot
    fig3 = px.scatter(rfm, x='Recency', y='Monetary', color=rfm['Segment'],
                      hover_data=['CustomerID', 'Frequency'], title="Customer Segmentation (RFM)")
    fig3.write_image("charts/rfm_scatter.png", width=1000, height=500)

    # 4. Revenue by Country
    country_rev = data.groupby('Country')['Revenue'].sum().nlargest(10).reset_index()
    fig4 = px.bar(country_rev, x='Country', y='Revenue', title="Top Countries by Revenue")
    fig4.write_image("charts/country_revenue.png", width=1000, height=500)

    # 5. Revenue Heatmap
    heatmap_data = data.copy()
    heatmap_data['YearMonth'] = heatmap_data['InvoiceDate'].dt.to_period('M').astype(str)
    pivot = heatmap_data.pivot_table(index='Country', columns='YearMonth', values='Revenue', aggfunc='sum', fill_value=0)

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, cmap='Blues', linewidths=0.5)
    plt.title("Monthly Revenue Heatmap by Country")
    plt.tight_layout()
    plt.savefig("charts/revenue_heatmap.png")
    plt.close()
# Generating PDF
class PDFReport(FPDF):
    def header(self):
        if self.page_no() == 1:
            # No header on cover
            return  
        self.set_font("Arial", "B", 12)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, "Customer Segmentation Report", ln=True, align='C')
        self.ln(5)

    def footer(self):
        if self.page_no() == 1:
            # No footer on cover
            return  
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align='C')

    def cover_page(self):
        self.add_page()
        self.set_font("Arial", "B", 24)
        self.set_text_color(0, 102, 204)
        # Spacer
        self.cell(0, 80, "", ln=True)  
        self.cell(0, 10, "Customer Segmentation Dashboard", ln=True, align='C')
        self.ln(10)
        self.set_font("Arial", "I", 14)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, "Generated Report using RFM Analysis", ln=True, align='C')

    def section_title(self, title):
        self.set_font("Arial", "B", 14)
        self.set_text_color(0, 102, 204)
        self.ln(8)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def section_body(self, text):
        self.set_font("Arial", "", 11)
        self.set_text_color(0)
        self.multi_cell(0, 7, text)
        self.ln(2)

    def add_image_section(self, path, title):
        self.add_page()
        self.section_title(title)
        self.image(path, w=180)

def generate_pdf_report(insights_text):
    pdf = PDFReport()
    
    # Cover Page
    pdf.cover_page()

    # Objective & Dataset
    pdf.add_page()
    pdf.section_title("1. Objective & Problem Statement")
    pdf.section_body(
        "The primary goal of this dashboard is to segment customers using RFM methodology (Recency, Frequency, "
        "Monetary), enabling the business to better understand customer behavior, target marketing campaigns, "
        "and boost customer retention strategies. We also monitor revenue trends, customer growth, and top contributors."
    )

    pdf.section_title("2. Dataset Description")
    pdf.section_body(
        "This dataset comes from an online retail store. It includes transactions such as InvoiceNo, StockCode, "
        "Description, Quantity, UnitPrice, InvoiceDate, CustomerID, Country, etc. Additional features such as "
        "Revenue, RFM Scores, and Clusters were engineered to support advanced analytics."
    )

    pdf.section_title("3. Methodology")
    pdf.section_body(
        "We used RFM analysis to group customers into clusters like Champions, Loyal, At Risk, etc. Based on these, "
        "KPIs such as AOV (Average Order Value), Revenue by Region, and Customer Growth were derived. Dash visualizations "
        "were then built to make this data actionable."
    )

    # Insights
    pdf.add_page()
    pdf.section_title("4. Executive Summary (Insights)")
    pdf.section_body(clean_markdown(insights_text))

    # Visualizations
    chart_titles = {
        "monthly_revenue.png": "Monthly Revenue Trend",
        "top_customers.png": "Top 10 Customers by Revenue",
        "rfm_scatter.png": "Customer Segments (RFM)",
        "country_revenue.png": "Revenue by Country",
        "revenue_heatmap.png": "Monthly Revenue Heatmap by Country"
    }

    for file, title in chart_titles.items():
        pdf.add_image_section(os.path.join("charts", file), title)

    # Saving
    output_path = "reports/Customer_Segmentation_Report.pdf"
    os.makedirs("reports", exist_ok=True)
    pdf.output(output_path)
    print(f"PDF saved to {output_path}")
    return output_path
def build_and_return_pdf(data, rfm, insights_text):
    save_charts(data, rfm)
    return generate_pdf_report(insights_text)
