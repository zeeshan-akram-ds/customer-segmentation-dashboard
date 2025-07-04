import os
import gdown
# File ID from your Google Drive share link
file_id = "10P54JcuW-SPJYjX_ce623Ql1N7AH78oF"
output_path = "retail data.csv"

# Download if not already present
if not os.path.exists(output_path):
    print("Downloading dataset from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
from time import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from generate_report import build_and_return_pdf
# DATA LOADING & PREPROCESSING
### It has encoding issues, so I specify the encoding to match with its encoding
data = pd.read_csv('retail data.csv', encoding='ISO-8859-1')
print(data.head())
## displaying the shape of the dataset
print(f"Shape of the dataset: {data.shape}")
## Checking for missing values
print(data.isnull().sum())
## Dropping missing where customer ID is null because i can't analyze it without customer ID
data.dropna(subset=['CustomerID'], inplace=True)
## Again checking for shape of the dataset
print(f"Shape of the dataset after dropping missing values: {data.shape}")
## Again checking for missing values
print(data.isnull().sum())
## Lets look at summary statistics
print(data.describe())
print(f"Number of negative values in Quantity: {data['Quantity'][data['Quantity']<0].count()}")
print(f"Number of negative values in UnitPrice: {data['UnitPrice'][data['UnitPrice']<0].count()}")
## Lets drop rows where quantity is less than 0 or unit price is less than 0 or equal to zero
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
## Number of negative values in Quantity and UnitPrice
## Again checking for shape of the dataset
print(f"Shape of the dataset after dropping rows with negative values: {data.shape}")
## Creating a New Revenue Column
data['Revenue'] = data['Quantity'] * data['UnitPrice']
## Converting InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')
## Extracting year, month, and day from InvoiceDate
data['Year'] = data['InvoiceDate'].dt.year
data['Month'] = data['InvoiceDate'].dt.month
data['Day'] = data['InvoiceDate'].dt.day
## Lets preview the dataset after all the preprocessing
print(data.head())
## Lets Handle Cancellations
data['Cancelled'] = data['InvoiceNo'].str.contains('C', na=False)
## Lets check how many cancellations we have
print(f"Number of cancellations: {data['Cancelled'].sum()}")
## Lets drop the cancelled rows
data = data[~data['Cancelled']]
## Again checking for shape of the dataset
print(f"Shape of the dataset after dropping cancelled rows: {data.shape}")
## Preparing for RFM Segmentation
### Getting latest invoice date
latest_date = data['InvoiceDate'].max()
### Calculating Recency, Frequency, and Monetary Value
rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Revenue': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Revenue': 'Monetary'
}).reset_index()
## Displaying the RFM DataFrame
print(rfm.head())
## Lets segment customers using Kmeans algorithm
## Preprocessing for KMeans
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
## Lets print first 5 rows of the scaled data
print(pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary']).head())
##  Lets Plot the Elbow Curve to find the optimal number of clusters
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
## From the elbow curve, I can see that the optimal number of clusters is around 4
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(rfm_scaled)
## Lets add the cluster labels to the RFM DataFrame
rfm['Cluster'] = kmeans.labels_
## Displaying the RFM DataFrame with Cluster labels
print(rfm.head())
## Analyzing the Clusters
cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()
print(cluster_analysis)
## Lets merge cluster label with original data for further analysis
data = data.merge(rfm[['CustomerID', 'Cluster']], on='CustomerID', how='left')
customer_country_map = data.groupby('CustomerID')['Country'].first().reset_index()
rfm = pd.merge(rfm, customer_country_map, on='CustomerID', how='left')
## Lets check sillehoutte score to evaluate the clustering
silhouette_avg = silhouette_score(rfm_scaled, rfm['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.2f}')
cluster_labels = {
    2: "Champions",
    3: "Loyal Customers",
    0: "Potential Loyalists",
    1: "At Risk"
}

rfm['Segment'] = rfm['Cluster'].map(cluster_labels)
data['YearMonth'] = data['InvoiceDate'].dt.to_period('M').astype(str)

def detect_anomalies(monthly_data, column='Revenue', threshold=2.0):
    # Calculate % change
    monthly_data['PctChange'] = monthly_data[column].pct_change()

    # Calculate mean and std dev
    mean_change = monthly_data['PctChange'].mean()
    std_change = monthly_data['PctChange'].std()

    # Set dynamic thresholds
    high_threshold = mean_change + threshold * std_change
    low_threshold = mean_change - threshold * std_change

    # Detect anomalies
    alerts = []
    for i, row in monthly_data.iterrows():
        if row['PctChange'] > high_threshold:
            alerts.append(f"Sudden spike in {column} in **{row['YearMonth'].strftime('%b %Y')}** (+{row['PctChange']*100:.1f}%)")
        elif row['PctChange'] < low_threshold:
            alerts.append(f"Sudden drop in {column} in **{row['YearMonth'].strftime('%b %Y')}** ({row['PctChange']*100:.1f}%)")

    return alerts
## Lets start Dash Dashboard Layout (Foundation)
## Lets create a Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
# Reusable KPI Card
def create_kpi_card(title, value, color):
    return html.Div([
        html.H4(title, style={'color': color, 'marginBottom': '10px'}),
        html.H2(f"{value:,}" if isinstance(value, int) else f"{value}", style={
            'fontWeight': 'bold',
            'fontSize': '28px'
        })
    ], style={
        'border': f'2px solid {color}',
        'borderRadius': '10px',
        'padding': '20px',
        'width': '22%',
        'backgroundColor': '#ffffff',
        'boxShadow': '2px 2px 8px rgba(0,0,0,0.07)',
        'textAlign': 'center',
        'marginBottom': '20px'
    })
# ----------------------

# Layout
app.layout = html.Div([

    html.Div([
        html.H2("Filters", style={'marginBottom': '20px'}),
        
        html.Label("Select Cluster(s)"),
        dcc.Dropdown(
            id='cluster-dropdown',
            options=[
                {'label': cluster_labels[i], 'value': i}
                for i in sorted(rfm['Cluster'].unique())
            ],
            value=rfm['Cluster'].unique().tolist(),
            multi=True,
            placeholder="Select Clusters..."
        ),
        html.Br(),

        html.Label("Select Country(ies)"),
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': c, 'value': c} for c in sorted(data['Country'].unique())],
            value=[],  # All by default
            multi=True,
            placeholder="Select Country...",
            clearable=True
        ),
        html.Label("Select Date Range"),
        dcc.DatePickerRange(
            id='date-range-picker',
            start_date=data['InvoiceDate'].min(),
            end_date=data['InvoiceDate'].max(),
            display_format='YYYY-MM-DD',
            style={'marginBottom': '10px'}
        ),
        html.Br(),
        html.Label("Search Customer ID"),
        dcc.Dropdown(
            id='customer-id-dropdown',
            options=[
                {'label': f"Customer {int(cid)}", 'value': int(cid)}
                for cid in sorted(data['CustomerID'].dropna().unique())
            ],
            # Nothing selected initially
            value=None, 
            placeholder="Search by Customer ID...",
            clearable=True
            ),
        html.Button("Reset", id='reset-button', n_clicks=0, style={
            'marginTop': '10px',
            'backgroundColor': '#dc3545',
            'color': 'white',
            'border': 'none',
            'padding': '8px 14px',
            'borderRadius': '6px',
            'cursor': 'pointer',
            'width': '100%'
        }),
    ], style={
        'width': '17%',
        'display': 'inline-block',
        'verticalAlign': 'top',
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'height': '100vh',
        'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
        'position': 'fixed',
        'left': '0',
        'top': '0'
    }),

    # Main Content (right)
    html.Div([
        dcc.Tabs(id="tabs", value='dashboard', children=[
            dcc.Tab(label='Dashboard View', value='dashboard'),
            dcc.Tab(label='Raw Data View', value='data'),
            dcc.Tab(label='Dashboard Guide', value='guide')
                    ]),
    # Dynamic content goes here
    html.Div(id='tab-content')  
], style={
    'marginLeft': '20%',
    'padding': '20px',
    'height': '100vh',
    'overflowY': 'auto'
})
]) 
@app.callback(
    Output('kpi-container', 'children'),
    Input('cluster-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
)
def update_all_kpis(selected_clusters, selected_countries, start_date, end_date):
    # Use all clusters/countries if none are selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Filter the dataset based on cluster, country, and date range
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # Merge with RFM metrics
    merged = pd.merge(filtered, rfm, on='CustomerID', how='left')

    # KPIs
    total_customers = merged['CustomerID'].nunique()
    total_revenue = round(merged['Revenue'].sum(), 2)
    avg_frequency = round(merged['Frequency'].mean(), 2)
    avg_recency = round(merged['Recency'].mean(), 2)

    # Average Order Value (AOV)
    order_count = merged['InvoiceNo'].nunique()
    aov = round(total_revenue / order_count, 2) if order_count else 0

    # Returning vs. One-Time Customers
    cust_order_counts = merged.groupby('CustomerID')['InvoiceNo'].nunique()
    returning = cust_order_counts[cust_order_counts > 1].count()
    one_time = cust_order_counts[cust_order_counts == 1].count()

    # Return styled KPI cards
    kpis = [
        create_kpi_card("Total Customers", total_customers, "#007bff"),
        create_kpi_card("Total Revenue", f"${total_revenue}", "#28a745"),
        create_kpi_card("Avg Frequency", avg_frequency, "#ffc107"),
        create_kpi_card("Avg Recency (days)", avg_recency, "#dc3545"),
        create_kpi_card("Avg Order Value", f"${aov}", "#6f42c1"),
        create_kpi_card("Returning Customers", returning, "#17a2b8"),
        create_kpi_card("One-Time Customers", one_time, "#fd7e14")
    ]
    
    return kpis
# Add each customer's latest invoice date to the RFM table
rfm = pd.merge(
    rfm,
    data[['CustomerID', 'InvoiceDate']].drop_duplicates(subset='CustomerID'),
    on='CustomerID',
    how='left'
)
# Callback to update RFM scatter plot
@app.callback(
    Output('rfm-scatter', 'figure'),
    Input('cluster-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('customer-id-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
)
def update_rfm_scatter(selected_clusters, selected_countries, selected_customer, start_date, end_date):
    # Use all clusters/countries if none selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Filter RFM based on selected filters
    filtered = rfm[
        (rfm['Cluster'].isin(selected_clusters)) &
        (rfm['Country'].isin(selected_countries)) &
        (rfm['InvoiceDate'] >= start_date) &
        (rfm['InvoiceDate'] <= end_date)
    ]

    # Return placeholder chart if no data matches
    if filtered.empty:
        return px.scatter(title="No Data Available")

    # Create main scatter plot: Recency vs Monetary, colored by Cluster
    fig = px.scatter(
        filtered,
        x='Recency',
        y='Monetary',
        color=filtered['Cluster'].astype(str),
        hover_data={
            'CustomerID': True,
            'Recency': True,
            'Frequency': True,
            'Monetary': ':.2f',
            'Cluster': True
        },
        labels={'color': 'Cluster'},
        title="Customer Segments: Recency vs Monetary",
        template='none',
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    # Layout adjustments
    fig.update_layout(
        height=520,
        margin=dict(t=60, l=80, r=20, b=40),
        legend_title='Cluster'
    )

    # Scatter marker styling
    fig.update_traces(
        marker=dict(size=10, opacity=0.6, line=dict(width=1, color='black'))
    )

    # If a customer is selected, highlight on the chart
    if selected_customer and selected_customer in filtered['CustomerID'].values:
        row = filtered[filtered['CustomerID'] == selected_customer]

        # Add a special trace for the selected customer
        fig.add_trace(
            go.Scatter(
                x=row['Recency'],
                y=row['Monetary'],
                mode='markers+text',
                marker=dict(
                    size=24,
                    color='red',
                    line=dict(width=3, color='black'),
                    symbol='circle-open-dot',
                    opacity=1
                ),
                text=[f"Customer {selected_customer}"],
                textposition='top center',
                name='Selected Customer',
                hovertemplate='Customer ID: %{text}<br>Recency: %{x}<br>Monetary: %{y}<extra></extra>'
            )
        )

        # Zoom around selected customer's data
        fig.update_layout(
            xaxis=dict(range=[row['Recency'].values[0] - 10, row['Recency'].values[0] + 10]),
            yaxis=dict(range=[row['Monetary'].values[0] - 500, row['Monetary'].values[0] + 500])
        )

    return fig
# Callback to update the Monthly Revenue line chart
@app.callback(
    Output('monthly-revenue-line', 'figure'),
    Input('cluster-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
)
def update_monthly_revenue(selected_clusters, selected_countries, start_date, end_date):
    # Use all clusters/countries if none selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Filter main dataset based on user selections
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # Prepare YearMonth column
    filtered = filtered.copy()
    filtered['YearMonth'] = filtered['InvoiceDate'].dt.to_period('M').astype(str)

    # Aggregate revenue per month
    monthly_rev = filtered.groupby('YearMonth')['Revenue'].sum().reset_index()
    monthly_rev['YearMonth'] = pd.to_datetime(monthly_rev['YearMonth'])
    monthly_rev.sort_values('YearMonth', inplace=True)

    # Build line chart
    fig = px.line(
        monthly_rev,
        x='YearMonth',
        y='Revenue',
        markers=True,
        title='Monthly Revenue Trend',
        labels={'Revenue': 'Total Revenue', 'YearMonth': 'Month'},
        template='plotly_white'
    )

    # Style line
    fig.update_traces(line=dict(color='#007bff', width=3))
    fig.update_traces(
        hovertemplate='%{x|%b %Y}<br>Revenue: $%{y:,.2f}',
        mode='lines+markers'
    )

    # Layout tweaks
    fig.update_layout(height=400, margin=dict(t=50, l=20, r=20, b=40))

    return fig
# Callback to update the "Top 10 Countries by Revenue" pie chart
@app.callback(
    Output('country-revenue-pie', 'figure'),
    Input('cluster-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_country_pie(selected_clusters, selected_countries, start_date, end_date):
    # Use all clusters/countries if user hasn't selected any
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Apply filters on the main dataset
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # Calculate total revenue per country and pick top 10
    country_rev = (
        filtered.groupby('Country')['Revenue']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    # Build donut-style pie chart
    fig = px.pie(
        country_rev,
        names='Country',
        values='Revenue',
        title='Top 10 Countries by Revenue',
        # Creates the donut shape
        hole=0.4,  
        color_discrete_sequence=px.colors.sequential.RdBu
    )

    # Style and formatting
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    fig.update_traces(
        hovertemplate="%{label}<br>Revenue: $%{value:,.2f}<br>Share: %{percent}"
    )

    return fig
# Callback to update the "Top 10 Customers by Revenue" bar chart
@app.callback(
    Output('top-customers-bar', 'figure'),
    Input('cluster-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
)
def update_top_customers_bar(selected_clusters, selected_countries, start_date, end_date):
    # Fallback to all clusters if none are selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()

    # Fallback to all countries if none are selected
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Filter dataset based on inputs
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # Aggregate total revenue per customer and select top 10
    top_customers = (
        filtered.groupby('CustomerID')['Revenue']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    # Merge with RFM to attach customer segment labels
    top_customers = pd.merge(
        top_customers,
        rfm[['CustomerID', 'Segment']],
        on='CustomerID',
        how='left'
    )

    # Create horizontal bar chart with color by segment
    fig = px.bar(
        top_customers,
        x='CustomerID',
        y='Revenue',
        color='Segment',
        title='Top 10 Customers by Revenue',
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    # Style the chart layout and hover info
    fig.update_layout(
        template='plotly_white',
        margin=dict(t=40, l=10, r=10, b=40)
    )
    fig.update_traces(
        hovertemplate="Customer ID: %{x}<br>Total Revenue: $%{y:,.2f}"
    )

    return fig
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    
)
def render_tab_content(tab):
    if tab == 'dashboard':
        return html.Div([
            html.H1("Customer Segmentation Dashboard", style={'textAlign': 'center'}),

            html.Div(id='kpi-container', style={
                'display': 'flex',
                'justifyContent': 'space-around',
                'flexWrap': 'wrap',
                'marginBottom': '40px'
            }),

            html.Div([
                html.Button("Download Filtered CSV", id="download-btn", n_clicks=0, style={
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                    'fontSize': '16px'
                }),
                dcc.Download(id="download-dataframe-csv")
            ], style={
                'width': '100%',
                'textAlign': 'right',
                'paddingRight': '30px',
                'marginBottom': '20px',
                'marginTop': '-10px'
            }),
            html.Button("📄 Download PDF Report", id="pdf-download-btn", n_clicks=0, className="btn btn-primary"),
                dcc.Download(id="pdf-download-link"),
            html.Div([
                html.H3("Business Insights"),
                dcc.Markdown(id='insights-box', style={
                    'backgroundColor': '#f1f1f1',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'boxShadow': '1px 1px 5px rgba(0,0,0,0.1)',
                    'fontSize': '16px'
                })
            ], style={'marginTop': '20px', 'marginBottom': '30px'}),

            # All Graphs Below
            html.Div([
                html.Div([dcc.Graph(id='rfm-scatter')], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),
                html.Div([dcc.Graph(id='monthly-revenue-line')], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),
                html.Div([dcc.Graph(id='country-revenue-pie')], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),
                html.Div([dcc.Graph(id='top-customers-bar')], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),
                html.Div([dcc.Graph(id='customer-growth-line')], style={'width': '100%', 'padding': '1%'}),
                html.Div([dcc.Graph(id='revenue-heatmap')], style={'width': '100%', 'padding': '1%'}),
                html.Div([dcc.Graph(id='returning-vs-onetime-bar')], style={'width': '100%', 'padding': '1%', 'marginTop': '20px'}),
                html.Div([dcc.Graph(id='revenue-region-map')], style={'width': '98%', 'padding': '1%'})
            ])
        ])

    elif tab == 'data':
        return html.Div([
            html.H3("Filtered Raw Data"),
            dcc.Loading(
                dcc.Graph(id='raw-data-table'),
                type="circle"
            )
        ])
    elif tab == 'guide':
        return html.Div([
            html.H2("Dashboard Guide", style={'textAlign': 'center', 'marginBottom': '30px'}),

            html.Div([
                html.H4("RFM Scatter Plot"),
                html.P("Shows customer segments by plotting Recency (how recently a customer purchased) against Monetary (how much they spent). Each dot represents a customer colored by their segment such as Champions, Loyal Customers, etc.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Monthly Revenue Line Chart"),
                html.P("Displays total revenue per month. Useful to identify trends, peaks, and revenue drops across time periods.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Customer Growth Trend"),
                html.P("Shows how the number of unique customers changes over time. Helps track business expansion or customer loss.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Top 10 Customers"),
                html.P("Highlights the highest-spending customers. Helps businesses focus on their most valuable clients.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Revenue by Region Map"),
                html.P("An interactive map showing revenue generated from different countries. The darker the color, the higher the revenue.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Monthly Revenue Heatmap"),
                html.P("Shows revenue concentration across countries and months. Helps identify regional seasonality and trends.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Returning vs One-time Customers"),
                html.P("Visualizes customer types by frequency. Helps evaluate retention rate and acquisition success.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Average Order Value (AOV)"),
                html.P("Shows the average purchase value of customers during the selected time range. Helpful to assess overall sales quality.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Customer Search by ID"),
                html.P("Allows you to search a specific customer ID and view their position on the RFM scatter chart. Helps in customer support or targeting.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Download Filtered CSV"),
                html.P("Lets you export the filtered dataset as a CSV file. Useful for external analysis or reporting.")
            ], style={'marginBottom': '25px'}),

            html.Div([
                html.H4("Dynamic Insights & Anomalies"),
                html.P("Automatically generated insights based on customer behavior, top countries, high revenue customers, and unusual revenue spikes or drops. Provides real-time storytelling to decision-makers.")
            ], style={'marginBottom': '25px'})
            
        ], style={'padding': '40px 60px'})
# Callback to display a preview table of the filtered raw data (top 100 rows)
@app.callback(
    Output('raw-data-table', 'figure'),
    Input('cluster-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_raw_table(selected_clusters, selected_countries, start_date, end_date):
    # Use all clusters if none are selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()

    # Use all countries if none are selected
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Apply filters on the full dataset
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # Select relevant columns and limit to top 100 rows for preview
    preview = filtered[['InvoiceNo', 'CustomerID', 'Country', 'Revenue']].head(100)

    # Create a simple table figure using Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(preview.columns),
            fill_color='lightgray',
            align='left'
        ),
        cells=dict(
            values=[preview[col] for col in preview.columns],
            align='left'
        )
    )])

    return fig
# Callback to update the Monthly Unique Customers line chart
@app.callback(
    Output('customer-growth-line', 'figure'),
    Input('cluster-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_customer_growth(selected_clusters, selected_countries, start_date, end_date):
    # Use all clusters if none selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()
    
    # Use all countries if none selected
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Apply filters to main dataset based on selections
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # Group by YearMonth to count unique customers per month
    monthly_growth = (
        filtered.groupby('YearMonth')['CustomerID']
        .nunique()
        .reset_index()
    )

    # Convert YearMonth to proper datetime for plotting
    monthly_growth['YearMonth'] = pd.to_datetime(monthly_growth['YearMonth'])

    # Line chart for visualizing unique customer growth over time
    fig = px.line(
        monthly_growth,
        x='YearMonth',
        y='CustomerID',
        markers=True,
        title='Monthly Unique Customers',
        labels={'CustomerID': 'Unique Customers', 'YearMonth': 'Month'},
        # No default theme to keep custom styling
        template=None  
    )

    # Styling the line and hover text
    fig.update_layout(
        height=400,
        margin=dict(t=50, l=20, r=20, b=40)
    )
    fig.update_traces(
        line=dict(color='#17a2b8', width=3),
        hovertemplate='%{x|%b %Y}<br>Customers: %{y}'
    )

    return fig
# Callback to update the Monthly Revenue Heatmap by Country
@app.callback(
    Output('revenue-heatmap', 'figure'),
    Input('cluster-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_revenue_heatmap(selected_clusters, selected_countries, start_date, end_date):
    # Use all clusters if none selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()
    
    # Use all countries if none selected
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Filter dataset based on selections
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # If no data after filtering, return empty placeholder heatmap
    if filtered.empty:
        return px.imshow(
            [[0]],
            labels=dict(x="Month", y="Country", color="Revenue"),
            title="No Data Available"
        )

    # Extract Year-Month and group revenue by Country and Month
    filtered = filtered.copy()
    filtered['Month'] = filtered['InvoiceDate'].dt.to_period('M').astype(str)
    pivot = filtered.pivot_table(
        index='Country',
        columns='Month',
        values='Revenue',
        aggfunc='sum',
        # Fill missing values with 0
        fill_value=0  
    )

    # Create heatmap using Plotly's imshow
    fig = px.imshow(
        pivot,
        aspect='auto',
        labels=dict(x="Month", y="Country", color="Revenue"),
        color_continuous_scale='Viridis',
        title="Monthly Revenue Heatmap by Country"
    )

    # Adjust layout for better viewing
    fig.update_layout(
        margin=dict(t=40, l=20, r=20, b=40),
        height=500
    )

    return fig
# Callback to reset the selected customer ID when the reset button is clicked
@app.callback(
    Output('customer-id-dropdown', 'value'),     
    Input('reset-button', 'n_clicks'),          
    prevent_initial_call=True                   
)
def reset_customer_selection(n_clicks):
    # Return None to clear the selection in the Customer ID dropdown
    return None
@app.callback(
    Output('returning-vs-onetime-bar', 'figure'),    
    Input('cluster-dropdown', 'value'),               
    Input('country-dropdown', 'value'),               
    Input('date-range-picker', 'start_date'),         
    Input('date-range-picker', 'end_date')            
)
def update_returning_vs_onetime(selected_clusters, selected_countries, start_date, end_date):
    # Use all clusters if none are selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()

    # Use all countries if none are selected
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Apply filters to the original transactional data
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # Group by customer to count unique invoices (orders)
    order_counts = filtered.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()

    # Label customers as 'Returning' if they have more than 1 order, else 'One-Time'
    order_counts['Type'] = order_counts['InvoiceNo'].apply(lambda x: 'Returning' if x > 1 else 'One-Time')

    # Summarize count of each customer type
    summary = order_counts['Type'].value_counts().reset_index()
    summary.columns = ['Customer Type', 'Count']

    # Create bar chart
    fig = px.bar(
        summary,
        x='Customer Type',
        y='Count',
        color='Customer Type',
        title='Returning vs One-Time Customers',
        color_discrete_map={
            'Returning': '#28a745',   
            'One-Time': '#dc3545'     
        }
    )

    # Update layout and hover info
    fig.update_layout(template='plotly_white', height=400, margin=dict(t=50, l=20, r=20, b=40))
    fig.update_traces(
        hovertemplate="Customer Type: %{x}<br>Count: %{y}<extra></extra>"
    )

    return fig
## for consistency in country names
data['Country'] = data['Country'].replace({
    'United Kingdom': 'UK',
    'EIRE': 'Ireland'
})
# Callback to update the choropleth map showing revenue by country
@app.callback(
    Output('revenue-region-map', 'figure'),           
    Input('cluster-dropdown', 'value'),               
    Input('country-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),         
    Input('date-range-picker', 'end_date')           
)
def update_revenue_map(selected_clusters, selected_countries, start_date, end_date):
    # If no clusters are selected, include all clusters
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()
    
    # If no countries are selected, include all countries
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Filter the dataset based on cluster, country, and date range
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # Group data by country to calculate total revenue per country
    country_rev = filtered.groupby('Country')['Revenue'].sum().reset_index()

    # Create a choropleth (map) using Plotly Express
    fig = px.choropleth(
        country_rev,
        locations='Country',                     # Column with country names
        locationmode='country names',            # Using country names instead of ISO codes
        color='Revenue',                         # Filling color based on revenue
        color_continuous_scale='Viridis',        # Using a visually appealing color scale
        title='Revenue by Country (Map)'         # Title of the map
    )

    # Clean up map layout 
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=False),
        height=500,
        margin=dict(t=40, l=10, r=10, b=10)
    )

    # Customize hover tooltip
    fig.update_traces(
        hovertemplate="<b>%{location}</b><br>Revenue: $%{z:,.2f}<extra></extra>"
    )

    return fig
# Callback to allow downloading the filtered dataset as a CSV file
@app.callback(
    Output("download-dataframe-csv", "data"),        
    Input("download-btn", "n_clicks"),              
    State("cluster-dropdown", "value"),            
    State("country-dropdown", "value"),         
    State("date-range-picker", "start_date"),     
    State("date-range-picker", "end_date"),          
    prevent_initial_call=True                      
)
def download_csv(n_clicks, selected_clusters, selected_countries, start_date, end_date):
    # Default to all clusters if none are selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()

    # Default to all countries if none are selected
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Filter data based on selections
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # Return downloadable CSV using Dash utility
    return dcc.send_data_frame(filtered.to_csv, filename="filtered_data.csv", index=False)
@app.callback(
    Output('insights-box', 'children'), 
    Input('cluster-dropdown', 'value'),  
    Input('country-dropdown', 'value'),  
    Input('date-range-picker', 'start_date'), 
    Input('date-range-picker', 'end_date')   
)
def update_insights(selected_clusters, selected_countries, start_date, end_date):
    # Use all clusters if none selected
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()

    # Use all countries if none selected
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Apply filters to the main dataset
    filtered = data[
        (data['Cluster'].isin(selected_clusters)) &
        (data['Country'].isin(selected_countries)) &
        (data['InvoiceDate'] >= start_date) &
        (data['InvoiceDate'] <= end_date)
    ]

    # If no data remains after filtering, show a fallback message
    if filtered.empty:
        return "No data available for the selected filters and date range."

    # Total number of unique customers and total revenue
    total_customers = filtered['CustomerID'].nunique()
    total_revenue = filtered['Revenue'].sum()

    # Best-performing country and customer (by revenue)
    top_country = filtered.groupby('Country')['Revenue'].sum().idxmax()
    top_customer = filtered.groupby('CustomerID')['Revenue'].sum().idxmax()

    # Average revenue per customer
    avg_revenue_per_customer = total_revenue / total_customers

    # Prepare monthly revenue trend
    monthly = filtered.copy()
    monthly['YearMonth'] = monthly['InvoiceDate'].dt.to_period('M').astype(str)
    monthly['YearMonth'] = pd.to_datetime(monthly['YearMonth'])

    monthly_rev = (
        monthly.groupby('YearMonth')['Revenue']
        .sum()
        .reset_index()
        .sort_values('YearMonth')
    )

    # Identify best and worst performing months
    best_month = monthly_rev.loc[monthly_rev['Revenue'].idxmax()]
    worst_month = monthly_rev.loc[monthly_rev['Revenue'].idxmin()]

    # Detect sudden spikes/dips in revenue
    anomalies = detect_anomalies(monthly_rev, column='Revenue', threshold=2.0)
    anomaly_text = '\n\n'.join(anomalies) if anomalies else "No unusual changes in recent revenue."

    # Construct formatted business summary text
    insights = f"""
**Summary for Selected Period**

A total of **{total_customers:,}** customers generated **${total_revenue:,.2f}** in revenue.
The average revenue per customer is **${avg_revenue_per_customer:,.2f}**.

- Top country by revenue: **{top_country}**
- Most valuable customer ID: **{int(top_customer)}**

- Best month: **{best_month['YearMonth'].strftime('%B %Y')}** (${best_month['Revenue']:,.2f})
- Lowest month: **{worst_month['YearMonth'].strftime('%B %Y')}** (${worst_month['Revenue']:,.2f})

**Revenue Change Alerts**
{anomaly_text}
"""

    return insights
from generate_report import build_and_return_pdf
@app.callback(
    Output("pdf-download-link", "data"),
    Input("pdf-download-btn", "n_clicks"),
    State("cluster-dropdown", "value"),
    State("country-dropdown", "value"),
    State("date-range-picker", "start_date"),
    State("date-range-picker", "end_date"),
    prevent_initial_call=True
)
def download_pdf(n_clicks, selected_clusters, selected_countries, start_date, end_date):
    if not selected_clusters:
        selected_clusters = rfm['Cluster'].unique().tolist()
    if not selected_countries:
        selected_countries = data['Country'].unique().tolist()

    # Generate updated insights based on filters
    insights = update_insights(selected_clusters, selected_countries, start_date, end_date)

    # Build & save PDF, get path
    pdf_path = build_and_return_pdf(data, rfm, insights)  # this should return: reports/Customer_Segmentation_Report.pdf

    # Now safely send file
    if os.path.exists(pdf_path):
        return dcc.send_file(pdf_path)
    else:
        raise FileNotFoundError("PDF not found. Something went wrong during generation.")
## run to check if the app is running
if __name__ == '__main__':
    app.run(debug=True)
