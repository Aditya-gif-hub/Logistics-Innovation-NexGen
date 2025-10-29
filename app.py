import streamlit as st
import pandas as pd

import io # Needed for download button
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Set page layout to wide
st.set_page_config(layout="wide")

# --- 1. Data Loading and Caching ---
@st.cache_data
def load_data():
    # Load all relevant datasets
    try:
        orders_df = pd.read_csv("orders.csv")
        delivery_df = pd.read_csv("delivery_performance.csv")
        routes_df = pd.read_csv("routes_distance.csv")
        cost_df = pd.read_csv("cost_breakdown.csv")
        
        # --- Data Merging ---
        data = pd.merge(orders_df, cost_df, on="Order ID", how="left")
        data = pd.merge(data, delivery_df[["Order ID", "Carrier", "Customer Rating", "Delivery Status"]], on="Order ID", how="left")
        
        # Use a subset of routes_df columns for a cleaner merge
        routes_subset = routes_df[["Order ID", "Distance (km)", "Fuel Consumption (L)"]]
        data = pd.merge(data, routes_subset, on="Order ID", how="left")
        
        # --- Feature Engineering (for Analysis) ---
        cost_cols = ['Fuel Cost', 'Labor Cost', 'Maintenance Cost', 'Insurance Cost', 'Packaging Cost', 'Technology Fee', 'Overhead']
        
        # Impute missing cost values with 0 for summation
        for col in cost_cols:
            data[col] = data[col].fillna(0)
            
        data['Total Cost'] = data[cost_cols].sum(axis=1)
        
        # Impute missing distance for calculation, e.g., with mean
        data['Distance (km)'] = data['Distance (km)'].fillna(data['Distance (km)'].mean())

        data['Cost per km'] = data['Total Cost'] / data['Distance (km)']
        data['Profitability'] = data['Order Value'] - data['Total Cost']
        
        # Handle division by zero or inf
        data['Cost per km'] = data['Cost per km'].replace([np.inf, -np.inf], 0).fillna(0)
        
        return data
    
    except FileNotFoundError:
        st.error("Error: Dataset files not found. Please make sure all 7 CSVs are in the same folder as app.py.")
        return pd.DataFrame()

# --- 2. ML Model Training (Bonus Feature) ---
# Cache the model to avoid retraining on every interaction
@st.cache_resource
def train_model(df):
    # Select features for the model
    # We'll predict 'Total Cost' based on 'Order Value', 'Distance (km)', and 'Product Category'
    
    # Drop rows where 'Total Cost' is 0 or 'Distance (km)' is missing, as they don't help the model
    model_df = df[df['Total Cost'] > 0].copy()
    model_df = model_df[['Order Value', 'Distance (km)', 'Product Category', 'Total Cost']].dropna()

    if model_df.empty:
        return None

    # Define features (X) and target (y)
    X = model_df.drop('Total Cost', axis=1)
    y = model_df['Total Cost']
    
    # Define categorical and numerical features
    categorical_features = ['Product Category']
    numerical_features = ['Order Value', 'Distance (km)']
    
    # Create a preprocessor
    # Numerical features: Impute missing values with mean
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    # Categorical features: One-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the full model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Train the model
    model_pipeline.fit(X, y)
    
    return model_pipeline

# Load data and train model
df = load_data()
if not df.empty:
    model = train_model(df)
else:
    model = None

# --- 3. Streamlit App UI ---
if not df.empty:
    st.title("🚚 NexGen Logistics: Cost Intelligence Platform")
    st.markdown("This tool identifies cost leakage and optimization opportunities to help NexGen reduce operational costs.")
    
    # --- Sidebar Filters ---
    st.sidebar.header("Filter Dashboard")
    st.sidebar.markdown("Analyze costs by different segments.")

    # Filter by Product Category
    category = st.sidebar.multiselect(
        "Select Product Category:",
        options=df["Product Category"].unique(),
        default=df["Product Category"].unique()
    )

    # Filter by Customer Segment
    segment = st.sidebar.multiselect(
        "Select Customer Segment:",
        options=df["Customer Segment"].unique(),
        default=df["Customer Segment"].unique()
    )

    # Filter by Delivery Priority
    priority = st.sidebar.multiselect(
        "Select Delivery Priority:",
        options=df["Priority"].unique(),
        default=df["Priority"].unique()
    )

    # Filter the dataframe based on selections
    df_filtered = df[
        df["Product Category"].isin(category) &
        df["Customer Segment"].isin(segment) &
        df["Priority"].isin(priority)
    ]

    # --- Main Page Layout ---
    if df_filtered.empty:
        st.warning("No data matches your filters. Please adjust your selection.")
    else:
        # --- KPI Metrics (Addresses: Data Analysis, Visualizations) ---
        st.header("High-Level Cost Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cost", f"₹{df_filtered['Total Cost'].sum():,.2f}")
        col2.metric("Avg. Cost per Order", f"₹{df_filtered['Total Cost'].mean():,.2f}")
        col3.metric("Avg. Cost per km", f"₹{df_filtered['Cost per km'].mean():,.2f}")

        st.markdown("---")

        # --- Visualizations (Addresses: Visualizations, Data Analysis, UX) ---
        st.header("Cost Analysis & Optimization")
        
        # Chart 1: Cost Component Breakdown (Bar Chart)
        cost_components_data = df_filtered[['Fuel Cost', 'Labor Cost', 'Maintenance Cost', 'Insurance Cost', 'Packaging Cost', 'Technology Fee', 'Overhead']].sum().reset_index()
        cost_components_data.columns = ['Cost Component', 'Total Cost']
        fig1 = px.bar(
            cost_components_data, 
            x='Cost Component', 
            y='Total Cost', 
            title='Total Cost Breakdown by Component',
            labels={'Total Cost': 'Total Cost (₹)'}
        )
        fig1.update_layout(xaxis_title=None)
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Cost by Carrier (Bar Chart)
        fig2 = px.bar(
            df_filtered.groupby('Carrier')['Total Cost'].mean().reset_index().sort_values("Total Cost", ascending=False),
            x='Carrier',
            y='Total Cost',
            title='Average Cost per Order by Carrier',
            labels={'Total Cost': 'Avg. Total Cost (₹)', 'Carrier': 'Carrier Name'}
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Use columns for the next two charts
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # Chart 3: Profitability (Order Value vs. Total Cost) (Scatter Plot)
            # (Addresses: Innovation - derived metric)
            fig3 = px.scatter(
                df_filtered,
                x="Order Value",
                y="Total Cost",
                color="Product Category",
                hover_data=["Order ID", "Profitability"],
                title="Order Profitability (Value vs. Cost)"
            )
            fig3.add_shape(type="line", x0=0, y0=0, x1=df_filtered['Order Value'].max(), y1=df_filtered['Order Value'].max(), line=dict(color="Red", dash="dash"), name="Break-even")
            st.plotly_chart(fig3, use_container_width=True)
        
        with fig_col2:
            # Chart 4: Cost by Product Category (Pie Chart)
            fig4 = px.pie(
                df_filtered,
                names="Product Category",
                values="Total Cost",
                title="Share of Total Cost by Product Category"
            )
            st.plotly_chart(fig4, use_container_width=True)

        # --- Interactive Data Table & Download (Addresses: UX, Technical) ---
        st.header("Raw Data Explorer")
        st.dataframe(df_filtered)
        
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(df_filtered)
        
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_cost_data.csv",
            mime="text/csv",
        )
        
        st.markdown("---")

        # --- 4. BONUS FEATURE SECTION (Addresses: Bonus Points) ---
        st.header("✨ Bonus: Predictive Cost Calculator")
        st.markdown("Use our ML model to predict the total cost of a new order.")

        if model:
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                order_val_input = st.number_input("Order Value (₹)", min_value=0, value=10000)
                distance_input = st.number_input("Distance (km)", min_value=0, value=100)
            
            with pred_col2:
                category_input = st.selectbox("Product Category", options=sorted(df['Product Category'].unique()))
                predict_button = st.button("Predict Cost", type="primary")

            if predict_button:
                # Create a DataFrame for the model to predict
                input_data = pd.DataFrame({
                    'Order Value': [order_val_input],
                    'Distance (km)': [distance_input],
                    'Product Category': [category_input]
                })
                
                # Make prediction
                predicted_cost = model.predict(input_data)[0]
                
                st.subheader(f"Predicted Total Cost: ₹{predicted_cost:,.2f}")
                st.caption("This prediction is based on a Linear Regression model trained on historical data.")
        else:
            st.warning("Model could not be trained. Please check data files.")

else:
    st.error("Data could not be loaded. The app cannot continue.")