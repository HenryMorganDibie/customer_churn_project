# customer_churn_analysis.py
# This script performs a full churn analysis, trains a Logistic Regression model,
# and saves the model, scaler, and feature list for future use.

# --- 1. Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# --- 2. Data Loading and Cleaning ---
try:
    contacts_raw = pd.read_excel("data/RetentionCaseStudy.xlsx", sheet_name="MeetingData")
    orders = pd.read_excel("data/RetentionCaseStudy-Data2.xlsx", sheet_name="OrderData")
    product_orders = pd.read_excel("data/RetentionCaseStudy-data3.xlsx", sheet_name="Sheet1")
    print("All data files loaded successfully.")
except FileNotFoundError:
    print("Error: One or more data files are missing. Please ensure they are in the 'data/' directory.")
    raise

# Data cleaning steps from the notebook
orders['cityname'] = orders['cityname'].fillna('Unknown')
orders['TownName'] = orders['TownName'].fillna('Unknown')
orders['OrderStatus'] = orders['OrderStatus'].fillna('Unknown')
contacts_raw['NoActivityReason'] = contacts_raw['NoActivityReason'].fillna('Active')
contacts_raw = contacts_raw.dropna(subset=['meetingid'])

# --- 3. Feature Engineering ---
cutoff_date = pd.to_datetime('2025-03-01')

def create_features(orders_df, contacts_df, product_orders_df):
    """
    Combines the three datasets and engineers the final feature set.
    """
    # Orders Aggregation
    orders_agg = orders_df.groupby('customerid').agg(
        total_orders=('orderid','nunique'),
        total_quantity=('TotalQty','sum'),
        total_spent=('TotalPrice','sum'),
        first_order_date=('deliveredDate','min'),
        last_order_date=('deliveredDate','max')).reset_index()

    # Contacts Aggregation
    contacts_agg = contacts_df.groupby('customerid').agg(
        num_meetings=('meetingid','nunique'),
        avg_meeting_duration=('MeetingDuration','mean'),
        num_no_activity=('NoActivityReason', lambda x: x.notna().sum())).reset_index()

    # Product Aggregation - Robust to missing 'customerid'
    if 'customerid' not in product_orders_df.columns:
        product_orders_df = product_orders_df.merge(orders_df[['orderid', 'customerid']], on='orderid', how='left')
    
    product_agg = product_orders_df.groupby('customerid').agg(
        total_order_quantity=('Quantity','sum'),
        total_order_price=('Price','sum')).reset_index()
    
    # Merge and fill missing values
    customer_df = orders_agg.merge(contacts_agg, on='customerid', how='left')
    customer_df = customer_df.merge(product_agg, on='customerid', how='left')
    numeric_cols_to_fill = ['total_order_quantity', 'total_order_price', 'num_meetings', 'avg_meeting_duration', 'num_no_activity']
    customer_df[numeric_cols_to_fill] = customer_df[numeric_cols_to_fill].fillna(0)
    
    # Calculate additional features
    customer_df['days_since_last_order'] = (cutoff_date - customer_df['last_order_date']).dt.days.fillna(999)
    customer_df['customer_tenure_days'] = (customer_df['last_order_date'] - customer_df['first_order_date']).dt.days.fillna(0)
    customer_df['order_frequency'] = customer_df['total_orders'] / customer_df['customer_tenure_days'].replace(0,1)
    customer_df['avg_order_value'] = customer_df['total_spent'] / customer_df['total_orders'].replace(0,1)
    customer_df['recency_ratio'] = customer_df['days_since_last_order'] / customer_df['customer_tenure_days'].replace(0,1)
    customer_df['meeting_engagement'] = customer_df['num_meetings'] / (customer_df['customer_tenure_days'].replace(0, 1))
    customer_df['brand_loyalty'] = customer_df['total_order_price'] / customer_df['total_spent'].replace(0, 1)

    return customer_df.drop(columns=['first_order_date', 'last_order_date'])

# Get features for training
orders_pre_cutoff = orders[orders['deliveredDate'] < cutoff_date]
contacts_pre_cutoff = contacts_raw[contacts_raw['MeetingStartDate'] < cutoff_date]
product_orders_pre_cutoff = product_orders.merge(orders_pre_cutoff[['orderid', 'customerid']], on='orderid', how='inner')
customer_df_fixed = create_features(orders_pre_cutoff, contacts_pre_cutoff, product_orders_pre_cutoff)

# Define the target variable (churn)
retained_customers_post_cutoff = orders[orders['deliveredDate'] >= cutoff_date]['customerid'].unique()
customer_df_fixed['churn'] = customer_df_fixed['customerid'].apply(lambda x: 0 if x in retained_customers_post_cutoff else 1)

# --- 4. Model Training and Saving ---
print("Training the Logistic Regression model...")
X = customer_df_fixed.drop(columns=['customerid', 'churn'])
y = customer_df_fixed['churn']
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
lr_model = LogisticRegression(solver='liblinear', random_state=42)
lr_model.fit(X_train, y_train)

# Save the trained model, scaler, and feature list
joblib.dump(lr_model, 'models/lr_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(X_train.columns.tolist(), 'models/train_columns.joblib')
print("Model, scaler, and training columns saved successfully in the 'models/' directory.")
