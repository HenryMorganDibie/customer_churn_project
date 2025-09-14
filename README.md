# 📊 Customer Churn Analysis & Retention Strategy

## 📌 Project Overview
This project provides a **comprehensive business intelligence solution** to address low customer retention.  
It combines a **robust data analysis pipeline** with an **interactive Power BI dashboard** to identify and visualize at-risk customers, empowering stakeholders to make **proactive, data-driven decisions**.  

The solution is divided into:
- **Analytical Phase** → Data preparation, feature engineering, and predictive modeling.  
- **Reporting Phase** → Interactive Power BI dashboard with actionable insights.  

---

## 🔑 Key Features
- **Data Preparation:** Reproducible pipeline for cleaning and handling missing values across three datasets (*orders, product_orders, contacts_raw*).  
- **Feature Engineering:** Created domain-specific features (*recency, tenure, order frequency, brand loyalty*) to capture customer behavior.  
- **Predictive Modeling:** Trained and compared **Logistic Regression, Random Forest, and XGBoost** for churn prediction.  
- **Interactive Dashboard:** Designed a **Power BI dashboard** with KPIs and visuals to track retention and highlight at-risk customers.  

---

## 📂 Project Structure

The repository is organized for clarity and collaboration.

<pre lang="markdown">
customer_churn_project/
├── data/                         # Contains all raw data files (.xlsx)
├── models/                       # Stores trained model assets (.joblib files)
│   ├── lr_model.joblib
│   ├── scaler.joblib
│   └── train_columns.joblib
├── notebooks/                    # Holds the Jupyter Notebook detailing the analysis
│   └── Customer Churn Prediction & Retention Strategy.ipynb
├── customer_churn_analysis.py    # Main script for training the model
├── predict.py                    # Script for making new predictions
└── requirements.txt              # Lists project dependencies
</pre>

# How to Run the Project
Follow these steps to set up and run the analysis.

Prerequisites
Python 3.8+
```bash
pip (Python package installer)
```

1. Install Dependencies
From the project root directory, install all necessary libraries:

```bash
pip install -r requirements.txt
```
2. Train the Model (One-Time Step)
Run this command to execute the full analysis, train the model, and save the model assets.

```bash
python customer_churn_analysis.py
```
3. Make Predictions
Once the model files are in the models/ directory, you can run this script to generate churn probabilities for your most recent customer data.

```bash
python predict.py
```
The output will be a list of the top customers with the highest churn risk, ready for your business teams to use.

### ✨ Dashboard Analysis & Key Findings

[Dashboard](<output/customer retention & churn risk dashboard.pdf>)

Based on the predictive model, I've built a Power BI dashboard that provides a clear and actionable view of customer retention. The key insights derived from the analysis and the dashboard are:

1. High-Level Metrics:

- Total Revenue: A snapshot of total revenue (#203bn).

- Total Orders: The total number of orders placed (109K).

- Customer Base: A clear distinction between the total active customers (18K) and the subset of those at risk (16K), which provides a meaningful Churn Risk Rate.

2. Product & Geographical Insights:

- Product Risk: The dashboard's visuals show that Noodles and Milk & Dairy categories have the highest churn risk concentration.

- Geographical Risk: The Churn Risk Rate by State chart identifies Nasarawa, Oyo, and Rivers as the top three states with the highest risk rates, providing a clear target for localized retention campaigns.

3. Customer Behavior:

- Churn Indicators: The analysis found that at-risk customers tend to have lower total spending and lower order frequency compared to retained customers.

- Actionable List: The Top 10 Churn Risk Customers table provides a prioritized list, complete with churn probability and Customer Lifetime Value (CLV), to guide targeted outreach.

This project successfully bridges the gap between data science and business strategy, providing a dynamic tool for proactive customer retention.

### 📝 Results & Insights

Based on the analysis, the Logistic Regression model provided the best balance of performance and interpretability, achieving an Accuracy of 69% and a ROC AUC of 0.76.

The final prediction script successfully identifies the customers most likely to churn based on their recent behavior, providing a clear and actionable list for prioritizing retention efforts.

### Key Insights from Exploratory Data Analysis
- Churn Distribution: The dataset shows a balanced churn rate, with 46.6% of customers churning and 53.4% being retained.

- Behavioral Patterns: Customers who churned typically had a lower total number of orders and a lower total spent compared to retained customers.

- Recency and Frequency: The order_frequency feature was a strong indicator of churn. Retained customers tended to have a higher frequency of orders, suggesting that recent and regular engagement is a critical factor in customer loyalty.

- Model Performance Comparison
The following metrics were used to evaluate the models, with Logistic Regression chosen for its strong performance and high interpretability.

 - Model
 - Accuracy
 - Precision
 - Recall
 - F1-Score
 - ROC AUC

#### Random Forest

   - 0.690

   - 0.670

   - 0.661

   - 0.665

   - 0.762

#### Logistic Regression

   - 0.694

   - 0.666

   - 0.690

   - 0.678

   - 0.763

#### XGBoost

   - 0.686

   - 0.660

   - 0.674

   - 0.667

   - 0.754

## Analysis Summary and Key Findings

My project successfully implements a full predictive modeling pipeline to identify customers at risk of churning. Here is a summary of the key insights and findings from my analysis:

1. Data Cleaning and Feature Engineering
Successful Data Preparation: My script correctly loads and cleans the three datasets, handling missing values and outlier treatment with the 99th percentile cutoff.

- Resolved KeyError: The fix implemented in the create_features function—by merging the customerid column from the orders DataFrame into the product_orders DataFrame—successfully resolves the data issue and allows the rest of the script to run smoothly.

2. Exploratory Data Analysis (EDA)
Churn Rate: The analysis shows a nearly balanced churn rate, with approximately 47% of customers identified as churned.

- Key Indicators: The boxplots and histograms provide valuable insights:

- Total Orders and Total Spent: Customers who churned tend to have a lower total number of orders and lower total spending compared to retained customers.

- Order Frequency: Churned customers show a wider distribution of order frequency, with many having very low frequency, which is a strong indicator of disengagement.

3. Predictive Modeling and Evaluation
- Model Performance: I trained and evaluated three models: Random Forest, Logistic Regression, and XGBoost.

- ROC AUC Score: The Logistic Regression model performed marginally better with an ROC AUC score of 0.763, indicating a good balance between identifying true positives and avoiding false positives.

- Consistency: The models show similar overall performance, which suggests the features I engineered are robust and predictive across different algorithms.

**Actionable Insights:**

Targeted Retention: My script successfully provides a list of the top 10 customers with the highest churn probability. This list is a crucial business asset.

**Strategy Recommendation:** 

The business can now use this list to launch a targeted retention campaign. This might include:

- Offering special discounts or promotions.

- Providing personalized customer service outreach.

- Sending exclusive content or product recommendations to re-engage these at-risk customers.

This is a great foundation for a business to start proactively addressing customer retention.

**Contact**

For any questions or collaborations, feel free to reach out.

Henry Dibie

henrymorgan273@yahoo.com

Location: Nigeria