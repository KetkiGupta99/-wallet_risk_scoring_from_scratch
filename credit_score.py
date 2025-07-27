import pandas as pd
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import os

# Ensure your JSON file path is correct
JSON_FILE_PATH = r"D:\Credit_Score_Scratch\compound_wallet_transactions.json"


try:
    with open(JSON_FILE_PATH, 'r') as file:
        data = json.load(file)
        print("JSON is properly formatted.")
except FileNotFoundError:
    print(f"Error: JSON file not found at {JSON_FILE_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {JSON_FILE_PATH}. Check file format.")
    exit()

rows = []
for wallet, txs in data.items():
    for tx in txs:
        tx["wallet"] = wallet
        rows.append(tx)

df = pd.DataFrame(rows)

# Convert data types and scale values
df["timeStamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit='s')
df["value"] = df["value"].astype(float) / 1e18  # Convert from Wei to ETH
df["gas"] = df["gas"].astype(float)
df["gasPrice"] = df["gasPrice"].astype(float) / 1e9  # Gwei
df["gasUsed"] = df["gasUsed"].astype(float)
df["isError"] = df["isError"].astype(int)


df_filtered = df[df['from'] != df['to']].copy()

#  Feature Engineering 

def get_unique_counterparties(group):
    wallet_address = group.name 
    all_related_addresses = set(group['from'].tolist() + group['to'].tolist())
    all_related_addresses.discard(wallet_address) 
    return len(all_related_addresses)

features = df_filtered.groupby("wallet", group_keys=False).apply(
    lambda g: pd.Series({
        "total_txns": g["hash"].count(),
        "total_sent": g[g["from"] == g.name]["value"].sum(),
        "total_received": g[g["to"] == g.name]["value"].sum(),
        "failed_txns": g["isError"].sum(),
        "avg_gas_used": g["gasUsed"].mean(),
        "avg_gas_price": g["gasPrice"].mean(),
        "unique_counterparties": get_unique_counterparties(g), 
        "first_tx_time": g["timeStamp"].min(),
        "last_tx_time": g["timeStamp"].max(),
    }), include_groups=False
)

features = features.fillna(0)

# Add derived features
features["active_days"] = (features["last_tx_time"] - features["first_tx_time"]).dt.days + 1
features["active_days"] = features["active_days"].replace(0, 1) 

features["txn_per_day"] = features["total_txns"] / features["active_days"]
features["fail_rate"] = features["failed_txns"] / features["total_txns"].replace(0, 1) # Avoid div by zero
features["net_received"] = features["total_received"] - features["total_sent"]

features["high_fail_rate"] = (features["fail_rate"] > 0.5).astype(int)
features["high_outflow"] = (features["net_received"] < 0).astype(int)
features["suspicious_gas"] = (features["avg_gas_price"] > 100).astype(int) # Gwei
features["low_diversity"] = (features["unique_counterparties"] < 3).astype(int)

initial_rows = len(features)
features = features.replace([np.inf, -np.inf], np.nan).dropna(subset=[
    "total_txns", "total_sent", "total_received", "failed_txns",
    "avg_gas_used", "avg_gas_price", "unique_counterparties",
    "active_days", "txn_per_day", "fail_rate", "net_received"
])

# Data Visualization 

# Distribution of Total Transactions per Wallet
plt.figure(figsize=(10, 5))
sns.histplot(features["total_txns"], bins=50, kde=True)
plt.title("Distribution of Total Transactions per Wallet")
plt.xlabel("Total Transactions")
plt.ylabel("Number of Wallets")
#plt.show()
plt.savefig('Distribution of Total Transactions per Wallet.png')
plt.close()

# Wallets: ETH Sent vs Received
plt.figure(figsize=(10, 6))
sns.scatterplot(x="total_sent", y="total_received", data=features, alpha=0.6)
plt.title("Wallets: ETH Sent vs Received")
plt.xlabel("Total ETH Sent")
plt.ylabel("Total ETH Received")
plt.grid(True)
#plt.show()
plt.savefig('Wallets_ETH_Sent_vs_Received.png')
plt.close()

# Distribution of Transaction Fail Rate
plt.figure(figsize=(10, 5))
sns.histplot(features["fail_rate"], bins=40, kde=True, color='red')
plt.title("Distribution of Transaction Fail Rate")
plt.xlabel("Fail Rate")
plt.ylabel("Wallet Count")
#plt.show()
plt.savefig('Distribution_of_Transaction_Fail_Rate.png')
plt.close()

# Monthly Transaction Volume (across all wallets)
tx_time_series = df_filtered.groupby(df_filtered["timeStamp"].dt.to_period("M")).size()
tx_time_series.index = tx_time_series.index.to_timestamp()

plt.figure(figsize=(12, 6))
tx_time_series.plot()
plt.title("Monthly Transaction Volume")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")
plt.grid(True)
#plt.show()
plt.savefig('Monthly_Transaction_Volume.png')
plt.close()

# Interactive Scatter Plot with Plotly
'''
fig = px.scatter(
    features,
    x="txn_per_day",
    y="fail_rate",
    size="total_txns",
    color="net_received",
    hover_name=features.index,
    title="Wallet Risk Patterns (Interactive)",
    labels={
        "txn_per_day": "Transactions Per Day",
        "fail_rate": "Fail Rate",
        "total_txns": "Total Transactions",
        "net_received": "Net ETH Received"
    },
    color_continuous_scale=px.colors.sequential.Viridis # Or any other colormap
)
fig.show()
'''


# Feature Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_features_for_corr = features.select_dtypes(include=np.number).columns
corr = features[numeric_features_for_corr].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig('Feature_Correlation_Heatmap.png')
plt.close()


# Develop a Risk Scoring 
selected_features_for_custom_score = features[[
    "total_txns", "fail_rate", "net_received", "txn_per_day",
    "unique_counterparties", "avg_gas_price"
]].copy() 

selected_features_for_custom_score = selected_features_for_custom_score.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = MinMaxScaler()
normalized_features = pd.DataFrame(
    scaler.fit_transform(selected_features_for_custom_score),
    columns=selected_features_for_custom_score.columns,
    index=selected_features_for_custom_score.index
)

normalized_features["inv_fail_rate"] = 1 - normalized_features["fail_rate"]
normalized_features["good_txn_per_day"] = normalized_features["txn_per_day"]
normalized_features["inv_avg_gas_price"] = 1 - normalized_features["avg_gas_price"] 

weights = {
    "inv_fail_rate": 0.25,          
    "good_txn_per_day": 0.15,       
    "net_received": 0.20,           
    "unique_counterparties": 0.15,  
    "total_txns": 0.10,             
    "inv_avg_gas_price": 0.15       
}

score = (
    weights["inv_fail_rate"] * normalized_features["inv_fail_rate"] +
    weights["good_txn_per_day"] * normalized_features["good_txn_per_day"] +
    weights["net_received"] * normalized_features["net_received"] +
    weights["unique_counterparties"] * normalized_features["unique_counterparties"] +
    weights["total_txns"] * normalized_features["total_txns"] +
    weights["inv_avg_gas_price"] * normalized_features["inv_avg_gas_price"]
)


features["risk_score"] = (score * 1000).round().astype(int)

# Distribution of Wallet Risk Scores
plt.figure(figsize=(10, 6))
sns.histplot(features["risk_score"], bins=30, kde=True, color='purple')
plt.title("Distribution of Wallet Risk Scores")
plt.xlabel("Risk Score (0 - 1000)")
plt.ylabel("Number of Wallets")
plt.grid(True)
plt.savefig('Distribution_of_Wallet_Risk_Scores_Custom.png')
plt.close()

plt.figure(figsize=(10, 2))
sns.boxplot(data=features, x="risk_score", color='lightblue')
plt.title("Wallet Risk Score - Boxplot")
plt.xlabel("Risk Score")
plt.savefig('Wallet_Risk_Score_Boxplot_Custom.png')
plt.close()

def risk_label(score):
    if score >= 750:
        return "Low Risk"
    elif score >= 500:
        return "Medium Risk"
    else:
        return "High Risk"

features["risk_category"] = features["risk_score"].apply(risk_label)

plt.figure(figsize=(12, 8))
sns.countplot(data=features, x="risk_category", palette="Set1", order=["Low Risk", "Medium Risk", "High Risk"])
plt.title("Count of Wallets by Risk Category (Custom Rule-Based)")
plt.xlabel("Risk Category")
plt.ylabel("Number of Wallets")
plt.savefig('Count_of_Wallets_by_Risk_Category_Custom.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.scatterplot(data=features, x="fail_rate", y="risk_score", hue="risk_category", palette="Set1", alpha=0.7)
plt.title("Risk Score (Custom) vs. Transaction Fail Rate")
plt.xlabel("Fail Rate")
plt.ylabel("Risk Score")
plt.grid(True)
plt.savefig('Risk_Score_Custom_vs_Fail_Rate.png')
plt.close()


# Model 
# Features (X) to predict the risk score (y)
X = features[[
    "total_txns", "fail_rate", "net_received", "txn_per_day",
    "unique_counterparties", "avg_gas_price"
]].copy() 

y = features["risk_score"]

X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean()) 
y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "MLP": MLPRegressor(max_iter=1000, random_state=42) 
}

results = []
best_r2 = -float('inf')
best_model_name = ""
best_model_instance = None

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R²": round(r2, 4)
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_instance = model
    except Exception as e:
        results.append({
            "Model": name,
            "MAE": "N/A",
            "RMSE": "N/A",
            "R²": "N/A"
        })

results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False)
print(results_df)

if best_model_instance:
    print(f"\nBest performing model based on R²: {best_model_name}")
    features["ml_risk_score"] = best_model_instance.predict(X)
else:
    print("\nNo best model found.")
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train) # Fit on training data
    features["ml_risk_score"] = lin_reg_model.predict(X)

features["ml_risk_score"] = features["ml_risk_score"].clip(0, 1000).round().astype(int)

# Saving Results
features = features.reset_index() # Ensure 'wallet' is a regular column

output = features[["wallet", "ml_risk_score"]].copy()
output.columns = ["wallet_id", "score"]
output.to_csv("wallet_risk_scores.csv", index=False)
