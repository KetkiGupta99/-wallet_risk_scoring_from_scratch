# -wallet_risk_scoring_from_scratch

## Objective

Retrieve the transaction data for each provided wallet address, Organize and preprocess the transaction data to create meaningful features that can reflect each wallet's risk profile and Develop a scoring model that assigns each wallet a risk score ranging from 0 to 1000.

# Project Structure

      Wallet id.csv # all wallet id  
      wallet_scoring_rough_work.ipynb # Rough work
      credit_score.py/ # main python file
      requirements.txt # Required Python packages
      retrieve_trasaction_data.py # retrieving transaction data python file 
      compound_wallet_transactions.json # transaction data in JSON 
      wallet_risk_scores.csv # Final output
      .env file 

1. Create and activate a virtual environment
   
       python -m venv credit
       credit\Scripts\activate
   
2. Install dependencies: `pip install -r requirements.txt`
   
3.  Data Collection Method:
    1. The goal was to retrieve historical transaction data for a list of wallet addresses interacting with the Compound V2 or V3 lending protocol.
    2. While initially intended to use The Graph’s Compound V2/V3 subgraphs for on-chain data access, these endpoints are currently deprecated or unavailable, returning errors such as:
    `{'0xfe5a05c0f8b24fca15a7306f6a4ebb7dcf2186ac': {'message': 'Not found'}} OR "This endpoint has been removed. If you have any questions, reach out to support@thegraph.zendesk.com"`
    3. So I have used Etherscan API: Created an API for trasaction detail: https://etherscan.io/apidashboard. I have collected transaction data for each wallet using the Etherscan API       to extract historical transaction records.
4. API Credentials:
   - Filename: `.env`
   - Add `ETHERSCAN_API_KEY="<YOUR_API_KEY>"` in .env file.

6. Data Preparation and Feature Engineering:
   - This section prepares Ethereum transaction data for risk scoring or analysis by cleaning, transforming, and engineering relevant features per wallet.
   1. Create DataFrame from Raw Rows
   2. Data Type Conversion and Normalization:
      - Converts UNIX timestamps to readable datetime format.
      - Scales value from Wei to ETH and gasPrice from Wei to Gwei.
      - Ensures consistent numeric types for downstream calculations.
   3. Filter Out Self-Transfers: Removes transactions where the sender and receiver are the same, as these are often not meaningful for risk analysis.
   4. Feature Engineering Per Wallet: Generates core features like total transactions, failed transactions, sent/received value, average gas usage and price, time span of activity, and         interaction diversity.
   5. Derived Features:
      - Computes the number of active days (ensuring minimum of 1 to avoid division by zero).
      - Creates behavioral columns like transaction frequency, failure rate, and net value received.
   6. Binary Risk Indicators: Adds flags for suspicious behavior:
                             - `High failure rate (>50%)`
                             - `Net ETH outflow`
                             - `Gas price > 100 Gwei (potential bot activity)`
                             - `Low interaction diversity (less than 3 unique addresses)`
   7. Handle Missing and Invalid Data: Replaces infinite values and drops rows with missing critical features to ensure clean and usable data for modeling or scoring.

7. Data Visualizations
8. Risk Scoring for wallets
   - This section assigns a risk score (0–1000) and a corresponding risk category (Low, Medium, High) to each wallet address, based on normalized transactional behavior.
   1. Select Relevant Features for Scoring: 
      Extracts key features that reflect wallet behavior: Transaction volume, failure rate, and transaction frequency, Net received ETH, interaction diversity, and gas pricing
   2. Handle Missing or Infinite Values: Cleans the dataset to avoid model distortion due to NaNs or invalid values.
   3. Normalize Features: Scales all selected features between 0 and 1 using Min-Max normalization to ensure fair comparison across features with different scales.
   4. Transform Features for Risk Interpretation:
      - Inverse metrics are calculated so that higher values always indicate lower risk:
        - inv_fail_rate: Lower failure rate = higher value
        - inv_avg_gas_price: Lower gas price = higher value
      - Transaction frequency and net received value are treated as-is (positive signals).
   5. Apply Weighted Scoring Formula:
      - Custom weights are assigned to each behavioral feature based on its significance in determining wallet risk.
      - Weighted linear combination of features produces a composite risk score between 0 and 1.
   6. Convert to Risk Score (0–1000 Scale): Final score is scaled from 0–1 to a 0–1000 range.
   7. Assign Risk Category Labels: Labels wallets into Low, Medium, or High Risk buckets based on thresholds:
      - 750–1000: Low Risk - safe behavior
      - 500–749: Medium Risk
      - 0–499: High Risk

 9. ML-Based Risk Scoring: Model Training and Evaluation
    1. Feature and Target Selection:
       - X: Input features representing wallet behavior
       - y: Target variable — custom risk score (0–1000) computed earlier.
    2. Handle Missing or Infinite Values: Cleans dataset by replacing inf and NaN with column means to ensure smooth model training.
    3. Train-Test Split: Splits data into 80% training and 20% testing to evaluate model generalization.
    4. Define Multiple Regression Models: A range of linear and nonlinear models are included for evaluation:
       - Regularized Linear Models: Ridge, Lasso
       - Tree-based models: Decision Tree, Random Forest, Gradient Boosting
       - Other: SVR, KNN, MLP
    5. Train and Evaluate Each Model:
       - Each model is trained and evaluated using:
         - MAE (Mean Absolute Error) – lower is better
         - RMSE (Root Mean Squared Error) – penalizes larger errors
         - R² (Coefficient of Determination) – higher is better (1.0 is perfect)
       - Keeps track of the best-performing model based on R².
    6. Display Results: Presents all models and their performance metrics in a sorted table.
       ###  Model Performance Comparison

          | Model              | MAE    | RMSE   | R²      |
          |--------------------|--------|--------|---------|
          | Linear Regression  | 0.28   | 0.32   | 1.0000  |
          | Lasso              | 4.33   | 6.05   | 0.9887  |
          | Ridge              | 11.45  | 17.07  | 0.9102  |
          | KNN                | 44.53  | 56.78  | 0.0062  |
          | SVR                | 42.57  | 58.50  | -0.0547 |
          | Gradient Boosting  | 32.80  | 60.23  | -0.1181 |
          | Random Forest      | 36.04  | 63.29  | -0.2347 |
          | Decision Tree      | 41.20  | 77.19  | -0.8362 |
          | MLP                | 286.80 | 385.61 | -44.8280 |
       Best performing model based on R²: Linear Regression
    7. Select and Apply Best Model:
       Uses the best model to predict ml_risk_score for all wallets
    8. Post-processing: Ensures final scores are within the [0, 1000] range and properly rounded
    9. Export Risk Scores: Saves the final ML-generated risk scores per wallet as a CSV file (wallet_risk_scores.csv).
    

       




      
        
