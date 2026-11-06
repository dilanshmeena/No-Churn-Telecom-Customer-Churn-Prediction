# No-Churn Telecom: Customer Churn Prediction

Machine learning project for **No-Churn Telecom** (project, telecom churn). The goal is to understand why customers leave, score each customer's churn risk, and flag customers for retention campaigns.

## Project goals

1. Understand the variables that influence customers to leave.
2. Create churn risk scores to drive retention campaigns.
3. Create a CHURN-FLAG (YES = 1, NO = 0) to target email campaigns.

## Data

- Source: SQL database `project_telecom`, table `telecom_churn_data`
- Size: 243,553 customers, 14 columns
- Target: `churn` (1 = left, 0 = stayed). About 20% of customers churned.
- Columns: `customer_id`, `telecom_partner`, `gender`, `age`, `state`, `city`, `pincode`, `date_of_registration`, `num_dependents`, `estimated_salary`, `calls_made`, `sms_sent`, `data_used`, `churn`

The data is private and not included in this repository.

## Approach

The whole project is in one notebook, `No-Churn_Telecom.ipynb`, in 11 phases:

1. Setup and data loading
2. Data understanding
3. Data cleaning (negative usage values replaced with the median, `pincode` dropped)
4. Exploratory data analysis
5. Preprocessing (tenure feature, encoding, scaling, 80/20 stratified split)
6. Model building (Logistic Regression, Decision Tree, Random Forest, XGBoost, with class weights)
7. Model evaluation and comparison
8. Feature importance
9. Churn risk score (0 to 100, Low / Medium / High bands)
10. CHURN-FLAG
11. Business suggestions and challenges

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.504 | 0.201 | 0.497 | 0.287 | 0.506 |
| Decision Tree | 0.283 | 0.200 | 0.858 | 0.324 | 0.500 |
| Random Forest | 0.508 | 0.199 | 0.478 | 0.281 | 0.499 |
| XGBoost | 0.498 | 0.201 | 0.505 | 0.288 | 0.500 |

## Key findings

- Churn is about 20% in every telecom partner, gender, city, state and registration year, and all correlations with churn are near 0.
- All four models score a ROC-AUC of about 0.50, the same as random guessing. The data has very little signal.
- Random Forest feature importances are small and spread out. No variable is a clear driver of churn.
- On test customers, the Low, Medium and High risk bands have almost the same churn (20.2%, 20.1%, 19.9%), so the risk score and CHURN-FLAG are a working pipeline, not a reliable predictor yet.
- A better model needs richer data, such as complaints, billing details and contract type.

## How to run

1. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Open `No-Churn_Telecom.ipynb` and fill in the database details (`host`, `user`, `password`) in the connection cell.
3. Run all cells. The last code cell saves `churn_flag_customers.csv` with `customer_id`, `churn_risk_score`, `risk_band` and `churn_flag`.

## Tech stack

Python, pandas, matplotlib, seaborn, SQLAlchemy, scikit-learn, XGBoost, Jupyter
