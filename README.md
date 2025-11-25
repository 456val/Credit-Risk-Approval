#  Credit Card Approval Prediction (Machine Learning Project)

An end-to-end machine learning solution that predicts whether a credit card/loan applicant will be approved based on demographic, financial, credit behavior, and loan intent features.

## 📑 Table of Contents
- Background  
- Objectives  
- Scope  
- Data  
- Methodology  
- Deliverables  
- Insights  
- Recommendations  
- Expected Impact  
- Future Work  
- License  

##  Background
Financial institutions rely on credit scoring and applicant profiling to determine whether an individual is eligible for credit card or loan approval.  
This project develops a predictive model to support **data-driven credit decision-making**, helping lenders reduce risk while improving approval efficiency.

## Objectives
- Build a predictive model to classify **loan_status** (Approved = 1, Not Approved = 0).  
- Perform exploratory data analysis (EDA) and address missing values.  
- Encode categorical variables and standardize numerical features.  
- Train and evaluate multiple ML algorithms.  
- Handle class imbalance using **SMOTE**.  
- Save the final model for deployment.

##  Scope

| Location / Context | Applicant Category | Focus / Challenge |
|-------------------|-------------------|------------------|
| Global / Lending | Credit Card & Loan Applicants | Predicting approval probability based on user and loan features |

##  Data

**Dataset:** `credit_risk_dataset.csv`  
**Rows:** 32,581  
**Columns:** 11  
**Target Variable:** `loan_status`  

### Key Features
- Applicant info: `person_age`, `person_income`, `person_emp_length`, `person_home_ownership`  
- Loan info: `loan_intent`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`  
- Credit behavior: `cb_person_default_on_file`, `cb_person_cred_hist_length`
  
##  Methodology

1. **Import Libraries & Load Dataset**  
2. **Data Cleaning & Missing Value Handling**  
   - Imputed employment length with 0  
   - Imputed interest rate with median  
3. **Categorical Encoding**  
   - Mapped ownership, loan intent, and credit history to numeric codes  
4. **Feature Scaling** using StandardScaler  
5. **Train-Test Split**  
6. **Model Training**  
   - Logistic Regression  
   - K-Nearest Neighbors  
   - **Voting Classifier (Soft Voting)** → Best performer  
7. **Model Evaluation**  
   - Accuracy  
   - Classification report  
   - Confusion matrix  
8. **Class Imbalance Handling**  
   - Applied **SMOTE**  
9. **Cross-Validation** (5-fold)  
10. **Model Saving** using `joblib.dump()`  

##  Deliverables
- Cleaned dataset (`credit_risk.csv`)  
- Model accuracy comparison (`model_accuracies.csv`)  
- Classification reports (before & after SMOTE)  
- Confusion matrix  
- Cross-validation scores  
- Saved ML model (`RiskApp2.6.pkl`)  

## Insights

| Challenge (Before) | Insight (After) | Measurable Impact |
|-------------------|----------------|------------------|
| Strong class imbalance (78% vs 22%) | Applied SMOTE for balanced training | Better recall for minority class |
| Missing values in key columns | Median/zero imputation applied | Full dataset usability |
| Raw categorical variables | Encoded numerical mappings | Improved model performance |
| Varying model accuracies | Voting Classifier combined strengths | **Accuracy improved to 0.8683** |

## Recommendations
- Collect more behavioral credit data (utilization, on-time payments).  
- Try ensemble models like XGBoost, Random Forest.  
- Deploy using FastAPI or Streamlit.  
- Add model explainability (SHAP values).  
- Implement automated hyperparameter tuning.

## Expected Impact
- Faster and more consistent credit decision-making.  
- Reduced manual screening time and human bias.  
- Improved prediction of high-risk applicants.  
- Better financial risk management.

## Future Work

- Hyperparameter tuning with GridSearchCV  
- Add XGBoost and LightGBM versions  
- Build REST API endpoint  
- Deploy on HuggingFace Spaces or Streamlit Cloud  
- Add dashboards for risk profile visualization  

##  License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute with attribution.

cd credit-card-approval-ml
