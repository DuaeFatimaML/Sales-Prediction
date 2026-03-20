🛒 Retail Sales Prediction using Random Forest

Predicting store sales from time-series data using an end-to-end scikit-learn pipeline — helping businesses make smarter inventory and revenue decisions.



🎯 Problem Statement
Retail businesses struggle to forecast sales accurately, leading to overstocking, understocking, and revenue loss. This project builds a full machine learning pipeline that predicts future sales from historical time-series data — enabling data-driven business decisions.

🚀 Live Demo
PlatformLink🔴
Streamlit App[Your Streamlit Link Here]

📊 Model Performance
Metric                     Score
R² Score                   0.83
Mean Absolute Error (MAE)  2,239.49
Model Error Rate           6.44%
Average Sales Value        34,748.65
✅ Overfitting Check
Split            R² Score
Training Set      0.8319
Validation (CV)   0.8319
Status✅ Healthy Model — No Overfitting Detected

Training and validation scores are identical, confirming the model generalizes well to unseen data and is not memorizing the training set.


🧠 Technical Implementation
Full Pipeline Architecture
This project uses a scikit-learn Pipeline — ensuring all preprocessing and model steps are chained together, preventing data leakage between training and prediction:
Raw CSV Data
     ↓
Date Feature Engineering
(Year, Month, Day, TimeIndex extracted — Date column dropped)
     ↓
ColumnTransformer
     ├── SimpleImputer  (fills missing values using median)
     └── QuantileTransformer  (maps to normal distribution)
     ↓
RandomForestRegressor (100 trees, max_depth=10)
     ↓
Predicted Sales + Evaluation Metrics

⚙️ Why QuantileTransformer?

Sales data is rarely normally distributed — it contains outliers from seasonal spikes and promotions. QuantileTransformer maps features to a normal distribution, making the model robust to these extremes without losing information.

⚙️ Why Random Forest?

-Handles non-linear relationships in sales naturally
-Robust to outliers common in retail data
-Provides feature importance — shows which factors drive sales most
-Ensemble averaging = stable, reliable predictions
-No manual feature scaling required


⚙️ Model Hyperparameters
Parameter     Value   Reason
n_estimators      100   Balances accuracy and training speed
max_depth          10   Prevents overfitting
min_samples_split  4    Controls tree growth
min_samples_leaf   2    Smooths leaf predictions
max_features     sqrt   Standard best practice for RF
bootstrap        True   Enables ensemble variance reduction
random_state      42    Full reproducibility

🗓️ Input Features
Feature         Type     Description  
Year            Numeric  Extracted from Date — captures yearly trends
Month           Numeric  Captures monthly seasonality
Day             Numeric  Captures day-level patterns
TimeIndex       Numeric  Sequential index — captures overall sales trend
Footfall        Numeric  Number of store visitors
Holiday_Season  Binary   Whether it is a holiday period
Marketing_Spend Numeric  Amount spent on marketing

The original Date column was dropped after feature extraction since ML models cannot process raw datetime objects directly.


📈 Visualization
Actual vs Predicted Sales graph for the last 60 days — visual validation of model performance beyond just metrics:

🔵 Blue line — Actual sales values
🔴 Red dashed line — Model predictions


🛠️ Tech Stack
ToolPurposePython 3.11Core 
languages :
scikit-learn 
-Pipeline
-preprocessing
-RF model
pandas
-Data loading 
-manipulation
numpy
-Feature engineering (TimeIndex)
matplotlib
-Actual vs Predicted visualization
joblib
-Model & feature serialization
Streamlit
-Web app deployment

Why Synthetic Dataset:
I used synthetic data to simulate real retail scenarios because real sales data is proprietary

⚠️ Model Limitations

Trained on synthetic AI-generated data — simulates real retail scenarios since actual proprietary sales data is not publicly available
TimeIndex feature means model is sensitive to data order — not ideal for predicting far into the future
Does not account for sudden external factors like economic shifts or supply chain disruptions
Performance may vary on datasets with very different sales value distributions


🔮 Future Improvements

 Test on real-world retail dataset (e.g. Kaggle Rossmann Store Sales)
 Add time-series specific models (Prophet, LSTM) for comparison
 Hyperparameter tuning with GridSearchCV
 Incorporate more external features like competitor pricing
 Add prediction confidence intervals to the Streamlit app


👨‍💻 Author
[Dua e Fatima]


⭐ If you found this project useful, please give it a star
