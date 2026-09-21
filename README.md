# nasa-turbofan-predictive-maintenance

# NASA Turbofan Engine Predictive Maintenance (CMAPSS)

An end-to-end Machine Learning pipeline for Remaining Useful Life (RUL) prediction and operational health state classification using NASA's CMAPSS dataset.

## 🚀 Project Overview
Predictive maintenance is critical in aerospace engineering to prevent catastrophic failures and optimize maintenance schedules. This project builds a robust predictive model, transforming raw multi-sensor time-series telemetry into actionable operational decisions.

### Key Engineering Features:
- **Data Leakage Prevention:** Strict engine-level splitting between training and testing sets.
- **Physics-Informed Feature Engineering:** Generation of rolling historical features (moving averages) to capture long-term structural fatigue.
- **Model Interpretability:** Utilization of SHAP values to uncover physical sensor degradation patterns.
- **Operational Classification:** Translation of numerical RUL into a 3-tier safety health semaphoring system (Healthy / Warning / Critical).

## 🛠️ Repository Structure
- `src/data_preprocessing.py`: Data cleaning and moving average feature engineering.
- `src/train_models.py`: XGBoost pipelines for Regression (RUL) and Classification (Health State).
- `notebooks/`: Exploratory data analysis, hyperparameter tuning (`RandomizedSearchCV`), and SHAP evaluations.

## 📊 Methodology & Key Results
The XGBoost Regressor was optimized using `RandomizedSearchCV` (`max_depth=3`, `subsample=0.8`, `learning_rate=0.05`) to prevent overfitting on early-life engine noise.
- **Optimized RMSE:** 33.73 cycles.

SHAP analysis proved the model relies heavily on historical moving averages of core temperatures and pressures (Sensors 4, 3, and 21), validating that the algorithm successfully learned underlying thermodynamic fatigue physics.
