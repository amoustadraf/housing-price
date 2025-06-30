# 🏡 Ames Housing Price Predictor

This project builds a regression model to **predict housing prices** using the Ames Housing dataset. It includes:

- Data preprocessing
- Feature engineering
- Model training with XGBoost
- Evaluation with RMSE and R²
- A Streamlit web app for real-time predictions

---

## 📊 Data Overview

- **Dataset**: Ames Housing Dataset ([Link to Kaggle Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset))
- **Rows**: ~2,900
- **Target**: `SalePrice` (log-transformed for modeling)
- **Features**: 80+ features including lot size, year built, number of bathrooms, and more

---

## ⚙️ Preprocessing

- Handled missing values
- Encoded categorical features (One-Hot and binary flags)
- Log-transformed skewed features
- Optimized types & memory usage

### Histograms (Before vs After)

| Before | After |
|--------|-------|
| ![Before](images/before_preprocessed_data_graphs.png) | ![After](images/preprocessed_data_graphs.png) |

---

## 🤖 Model

### Algorithm

**XGBoost Regressor**, manually tuned with early stopping.

```python
XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
```
---

## 📈 Results

| Metric   | Train   | Test    |
|----------|---------|---------|
| RMSE     | 0.0667  | 0.1291  |
| R² Score | 0.9734  | 0.8969  |

✅ Model generalizes well with minimal overfitting.

---

## 🚀 Streamlit App

Run it locally:

```bash
streamlit run app.py
```
Fill out a home’s details and get an instant price prediction.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```
Recommended: Python 3.10+

---

## 🧠 Key Takeaways

- Log-transforming SalePrice improves performance
- Early preprocessing avoids downstream bugs
- Monitoring both RMSE and R² is key to model evaluation
- A complete ML + UI setup makes this project portfolio-ready

---

## 🧭 Future Improvements

- Deploy app to Streamlit Cloud
- Add SHAP for model explainability
- Use Optuna or GridSearchCV for hyperparameter tuning

---

## 📘 License

This project is open-source and available under the [MIT License](LICENSE).
