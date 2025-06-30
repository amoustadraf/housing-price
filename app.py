import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Load your model
model = joblib.load("XGBRegressor_model.pkl")  # Save your model using joblib.dump() beforehand
model_features = joblib.load("model_features.pkl")  # This must be saved from training: joblib.dump(X_train.columns, "model_features.pkl")

st.title("Ames Housing Price Predictor")
st.markdown("Enter house characteristics below to estimate the sale price.")

# === USER INPUTS ===
# Basic numeric inputs
st.header("General Information")
overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
house_age = st.number_input("House Age (years)", min_value=0, value=20)
since_remod = st.number_input("Years Since Last Remodel", min_value=0, value=10)
ms_subclass = st.selectbox("Building class/type", ["20", "30", "40", "45", "50", "60", "70", "75", "80", "85", "90", "120", "150", "160", "180", "190"], index=0)
st.caption(
    "20: 1-Story 1946+, "
    "30: 1-Story pre-1945, "
    "40: 1-Story w/ Finished Attic, "
    "45: 1½ Story Unfinished Upstairs, "
    "50: 1½ Story Finished Upstairs, "
    "60: 2-Story 1946+, "
    "70: 2-Story pre-1945, "
    "75: 2½ Story All Ages, "
    "80: Split/Multi-Level, "
    "85: Split Foyer, "
    "90: Duplex, "
    "120: 1-Story PUD, "
    "150: 2-Story PUD, "
    "160: PUD Multi-Level, "
    "180: PUD Duplex, "
    "190: Conversion (e.g., single-family to apartments)"
)
ms_zoning = st.selectbox("Municipal Zoning classification ", ["RL", "RM", "RH", "FV", "C (all)", "I (all)", "A (agr)"], index=0)
st.caption("RL = Residential Low, RM = Medium, RH = High, FV = Floating Village, C (all) = Commercial, I (all) = Industrial, A (agr) = Agricultural")

st.header("Interior")
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=100, value=1500)
first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=100, value=1200)
full_bath = st.slider("Full Bathrooms", 0, 4, 2)
bsmt_full_bath = st.slider("Basement Full Bathrooms", 0, 2, 0)
kitchen_abvgr = st.selectbox("Number of Kitchens", [0, 1, 2], index=1)
kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
fireplaces = st.slider("Number of Fireplaces", 0, 3, 1)

st.header("Garage & Basement")
garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
garage_finish = st.selectbox("Garage Finish", ["Fin", "RFn", "Unf"], index=2)
st.caption("Fin = Finished, RFn = Rough Finished, Unf = Unfinished")
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, value=800)

st.header("Exterior & Utilities")
exter_cond = st.selectbox("Exterior Condition", ["Ex", "Gd", "TA", "Fa", "Po"], index=3)
central_air = st.radio("Central Air Conditioning", ["Yes", "No"], index=0)
paved_drive = st.radio("Paved Driveway", ["Yes", "No"], index=0)


# === CONSTRUCT INPUT DATAFRAME ===
input_data = pd.DataFrame([[0]*len(model_features)], columns=model_features)

# Set numeric/log-transformed features
input_data.loc[0, "Overall Qual"] = overall_qual
input_data.loc[0, "House Age"] = house_age
input_data.loc[0, "Since Remod"] = since_remod
input_data.loc[0, "Gr Liv Area"] = np.log1p(gr_liv_area)
input_data.loc[0, "1st Flr SF"] = np.log1p(first_flr_sf)
input_data.loc[0, "Total Bsmt SF"] = np.log1p(total_bsmt_sf)
input_data.loc[0, "Full Bath"] = full_bath
input_data.loc[0, "Bsmt Full Bath"] = bsmt_full_bath
input_data.loc[0, "Kitchen AbvGr"] = kitchen_abvgr
input_data.loc[0, "Fireplaces"] = fireplaces
input_data.loc[0, "Garage Cars"] = garage_cars

# Safe dummy set: only if column exists
def safe_set_dummy(prefix, value):
    col = f"{prefix}_{value}"
    if col in model_features:
        input_data.loc[0, col] = 1

# Apply safe dummy assignments
safe_set_dummy("Kitchen Qual", kitchen_qual)
safe_set_dummy("Exter Cond", exter_cond)
safe_set_dummy("Garage Finish", garage_finish)
safe_set_dummy("MS SubClass", ms_subclass)
safe_set_dummy("MS Zoning", ms_zoning)

if "Central Air_Y" in model_features and central_air == "Yes":
    input_data.loc[0, "Central Air_Y"] = 1

if "Paved Drive_Y" in model_features and paved_drive == "Yes":
    input_data.loc[0, "Paved Drive_Y"] = 1

# Predict
if st.button("Predict Sale Price"):
    log_price = model.predict(input_data)[0]
    price = np.expm1(log_price)  
    st.success(f"Estimated Sale Price: ${price:,.2f}")