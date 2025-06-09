import streamlit as st
import pandas as pd
import joblib

model, feature_columns = joblib.load("rf_churn_model.joblib")

st.title("📱 Expresso Churn Prediction App")

col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("REGION", ["Dakar", "Thies", "Other"])
    montant = st.number_input("MONTANT", min_value=0, value=470000)
    frequence_rech = st.number_input("FREQUENCE_RECH", min_value=1, max_value=133)
    revenue = st.number_input("REVENUE", min_value=1, max_value=532177)
    arpu_segment = st.number_input("ARPU_SEGMENT", min_value=0, max_value=177392)
    frequence = st.number_input("FREQUENCE", min_value=1, max_value=91)
    data_volume = st.number_input("DATA_VOLUME", min_value=0, max_value=1823866)

with col2:
    on_net = st.number_input("ON_NET", min_value=0, max_value=50809, value=10)
    orange = st.number_input("ORANGE", min_value=0, max_value=21323)
    tigo = st.number_input("TIGO", min_value=0, max_value=4174)
    zone1 = st.number_input("ZONE1", min_value=0, max_value=4792)
    zone2 = st.number_input("ZONE2", min_value=0, max_value=3697)
    mrg = st.selectbox("MRG", options=["NO", "YES"])
    regularity = st.number_input("REGULARITY", min_value=0, max_value=62)
    freq_top_pack = st.number_input("FREQ_TOP_PACK", min_value=1, max_value=713)

# Input data dictionary (excluding user_id, TOP_PACK, TENURE)
input_data = {
    'REGION': region,
    'MONTANT': montant,
    'FREQUENCE_RECH': frequence_rech,
    'REVENUE': revenue,
    'ARPU_SEGMENT': arpu_segment,
    'FREQUENCE': frequence,
    'DATA_VOLUME': data_volume,
    'ON_NET': on_net,
    'ORANGE': orange,
    'TIGO': tigo,
    'ZONE1': zone1,
    'ZONE2': zone2,
    'MRG': mrg,
    'REGULARITY': regularity,
    'FREQ_TOP_PACK': freq_top_pack
}

if st.button("Predict Churn"):
    input_df = pd.DataFrame([input_data])
    
    # Ensure same column order as training
    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 This customer is likely to churn. (Probability: {prob:.2%})")
    else:
        st.success(f"✅ This customer is likely to stay. (Probability: {prob:.2%})")
