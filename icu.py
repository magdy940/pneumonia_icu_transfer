# ============================================================
# 🧠 Pneumonia ICU Transfer Prediction App
# ============================================================
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# 1️⃣ Load model and feature list
# ============================================================
model = joblib.load("icu_transfer_model.joblib")

# نحاول تحميل أسماء الأعمدة
try:
    feature_names = joblib.load("icu_model_features.joblib")
except:
    feature_names = [
        'age', 'spo2', 'rr', 'hr', 'sbp', 'temp', 'gcs',
        'comorbidity_flag', 'sex_M', 'map',
        'spo2_rr_interaction', 'age_squared'
    ]

# ============================================================
# 2️⃣ Function to preprocess input
# ============================================================
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    
    # اشتقاق نفس المتغيرات اللي اتدرب عليها الموديل
    df['map'] = df['sbp']
    df['spo2_rr_interaction'] = df['spo2'] * df['rr']
    df['age_squared'] = df['age'] ** 2

    # التأكد من تطابق الأعمدة
    df = df.reindex(columns=feature_names, fill_value=0)
    return df

# ============================================================
# 3️⃣ Streamlit UI Design
# ============================================================
st.set_page_config(page_title="Pneumonia ICU Transfer Prediction", page_icon="🫁", layout="centered")

st.title("🫁 Pneumonia ICU Transfer Prediction App")
st.markdown("""
Enter patient data to estimate the **probability of ICU transfer** during hospital stay.
---
""")

# ============================================================
# 4️⃣ Input form
# ============================================================
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.number_input("🆔 Patient ID", min_value=1, step=1)
        age = st.number_input("🎂 Age (years)", min_value=0, max_value=120, value=50)
        sex = st.selectbox("⚧ Sex", ["Male", "Female"])
        spo2 = st.number_input("🩸 SpO₂ (%)", min_value=50.0, max_value=100.0, value=95.0)
        rr = st.number_input("🌬️ Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=20)
        hr = st.number_input("❤️ Heart Rate (bpm)", min_value=30, max_value=200, value=85)
    with col2:
        sbp = st.number_input("🩺 Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
        temp = st.number_input("🌡️ Temperature (°C)", min_value=30.0, max_value=42.0, value=37.0)
        gcs = st.number_input("🧠 GCS Score", min_value=3, max_value=15, value=15)
        comorbidity_flag = st.selectbox("🩻 Comorbidity", ["No", "Yes"])
        admission_id = st.number_input("🏥 Admission ID", min_value=1, step=1)

    submitted = st.form_submit_button("🔍 Predict ICU Transfer Probability")

# ============================================================
# 5️⃣ Prediction logic
# ============================================================
if submitted:
    comorbidity_flag_val = 1 if comorbidity_flag == "Yes" else 0

    patient_data = {
        'patient_id': patient_id,
        'admission_id': admission_id,
        'age': age,
        'sex': sex,
        'spo2': spo2,
        'rr': rr,
        'hr': hr,
        'sbp': sbp,
        'temp': temp,
        'gcs': gcs,
        'comorbidity_flag': comorbidity_flag_val
    }

    try:
        X_input = preprocess_input(patient_data)
        probability = model.predict_proba(X_input)[0][1] * 100

        st.success(f"🧾 Predicted Probability of ICU Transfer: **{probability:.2f}%**")

        if probability >= 70:
            st.error("⚠️ High Risk: Consider ICU monitoring.")
        elif probability >= 40:
            st.warning("🟠 Moderate Risk: Monitor closely.")
        else:
            st.info("🟢 Low Risk: Routine care appropriate.")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

