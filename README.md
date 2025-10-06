# pneumonia_icu_transfer

📋 Overview

This web application predicts the probability of ICU transfer among pneumonia patients during their hospital stay.
It was developed using Python, Streamlit, and a trained logistic regression model on real hospital data to assist clinicians in making timely and informed decisions.

⚙️ Features

Manual entry of patient data (Age, SpO₂, HR, BP, etc.)

Real-time prediction of ICU transfer risk

Clean and professional user interface built with Streamlit

Model loaded directly using joblib

Outputs prediction as a percentage probability

🧾 Input Variables
Variable	Description
patient_id	Unique patient identifier
admission_id	Unique hospital admission ID
age	Age in years
sex	Male/Female
spo2	Oxygen saturation (%)
rr	Respiratory rate (breaths/min)
hr	Heart rate (bpm)
sbp	Systolic blood pressure (mmHg)
temp	Body temperature (°C)
gcs	Glasgow Coma Scale score
comorbidity_flag	Presence of comorbidity (Yes/No)
📊 Output

The app returns the probability of ICU transfer (0–100%) for each patient based on the provided clinical data.

How to Run Locally

Clone this repository:](https://pneumonia-icu-predict.streamlit.app/)
