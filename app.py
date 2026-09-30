import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Phone Addiction Predictor",
    page_icon="📱",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("📱 Phone Addiction Level Predictor")

st.write(
    "Enter your daily habits and personal metrics "
    "to predict your phone addiction level."
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("phone_addiction_pipeline.pkl")


try:
    model = load_model()

except Exception as e:
    st.error("❌ Model loading failed.")
    st.exception(e)
    st.stop()


# ==========================================
# INPUT FORM
# ==========================================

with st.form("user_inputs"):

    st.subheader("👤 Personal Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        daily_usage = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=5.0)
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
        social_media = st.number_input("Time on Social Media (Hours)", min_value=0.0, max_value=24.0, value=2.0)
        gaming = st.number_input("Time on Gaming (Hours)", min_value=0.0, max_value=24.0, value=1.0)
        education = st.number_input("Time on Education (Hours)", min_value=0.0, max_value=24.0, value=1.5)
        purpose = st.selectbox("Usage Purpose", ["Browsing", "Education", "Gaming", "Social Media", "Other"])
        checks_per_day = st.number_input("Phone Checks Per Day", min_value=0, max_value=300, value=80)
        screen_before_bed = st.number_input("Screen Time Before Bed (Hours)", min_value=0.0, max_value=6.0, value=1.0)

    with col2:
        apps_used = st.number_input("Apps Used Daily", min_value=1, max_value=50, value=12)
        weekend_hours = st.number_input("Weekend Usage Hours", min_value=0.0, max_value=24.0, value=6.0)
        intellectual = st.slider("Intellectual Performance", 0, 100, 70)
        social_interactions = st.slider("Social Interactions Score", 0, 10, 5)
        exercise_hours = st.number_input("Exercise Hours", min_value=0.0, max_value=12.0, value=1.0)
        anxiety = st.slider("Anxiety Level", 0, 10, 3)
        depression = st.slider("Depression Level", 0, 10, 2)
        self_esteem = st.slider("Self Esteem Level", 0, 10, 7)
        family_comm = st.slider("Family Communication Score", 0, 10, 5)
    

    # ==========================================
    # PREDICT BUTTON
    # ==========================================

    submitted = st.form_submit_button(
        "🔮 Predict Addiction Level"
    )


# ==========================================
# PREDICTION
# ==========================================

if submitted:

    input_data = pd.DataFrame([{

        "Age": age,

        "Gender": gender,

        "Daily_Usage_Hours": daily_usage,

        "Sleep_Hours": sleep_hours,

        "Interllectual_Performance": intellectual,

        "Social_Interactions": social_interactions,

        "Exercise_Hours": exercise_hours,

        "Anxiety_Level": anxiety,

        "Depression_Level": depression,

        "Self_Esteem": self_esteem,

        "Screen_Time_Before_Bed": screen_before_bed,

        "Phone_Checks_Per_Day": checks_per_day,

        "Apps_Used_Daily": apps_used,

        "Time_on_Social_Media": social_media,

        "Time_on_Gaming": gaming,

        "Time_on_Education": education,

        "Phone_Usage_Purpose": purpose,

        "Family_Communication": family_comm,

        "Weekend_Usage_Hours": weekend_hours

    }])


    # Make prediction
    prediction = model.predict(input_data)[0]


    # Keep prediction between 0 and 10
    prediction = max(0, min(10, prediction))


    # ==========================================
    # DISPLAY RESULT
    # ==========================================

    st.divider()

    st.subheader("📊 Prediction Result")

    st.success(
        f"Predicted Addiction Level: **{prediction:.2f} / 10**"
    )


    # Interpretation

    if prediction < 3.5:

        st.info(
            "🟢 Low Phone Addiction Level"
        )

    elif prediction < 6.5:

        st.warning(
            "🟡 Moderate Phone Addiction Level"
        )

    else:

        st.error(
            "🔴 High Phone Addiction Level"
        )