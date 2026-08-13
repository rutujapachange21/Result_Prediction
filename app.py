import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page configuration with a wide layout and title
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("modelnb.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ 'modelnb.pkl' not found. Please place the model file in the app directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Header Section
st.title("🎓 Student Academic Result Predictor")
st.markdown("""
Predict whether a student is likely to **Pass** or **Fail** based on demographic info, academic habits, and assessment scores.
""")
st.divider()

if model is not None:
    # Form layout with 2 columns
    with st.form("prediction_form"):
        st.subheader("📋 Student Information Input")
        
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### 👤 Demographic Info")
            age = st.number_input("Age", min_value=15, max_value=60, value=20, step=1)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            
            # Map categorical gender to numeric if required by your training pipeline encoding
            # Default encoding mapping (Adjust if your pipeline used specific LabelEncoding/OneHot)
            gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
            gender_val = gender_mapping[gender]

            department = st.selectbox("Department", options=["Computer Science", "Engineering", "Business", "Arts", "Science"])
            # Map categorical department to numeric if needed
            dept_mapping = {"Computer Science": 0, "Engineering": 1, "Business": 2, "Arts": 3, "Science": 4}
            dept_val = dept_mapping[department]

            study_hours = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, value=4.5, step=0.5)

        with col2:
            st.markdown("### 📊 Academic Performance")
            attendance = st.slider("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
            assignments = st.number_input("Assignments Completed", min_value=0, max_value=50, value=10, step=1)
            midterm_score = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
            final_score = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=75.0, step=0.5)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🚀 Run Prediction")

    # Handle Form Submission
    if submit_btn:
        # Construct DataFrame matching model features precisely
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender_val,
            "Department": dept_val,
            "Study_Hours_Per_Day": study_hours,
            "Attendance_Percentage": attendance,
            "Assignments_Completed": assignments,
            "Midterm_Score": midterm_score,
            "Final_Score": final_score
        }])

        try:
            prediction = model.predict(input_data)[0]
            
            # Extract confidence probabilities if supported
            probabilities = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None

            st.divider()
            st.subheader("🎯 Prediction Result")

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if prediction == "Pass":
                    st.success("### Status: 🎉 PASS")
                else:
                    st.error("### Status: ⚠️ FAIL")

            with res_col2:
                if probabilities is not None:
                    class_labels = list(model.classes_)
                    fail_idx = class_labels.index("Fail") if "Fail" in class_labels else 0
                    pass_idx = class_labels.index("Pass") if "Pass" in class_labels else 1

                    st.write(f"**Pass Probability:** {probabilities[pass_idx]*100:.2f}%")
                    st.progress(float(probabilities[pass_idx]))
                    
                    st.write(f"**Fail Probability:** {probabilities[fail_idx]*100:.2f}%")
                    st.progress(float(probabilities[fail_idx]))

        except Exception as err:
            st.error(f"Prediction failed. Error: {err}")
            st.info("Tip: Ensure encoding (categorical variables like Gender & Department) matches how your model was trained.")

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ App Info")
    st.markdown("""
    * **Algorithm:** Gaussian Naive Bayes (`GaussianNB`)
    * **Target Output:** Pass / Fail
    * **Features Used:** 8 Input Parameters
    """)
    st.info("Ensure `modelnb.pkl` is located in the same directory as `app.py`.")
