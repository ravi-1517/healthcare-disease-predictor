# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Health Predictor Pro", layout="wide", page_icon="🩺")

# ------------------ SIDEBAR ------------------

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966483.png", width=120)
    st.title("🩺 Health Predictor Pro")
    st.markdown(
        """
        **Developed by:** Tharu Paranagama  
        **Powered by:** Python · Streamlit · Machine Learning  

        💡 *An AI-powered tool that predicts diabetes risk based on medical indicators.*

        ---
        **Connect with me:**  
        🔗 [LinkedIn](https://linkedin.com)  
        📧 tharu@example.com
        """
    )
    # use checkbox instead of non-existing st.toggle
    dark_mode = st.checkbox("🌗 Enable Dark Mode", value=False)

# ------------------ APPLY DARK/LIGHT THEME (simple CSS) ------------------
if dark_mode:
    st.markdown(
        """
        <style>
        /* page background and text */
        .reportview-container, .main {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* sidebar background */
        .css-1d391kg { background-color: #0B0D0F; }
        /* widget labels and values */
        .st-bx { color: #FAFAFA; }
        /* headers */
        h1, h2, h3, h4, h5, h6 { color: #FAFAFA; }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .reportview-container, .main {
            background-color: #FFFFFF;
            color: #0E1117;
        }
        .css-1d391kg { background-color: #F7F7F8; }
        h1, h2, h3, h4, h5, h6 { color: #0E1117; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------ DATA LOADING ------------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

# ------------------ MODEL TRAINING ------------------
@st.cache_resource
def train_model(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# ------------------ MAIN UI ------------------
st.markdown("<h1 style='text-align:center; color:#4BB543;'>Health Predictor Pro 🧠</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI-based Diabetes Risk Assessment</h4>", unsafe_allow_html=True)
st.markdown("---")

df = load_data()
model, acc = train_model(df)

# Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Predict Risk", "📊 Data Insights", "📈 Model Info"])

# ------------------ TAB 1: Prediction ------------------
with tab1:
    st.subheader("Enter Your Health Information")

    # Use Streamlit form properly
    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=0)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=0)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=0)
        with col2:
            insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900, value=0)
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=100)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=30)
            age = st.number_input("Age", min_value=1, max_value=120, value=1)

        # ✅ Correctly placed submit button inside form
        submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

    # --- When user submits ---
    if submitted:
        # Simple validation
        if glucose == 0 or height == 100 or weight == 30:
            st.warning("⚠️ Please fill in all fields with realistic values before predicting.")
        else:
            bmi = round(weight / ((height / 100) ** 2), 2)
            st.info(f"✅ Calculated BMI: **{bmi}**")

            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, 0.5, age]])
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.error(f"⚠️ High Risk of Diabetes (Probability: {proba:.2f})")
            else:
                st.success(f"✅ Low Risk of Diabetes (Probability: {proba:.2f})")


# ------------------ TAB 2: Data Insights ------------------
with tab2:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.markdown("#### Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm")
    plt.colorbar(im)
    st.pyplot(fig)

    st.markdown("#### Outcome Distribution")
    st.bar_chart(df['Outcome'].value_counts())

# ------------------ TAB 3: Model Info ------------------
with tab3:
    st.subheader("Model Details")
    st.write(f"🔹 Model Type: Random Forest Classifier")
    st.write(f"🔹 Accuracy: **{acc*100:.2f}%**")
    st.write("🔹 Dataset: Pima Indians Diabetes Database")
    st.write("This AI model uses key medical factors such as glucose, insulin, BMI, and age to predict the likelihood of diabetes.")
    st.info("Note: This is a predictive model for educational purposes — not a medical diagnosis tool.")


