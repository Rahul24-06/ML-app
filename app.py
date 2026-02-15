import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Heart Disease ML App", layout="wide")
st.title("Heart Disease Classification App")
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>
    Heart Disease Prediction Dashboard 
    </h1>
""", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_default_data():
    return pd.read_csv("heart.csv")
    
df = load_default_data()

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)
st.sidebar.markdown("---")
st.sidebar.write("Dataset Shape:", df.shape)
st.sidebar.write("Features:", df.shape[1] - 1)
st.sidebar.markdown("---")

st.sidebar.subheader("Download sample test CSV")
sample_test = df.sample(20, random_state=1)
csv_buffer = io.StringIO()
sample_test.to_csv(csv_buffer, index=False)

st.sidebar.download_button(
    label="Download Sample Test Data",
    data=csv_buffer.getvalue(),
    file_name="sample_test_heart.csv",
    mime="text/csv"
)
st.markdown("---")

st.sidebar.subheader("Upload test CSV for prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
st.markdown("---")


X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    elif name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    elif name == "Naive Bayes":
        return GaussianNB()
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss")

model = get_model(model_name)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader("Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("AUC", f"{auc:.4f}")
col3.metric("Precision", f"{precision:.4f}")
col4, col5, col6 = st.columns(3)
col4.metric("Recall", f"{recall:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")
st.markdown("---")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4,4))
ax.matshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)
st.markdown("---")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(test_data.head())

    test_scaled = scaler.transform(test_data.drop("target", axis=1))
    predictions = model.predict(test_scaled)

    test_data["Predicted Target"] = predictions
    st.write("Prediction Results:")
    st.dataframe(test_data.head())

    output_buffer = io.StringIO()
    test_data.to_csv(output_buffer, index=False)

    st.download_button(
        label="Download Predictions CSV",
        data=output_buffer.getvalue(),
        file_name="predictions.csv",
        mime="text/csv"
    )
