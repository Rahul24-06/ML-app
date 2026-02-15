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
st.markdown(
    """
    <h4 style='text-align:center;color:gray;'>
    Developed by: RAHUL KHANNA D 
    2025AB05245
    </h4>
    """,
    unsafe_allow_html=True
)
st.info("Instructions: Please upload a CSV file from the sidebar OR select the default heart.csv dataset before proceeding.")
st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

default_df = load_data()

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
data_option = st.sidebar.radio(
    "Choose Dataset",
    ["Use Default heart.csv", "Upload Custom CSV"]
)

uploaded_file = None
df = None

if data_option == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    df = default_df

if df is None:
    st.stop()

st.sidebar.write("Dataset Shape:", df.shape)
st.sidebar.write("Features:", df.shape[1] - 1)
st.sidebar.markdown("---")

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

col_left, col_middle, col_right = st.columns(3)

with col_left:
    st.subheader("Model Evaluation Metrics")
    st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("AUC", f"{auc:.4f}")
    st.metric("Precision", f"{precision:.4f}")
    st.metric("Recall", f"{recall:.4f}")
    st.metric("F1 Score", f"{f1:.4f}")
    st.metric("MCC", f"{mcc:.4f}")

with col_middle:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(2,2))
    ax.matshow(cm)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center",  fontsize=8)
    plt.tight_layout()  
    st.pyplot(fig)

with col_right:
    st.subheader("Predictions - 10 Rows")
    preview_df = df.copy()
    preview_df["Predicted"] = model.predict(scaler.transform(X))
    st.table(preview_df[["age", "sex", "Predicted"]].head(10))

    st.markdown("---")
    output_buffer = io.StringIO()
    preview_df.to_csv(output_buffer, index=False)

    st.download_button(
        label="Download prediction CSV",
        data=output_buffer.getvalue(),
        file_name="predicted_heart_disease.csv",
        mime="text/csv"
    )

st.markdown("---")

st.subheader("Prediction Analysis")
analysis_col1, analysis_col2 = st.columns(2)

#age
with analysis_col1:
    st.write("Prediction vs Age")
    temp_df = df.copy()
    temp_df["Predicted"] = model.predict(scaler.transform(X))
    fig_age, ax_age = plt.subplots()
    ax_age.scatter(temp_df["age"], temp_df["Predicted"], alpha=0.3)
    ax_age.set_xlabel("Age")
    ax_age.set_ylabel("Predicted Class")
    st.pyplot(fig_age)

#sex
with analysis_col2:
    st.write("Prediction vs Sex")
    temp_df = df.copy()
    temp_df["Predicted"] = model.predict(scaler.transform(X))
    temp_df["Sex_Label"] = temp_df["sex"].map({0: "F", 1: "M"})
    
    sex_counts = temp_df.groupby(["Sex_Label", "Predicted"]).size().unstack()
    fig_sex, ax_sex = plt.subplots()
    sex_counts.plot(kind="bar", ax=ax_sex)
    ax_sex.set_xlabel("Sex")
    ax_sex.set_ylabel("Count")
    ax_sex.legend(["No Disease (0)", "Disease (1)"])
    st.pyplot(fig_sex)

st.markdown("---")


