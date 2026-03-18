
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import shap
import lime
import lime.lime_tabular

# --- Helper Functions ---

def train_model():
    """Trains a simple RandomForestClassifier on the Iris dataset."""
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, feature_names, class_names

def explain_with_shap(model, X_train, X_instance, feature_names):
    """Generates SHAP explanations for a given instance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)
    return shap_values

def explain_with_lime(model, X_train, X_instance, feature_names, class_names):
    """Generates LIME explanations for a given instance."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=X_instance,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    return explanation

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Explainable AI Dashboard")
st.title("🔬 Explainable AI Dashboard")
st.markdown("Understand your Machine Learning model predictions with SHAP and LIME.")

# Load and train model
model, X_train, X_test, y_train, y_test, feature_names, class_names = train_model()

st.sidebar.header("Configuration")
explanation_method = st.sidebar.selectbox(
    "Choose Explanation Method:",
    ("SHAP", "LIME")
)

st.sidebar.subheader("Select Data Instance")
instance_index = st.sidebar.slider(
    "Select an instance from the test set:",
    0, len(X_test) - 1, 0
)
X_instance = X_test[instance_index]
y_true = y_test[instance_index]
y_pred_proba = model.predict_proba(X_instance.reshape(1, -1))[0]
y_pred_class = np.argmax(y_pred_proba)

st.write(f"### Instance {instance_index} Details")
st.write(pd.DataFrame([X_instance], columns=feature_names))
st.write(f"**True Class**: {class_names[y_true]}")
st.write(f"**Predicted Class**: {class_names[y_pred_class]} (Probability: {y_pred_proba[y_pred_class]:.2f})")

st.write(f"### {explanation_method} Explanation")

if explanation_method == "SHAP":
    shap_values = explain_with_shap(model, X_train, X_instance, feature_names)
    st.subheader("SHAP Feature Importance")
    # For multi-class, SHAP returns a list of arrays. Let's focus on the predicted class.
    if isinstance(shap_values, list):
        shap_values_for_pred_class = shap_values[y_pred_class]
    else:
        shap_values_for_pred_class = shap_values

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values_for_pred_class
    }).sort_values(by="SHAP Value", ascending=False)
    st.dataframe(shap_df)
    
    # More advanced SHAP plots would require matplotlib/plotly integration
    # For simplicity, we'll just show the table.

elif explanation_method == "LIME":
    explanation = explain_with_lime(model, X_train, X_instance, feature_names, class_names)
    st.subheader("LIME Local Explanation")
    st.write("Features contributing to the prediction:")
    st.json(explanation.as_list())

# Ensure this file has 100+ lines of functional code.
