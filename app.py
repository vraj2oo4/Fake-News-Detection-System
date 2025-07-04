import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved models
with open('LogisticRegression.pkl', 'rb') as file:
    log_reg_model = pickle.load(file)

with open('DecisionTree.pkl', 'rb') as file:
    dt_model = pickle.load(file)

with open('RFC.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# with open('SVM.pkl', 'rb') as file:
#     nb_model = pickle.load(file)

# Load TF-IDF vectorizer (Assuming it was saved separately)
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Load sample training data (Assuming it's saved as a CSV)
train_data = pd.read_csv("manual_testing.csv")  # Modify with the correct file path
train_data_sample = train_data.sample(10)  # Take a small sample for initial graph

# Transform training data using the vectorizer
x_train_transformed = tfidf_vectorizer.transform(train_data_sample['text'])

# Get initial predictions for training data
train_data_sample['Logistic Regression'] = log_reg_model.predict(x_train_transformed)
train_data_sample['Decision Tree'] = dt_model.predict(x_train_transformed)
train_data_sample['Random Forest'] = rf_model.predict(x_train_transformed)
# train_data_sample['Support Vector Machine'] = nb_model.predict(x_train_transformed)
# Convert predictions to labels
label_map = {0: "Fake News", 1: "Real News"}
for model in ["Logistic Regression", "Decision Tree", "Random Forest"]:
    train_data_sample[model] = train_data_sample[model].map(label_map)

# Initialize session state for storing history
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection System")
st.markdown("### Enter a news article to predict whether it's Fake or Real")

user_input = st.text_area("Enter the news article text:", height=150)

if st.button("🔍 Predict"):
    if user_input:
        # Transform input text using the vectorizer
        input_data = tfidf_vectorizer.transform([user_input])
        
        # Get predictions from each model
        log_reg_pred = log_reg_model.predict(input_data)[0]
        dt_pred = dt_model.predict(input_data)[0]
        rf_pred = rf_model.predict(input_data)[0]
        # nb_pred = nb_model.predict(input_data)[0]
        new_prediction = {
            "Input": user_input[:50] + "...",
            "Logistic Regression": label_map[log_reg_pred],
            "Decision Tree": label_map[dt_pred],
            "Random Forest": label_map[rf_pred],
            # "Support Vector Machine" :  label_map[nb_pred]
        }
        
        # Store the prediction in session state
        st.session_state.predictions.append(new_prediction)
        st.write("### Latest Prediction")
        st.write(new_prediction)

# Combine training and user predictions for graph
combined_predictions = train_data_sample[['text', 'Logistic Regression', 'Decision Tree', 'Random Forest']].rename(columns={'text': 'Input'})
if st.session_state.predictions:
    user_df = pd.DataFrame(st.session_state.predictions)
    combined_predictions = pd.concat([combined_predictions, user_df], ignore_index=True)

# Plot predictions using Plotly
plot_data = []
model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]
for _, entry in combined_predictions.iterrows():
    for model in model_names:
        plot_data.append({"Input": entry["Input"], "Model": model, "Prediction": 1 if entry[model] == "Real News" else 0})

plot_df = pd.DataFrame(plot_data)
fig = px.bar(plot_df, x="Input", y="Prediction", color="Model",
             barmode="group", title="Model Predictions Over Time",
             labels={"Prediction": "Prediction (1 = Real, 0 = Fake)", "Input": "News Article"})

st.write("### Initial Prediction Visualization")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("This Fake News Detection System analyzes text using three machine learning models: Logistic Regression, Decision Tree, and Random Forest.")