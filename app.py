import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('random_forest_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="🚢 Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction App")

# User inputs
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 1, 100, 25)
sibsp = st.number_input('No. of Siblings/Spouses Aboard', 0, 8, 0)
parch = st.number_input('No. of Parents/Children Aboard', 0, 6, 0)
fare = st.number_input('Fare ($)', 0.0, 600.0, 32.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Encoding inputs
sex_encoded = 1 if sex == 'male' else 0
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
embarked_encoded = embarked_mapping[embarked]

# Prediction
input_features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
prediction = model.predict(input_features)

# Output
if st.button('Predict Survival'):
    if prediction[0] == 1:
        st.success("🎉 Passenger would Survive!")
    else:
        st.error("💀 Passenger would NOT Survive.")

