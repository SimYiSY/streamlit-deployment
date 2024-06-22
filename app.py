# app.py
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the California Housing dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='MedHouseVal')

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('California Housing Price Prediction')

st.write("""
This app uses a Linear Regression model to predict the median house value in California based on various features.
""")

st.write('---')

# User input features
st.sidebar.header('Specify Input Parameters')
def user_input_features():
    MedInc = st.sidebar.slider('MedInc', float(X.MedInc.min()), float(X.MedInc.max()), float(X.MedInc.mean()))
    HouseAge = st.sidebar.slider('HouseAge', float(X.HouseAge.min()), float(X.HouseAge.max()), float(X.HouseAge.mean()))
    AveRooms = st.sidebar.slider('AveRooms', float(X.AveRooms.min()), float(X.AveRooms.max()), float(X.AveRooms.mean()))
    AveBedrms = st.sidebar.slider('AveBedrms', float(X.AveBedrms.min()), float(X.AveBedrms.max()), float(X.AveBedrms.mean()))
    Population = st.sidebar.slider('Population', float(X.Population.min()), float(X.Population.max()), float(X.Population.mean()))
    AveOccup = st.sidebar.slider('AveOccup', float(X.AveOccup.min()), float(X.AveOccup.max()), float(X.AveOccup.mean()))
    Latitude = st.sidebar.slider('Latitude', float(X.Latitude.min()), float(X.Latitude.max()), float(X.Latitude.mean()))
    Longitude = st.sidebar.slider('Longitude', float(X.Longitude.min()), float(X.Longitude.max()), float(X.Longitude.mean()))
    data = {'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input features
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Predict the price
prediction = model.predict(df)
st.header('Prediction of Median House Value')
st.write(prediction)
