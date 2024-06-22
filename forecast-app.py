# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import plotly.graph_objs as go
from datetime import datetime, timedelta
# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Set page config
apptitle = 'Air Passenger Forecast'

st.set_page_config(page_title=apptitle, layout='wide')

# Load the AirPassengers dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    return data

data = load_data()

# Convert Date format to year-month-day
data.loc[:, 'Date'] = data.index.strftime('%Y-%m-%d')

# Streamlit app
st.title('Air Passengers Forecasting with SVR and Plotly')

st.write("""
This app uses a Support Vector Regression (SVR) model to forecast the number of air passengers.
""")

st.write('---')

# Display the data
st.subheader('Air Passengers Data')
show_data = data.copy()
show_data.index = show_data['Date']
show_data = show_data.drop("Date", axis = 1)
st.write(show_data[:5])

# Plot the data using Plotly
st.subheader('Time Series Plot')

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Passengers'], mode='lines', name='Observed'))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Number of Passengers',
)
st.plotly_chart(fig)

# Splitting the data
train_data = data[:'1959']
test_data = data['1960':]

# Feature engineering function
def create_features(df, label=None):
    df['date'] = df.index
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    X = df[['month', 'year']]
    if label:
        y = df[label]
        return X, y
    return X

# Create features and target
X_train, y_train = create_features(train_data, label='Passengers')
X_test, y_test = create_features(test_data, label='Passengers')

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the SVR model
model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

# Train the SVR model
model.fit(X_train_scaled, y_train)

# Forecasting
st.write('---')
st.subheader('Forecasting')

# Create future dates for forecasting
forecast_periods = st.slider('Select number of months to forecast:', 1, 36, 12)
future_dates = [data.index[-1] + pd.DateOffset(months=x) for x in range(1, forecast_periods + 1)]

# Create features for future dates
future_X = create_features(pd.DataFrame(index=future_dates))

# Scale future features
future_X_scaled = scaler.transform(future_X)

# Make predictions
forecast = model.predict(future_X_scaled)

# Create a forecast dataframe
forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Forecast'])

# Plot the forecast using Plotly
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=data.index, y=data['Passengers'], mode='lines', name='Observed'))
fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='red')))
fig_forecast.update_layout(
    xaxis_title='Date',
    yaxis_title='Number of Passengers',
)
st.plotly_chart(fig_forecast)

# Display forecasted values
st.subheader('Forecasted Values')
forecast_df.loc[:, 'Date'] = forecast_df.index.strftime('%Y-%m-%d')
show_data = forecast_df.copy()
show_data.index = show_data['Date']
show_data = show_data.drop("Date", axis = 1)
st.write(show_data.T)
